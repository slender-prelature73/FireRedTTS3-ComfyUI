"""INT8 ConvRot tests: math, metadata, module replacement, device moves.

Runs without the FireRedTTS3 checkpoint (that coverage lives in
tools/validate_int8_convrot.py). Execute with pytest or `python tests/test_int8_convrot.py`.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import int8 as int8rt  # noqa: E402
from int8 import ConvRotInt8Linear  # noqa: E402

CUDA = torch.device("cuda") if torch.cuda.is_available() else None


def _hadamard(size):
    from comfy_kitchen.tensor.int8_utils import _build_hadamard

    return _build_hadamard(size, device="cpu", dtype=torch.float32)


def _reference_rotate_weight(w, h, g):
    n_groups = w.shape[1] // g
    return torch.matmul(w.reshape(w.shape[0], n_groups, g), h.T).reshape(w.shape)


def _reference_rotate_activation(x, h, g):
    n_groups = x.shape[-1] // g
    return torch.matmul(x.reshape(-1, n_groups, g), h).reshape(x.shape)


def test_convrot_hadamard_identity():
    for g in (4, 16, 64, 256, 1024):
        h = _hadamard(g)
        identity = h.T @ h
        assert torch.allclose(identity, torch.eye(g), atol=1e-5), g
        torch.manual_seed(g)
        x = torch.randn(8, g * 3)
        w = torch.randn(64, g * 3)
        y0 = x @ w.T
        y1 = _reference_rotate_activation(x, h, g) @ _reference_rotate_weight(w, h, g).T
        assert torch.allclose(y0, y1, rtol=1e-4, atol=1e-4), g


def test_convrot_group_validation():
    for bad in (0, 2, 8, 100, 512):
        try:
            int8rt.validate_group_size(bad, 256)
            raise AssertionError(f"group size {bad} should be rejected")
        except ValueError:
            pass
    for good in (4, 16, 64, 256, 1024):
        int8rt.validate_group_size(good, good * 4)
    try:
        int8rt.validate_group_size(256, 1000)
        raise AssertionError("non-divisible in_features should be rejected")
    except ValueError:
        pass
    assert math.log(256, 4) % 1 == 0


def test_convrot_quantize_dequantize():
    from comfy_kitchen.tensor import TensorWiseINT8Layout

    torch.manual_seed(1)
    for shape in [(512, 2048), (1024, 1024), (64, 512)]:
        w = torch.randn(*shape) * 0.02
        q, params = TensorWiseINT8Layout.quantize(
            w, is_weight=True, per_channel=True, convrot=True, convrot_groupsize=256, stochastic_rounding=0
        )
        assert q.dtype == torch.int8 and q.shape == w.shape
        assert params.scale.dtype == torch.float32 and params.scale.shape == (shape[0], 1)
        assert torch.isfinite(params.scale).all() and (params.scale > 0).all()
        w_dq = TensorWiseINT8Layout.dequantize(q, params)
        rel = ((w_dq - w).norm() / w.norm()).item()
        cos = F.cosine_similarity(w_dq.flatten(), w.flatten(), dim=0).item()
        assert rel <= 0.02, (shape, rel)
        assert cos >= 0.999, (shape, cos)


def test_convrot_linear_matches_fp():
    import comfy_kitchen
    from comfy_kitchen.tensor import TensorWiseINT8Layout

    torch.manual_seed(2)
    device = CUDA or torch.device("cpu")
    for n, k in [(1024, 2048), (3072, 1024)]:
        w = torch.randn(n, k, device=device) * 0.02
        x = torch.randn(32, k, device=device) * 0.5
        q, params = TensorWiseINT8Layout.quantize(
            w, is_weight=True, per_channel=True, convrot=True, convrot_groupsize=256, stochastic_rounding=0
        )
        y_ref = F.linear(x, w)
        y = comfy_kitchen.int8_linear(
            x, q, params.scale, None, out_dtype=x.dtype, convrot=True, convrot_groupsize=256
        )
        cos = F.cosine_similarity(y_ref.flatten(), y.flatten(), dim=0).item()
        rel = ((y_ref - y).norm() / y_ref.norm()).item()
        assert torch.isfinite(y).all()
        assert cos >= 0.995, (n, k, cos)
        assert rel <= 0.05, (n, k, rel)


def test_convrot_metadata_roundtrip():
    meta = {"format": "int8_tensorwise", "convrot": True, "convrot_groupsize": 256}
    tensor = torch.tensor(list(json.dumps(meta).encode("utf-8")), dtype=torch.uint8)
    parsed = json.loads(tensor.numpy().tobytes())
    assert parsed == meta

    layer = ConvRotInt8Linear(256, 64, bias=False, group_size=256)
    sd = {"weight": torch.zeros(64, 256, dtype=torch.int8),
          "weight_scale": torch.ones(64, 1),
          f"proj.{int8rt.QUANT_META_SUFFIX}": tensor}
    layer._load_from_state_dict(sd, "proj.", {}, True, [], [], [])
    # the metadata key is consumed; real tensors are copied into the module by torch
    assert f"proj.{int8rt.QUANT_META_SUFFIX}" not in sd
    assert layer.weight.dtype == torch.int8


def test_quantized_loader_replaces_correct_linears():
    model = nn.Sequential(nn.Linear(256, 512), nn.ReLU(), nn.Linear(512, 256))
    quant_map = {
        "0": int8rt.QuantLayerInfo("0", 256, 256, 512, True),
        "2": int8rt.QuantLayerInfo("2", 256, 512, 256, True),
    }
    replaced = int8rt.replace_quantized_linears(model, quant_map)
    assert set(replaced) == {"0", "2"}
    assert isinstance(model[0], ConvRotInt8Linear) and isinstance(model[2], ConvRotInt8Linear)
    assert model[0].weight.dtype == torch.int8 and model[0].weight_scale.dtype == torch.float32
    assert model[0].convrot_groupsize == 256

    try:
        int8rt.replace_quantized_linears(nn.Linear(4, 4), {"": int8rt.QuantLayerInfo("", 8, 8, 4, False)})
        raise AssertionError("shape mismatch should raise")
    except RuntimeError:
        pass


def test_quantized_loader_preserves_fp_layers():
    model = nn.Sequential(nn.Linear(256, 512), nn.ReLU(), nn.Linear(300, 256))
    quant_map = {"0": int8rt.QuantLayerInfo("0", 256, 256, 512, True)}
    int8rt.replace_quantized_linears(model, quant_map)
    assert isinstance(model[2], nn.Linear) and not isinstance(model[2], ConvRotInt8Linear)
    assert isinstance(model[1], nn.ReLU)


def test_scan_rejects_unknown_format(tmp_path=None):
    from safetensors.torch import save_file

    bad = torch.tensor(list(json.dumps({"format": "nvfp4", "convrot": True, "convrot_groupsize": 16}).encode("utf-8")), dtype=torch.uint8)
    d = Path(tmp_path) if tmp_path else Path(__file__).parent / "_tmp_scan"
    d.mkdir(exist_ok=True)
    save_file({"a.comfy_quant": bad}, str(d / "model.safetensors"))
    try:
        int8rt.scan_checkpoint_quantization(d)
        raise AssertionError("unknown format must be rejected")
    except RuntimeError:
        pass
    finally:
        if tmp_path is None:
            import shutil

            shutil.rmtree(d, ignore_errors=True)


def test_device_move_preserves_int8():
    if CUDA is None:
        print("skip: no cuda")
        return
    layer = ConvRotInt8Linear(256, 128, bias=True, group_size=256)
    with torch.no_grad():
        layer.weight.copy_(torch.randint(-127, 128, (128, 256), dtype=torch.int8))
        layer.weight_scale.copy_(torch.rand(128, 1) + 0.1)
        layer.bias.copy_(torch.randn(128))
    layer = layer.to(CUDA)
    assert layer.weight.dtype == torch.int8 and layer.weight.device.type == "cuda"
    assert layer.weight_scale.dtype == torch.float32
    y = layer(torch.randn(4, 256, device=CUDA, dtype=torch.float32))
    assert y.shape == (4, 128) and y.dtype == torch.float32 and torch.isfinite(y).all()
    layer = layer.cpu()
    assert layer.weight.dtype == torch.int8
    assert int8rt.RUNTIME_STATS["calls"] >= 1
    assert int8rt.RUNTIME_STATS["weight_dtype"] == torch.int8
    assert int8rt.RUNTIME_STATS["groupsize"] == 256


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"[PASS] {name}")
            except Exception as exc:
                failures += 1
                print(f"[FAIL] {name}: {exc}")
    raise SystemExit(1 if failures else 0)
