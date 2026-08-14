"""Convert FireRedTTS3 core weights to Comfy INT8 ConvRot.

Uses the official comfy-kitchen quantizer (TensorWiseINT8Layout -> registry
"quantize_int8_convrot_weight") with per-row scales and offline Hadamard weight
rotation; the runtime (int8.py ConvRotInt8Linear) applies the matching online
activation rotation via comfy_kitchen.int8_linear.

Safe profile quantizes only the transformer-block linears under
backbone_llm.layers.*, patch_encoder.blocks.*, and dit.blocks.* whose
in_features % group_size == 0. Everything else (embeddings, norms, boundary
projections, stop_head, Conv1d, RedAE, CAM++) is preserved exactly.

The source checkpoint is never modified.

Example:
    python tools/quantize_fireredtts3_int8_convrot.py ^
        --source .../models/fireredtts3/FireRedTeam_FireRedTTS3 ^
        --out    .../models/fireredtts3/FireRedTTS3-int8-convrot ^
        --variant fireredtts3_base --group-size 256 --device cuda
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

NODE_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(NODE_ROOT))

from int8 import QUANT_META_SUFFIX, SUPPORTED_FORMAT, validate_group_size  # noqa: E402

SAFE_PREFIXES = (
    "backbone_llm.layers.",        # base: Qwen3Model holds layers directly
    "backbone_llm.model.layers.",  # instruct: Qwen3ForCausalLM wraps them in .model
    "patch_encoder.blocks.",
    "dit.blocks.",
)
SHARED_DIRS = ("redae", "campp", "text_tokenizer")


def _versions() -> dict:
    import comfy_kitchen
    import transformers

    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "transformers": transformers.__version__,
        "comfy_kitchen": getattr(comfy_kitchen, "__version__", "unknown"),
    }


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_core(variant: str, model_dir: Path):
    """Build the architecture on the meta device for module enumeration only."""
    from accelerate import init_empty_weights

    import native

    config = native.read_config(model_dir / variant)
    with init_empty_weights():
        if variant == "fireredtts3_base":
            return native.FireRedTTS3BaseCore(config, "sdpa"), config
        return native.FireRedTTS3InstructCore(config, "sdpa"), config


@dataclass
class LayerEntry:
    name: str
    in_features: int
    out_features: int
    has_bias: bool
    param_count: int
    selected: bool = False
    reason: str = ""
    group_size: int = 0
    original_bytes: int = 0


def build_inventory(model, group_size: int, profile: str) -> tuple[list[LayerEntry], dict]:
    entries: list[LayerEntry] = []
    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        if profile == "safe":
            in_block = name.startswith(SAFE_PREFIXES)
            if not in_block:
                reason = "outside transformer blocks (safe profile keeps boundary/conditioning layers)"
            elif module.in_features % group_size != 0:
                reason = f"in_features {module.in_features} % group_size {group_size} != 0"
            else:
                reason = ""
        else:
            raise ValueError(f"unknown profile: {profile}")
        entries.append(LayerEntry(
            name=name,
            in_features=module.in_features,
            out_features=module.out_features,
            has_bias=module.bias is not None,
            param_count=module.in_features * module.out_features,
            selected=reason == "",
            reason=reason or "selected",
            group_size=group_size if reason == "" else 0,
            original_bytes=module.in_features * module.out_features * 4 + (module.out_features * 4 if module.bias is not None else 0),
        ))
    total_params = sum(p.numel() for p in model.parameters())
    linear_params = sum(e.param_count for e in entries)
    selected_params = sum(e.param_count for e in entries if e.selected)
    totals = {
        "profile": profile,
        "group_size": group_size,
        "total_model_parameters": total_params,
        "total_linear_parameters": linear_params,
        "selected_int8_parameters": selected_params,
        "selected_share_of_core_weights": selected_params / total_params,
        "estimated_original_bytes": sum(e.original_bytes for e in entries),
        "estimated_quantized_bytes": sum(
            e.param_count + e.out_features * 4 + 256 for e in entries if e.selected
        ) + sum(e.original_bytes for e in entries if not e.selected),
        "layers_quantized": sum(1 for e in entries if e.selected),
        "layers_bf16": sum(1 for e in entries if not e.selected),
    }
    return entries, totals


def iter_shards(model_dir: Path) -> list[Path]:
    index = model_dir / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
        return [model_dir / s for s in sorted(set(weight_map.values()))]
    return [model_dir / "model.safetensors"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="FireRedTTS3 repo dir (never modified)")
    parser.add_argument("--out", required=True, help="output repo dir")
    parser.add_argument("--variant", default="fireredtts3_base", choices=["fireredtts3_base", "fireredtts3_instruct"])
    parser.add_argument("--group-size", type=int, default=256)
    parser.add_argument("--profile", default="safe", choices=["safe"])
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    source = Path(args.source).resolve()
    out = Path(args.out).resolve()
    variant_dir = source / args.variant
    if not (variant_dir / "model.safetensors").is_file() and not (variant_dir / "model.safetensors.index.json").is_file():
        raise SystemExit(f"no checkpoint found in {variant_dir}")
    if out == source:
        raise SystemExit("output directory must differ from source")
    (out / args.variant).mkdir(parents=True, exist_ok=True)

    validate_group_size(args.group_size, 1 << 30)  # power-of-four check; divisibility is per-layer

    from comfy_kitchen.tensor import TensorWiseINT8Layout

    core, _config = build_core(args.variant, source)
    entries, totals = build_inventory(core, args.group_size, args.profile)
    selected = {e.name for e in entries if e.selected}
    print(f"inventory: {totals['layers_quantized']} linears selected / {len(entries)} total, "
          f"{totals['selected_int8_parameters'] / 1e9:.2f}B params "
          f"({totals['selected_share_of_core_weights'] * 100:.1f}% of core)")

    device = torch.device(args.device)
    out_tensors: dict[str, torch.Tensor] = {}
    quantized_names: list[str] = []
    t0 = time.time()
    for shard in iter_shards(variant_dir):
        with safe_open(str(shard), framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                base = key[:-len(".weight")] if key.endswith(".weight") else None
                if base in selected:
                    w = tensor.to(device=device, dtype=torch.float32)
                    qdata, params = TensorWiseINT8Layout.quantize(
                        w,
                        is_weight=True,
                        per_channel=True,
                        convrot=True,
                        convrot_groupsize=args.group_size,
                        stochastic_rounding=0,
                    )
                    scale = params.scale.float().cpu()
                    qdata = qdata.cpu()
                    assert qdata.dtype == torch.int8, qdata.dtype
                    assert qdata.shape == tensor.shape, (qdata.shape, tensor.shape)
                    assert scale.dtype == torch.float32 and scale.shape == (tensor.shape[0], 1), scale.shape
                    assert torch.isfinite(scale).all() and (scale > 0).all()
                    assert tensor.shape[1] % args.group_size == 0
                    out_tensors[key] = qdata.contiguous()
                    out_tensors[f"{base}.weight_scale"] = scale.contiguous()
                    meta = json.dumps({
                        "format": SUPPORTED_FORMAT,
                        "convrot": True,
                        "convrot_groupsize": args.group_size,
                        "in_features": int(tensor.shape[1]),
                        "out_features": int(tensor.shape[0]),
                        "has_bias": f"{base}.bias" in f.keys(),
                    }).encode("utf-8")
                    out_tensors[f"{base}.{QUANT_META_SUFFIX}"] = torch.tensor(list(meta), dtype=torch.uint8)
                    quantized_names.append(base)
                    del w
                else:
                    out_tensors[key] = tensor.contiguous()

    missing = selected - set(quantized_names)
    if missing:
        raise SystemExit(f"selected layers absent from checkpoint: {sorted(missing)[:5]}")

    out_file = out / args.variant / "model.safetensors"
    save_file(out_tensors, str(out_file))
    out_size = out_file.stat().st_size
    src_size = sum(s.stat().st_size for s in iter_shards(variant_dir))
    del out_tensors

    for shared in SHARED_DIRS:
        src_shared = source / shared
        if not src_shared.is_dir():
            continue
        dst_shared = out / shared
        dst_shared.mkdir(parents=True, exist_ok=True)
        for item in src_shared.iterdir():
            dst = dst_shared / item.name
            if dst.exists():
                continue
            try:
                dst.hardlink_to(item)
            except OSError:
                import shutil

                shutil.copy2(item, dst)
    src_cfg = variant_dir / "config.json"
    (out / args.variant / "config.json").write_text(src_cfg.read_text(encoding="utf-8"), encoding="utf-8")

    manifest = {
        "source": str(source),
        "variant": args.variant,
        "profile": args.profile,
        "group_size": args.group_size,
        "layers": [vars(e) for e in entries],
        "totals": totals,
    }
    (out / args.variant / "quantization_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    report = {
        "versions": _versions(),
        "source_files": {str(s.relative_to(source)): sha256_of(s) for s in iter_shards(variant_dir)},
        "output_files": {str(out_file.relative_to(out)): sha256_of(out_file)},
        "source_bytes": src_size,
        "output_bytes": out_size,
        "quantized_layers": quantized_names,
        "skipped_layers": [e.name for e in entries if not e.selected],
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    (out / args.variant / "conversion_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"quantized {len(quantized_names)} layers in {report['elapsed_seconds']}s")
    print(f"checkpoint: {src_size / 2**30:.2f} GiB -> {out_size / 2**30:.2f} GiB")
    print(f"output: {out_file}")


if __name__ == "__main__":
    main()
