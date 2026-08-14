"""Convert the official FireRedTTS3 fp32 weights into the mixed-precision bf16 mirror.

Policy (matches the nodepack's runtime numerics):
- fireredtts3_base / fireredtts3_instruct: tensors under `backbone_llm.` -> bf16, rest fp32
- redae: tensors under `encoder.` -> bf16, decoder stays fp32
- campp stays fp32; tokenizer/configs copied unchanged
- adds fasttext/lid.176.ftz to the mirror repo

Usage: python tools/convert_bf16.py <official_dir> <out_dir>
"""

import json
import shutil
import sys
import time
import urllib.request
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

FASTTEXT_URL = "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.ftz"


def convert_file(src: Path, dst: Path, policy) -> tuple[int, int]:
    converted = kept = 0
    tensors = {}
    with safe_open(str(src), framework="pt", device="cpu") as f:
        metadata = f.metadata()
        for key in f.keys():
            tensor = f.get_tensor(key)
            target = policy(key)
            if target is not None and tensor.is_floating_point() and tensor.dtype != target:
                tensor = tensor.to(target)
                converted += 1
            else:
                kept += 1
            tensors[key] = tensor.contiguous()
    save_file(tensors, str(dst), metadata=metadata)
    del tensors
    return converted, kept


def core_policy(name: str):
    return torch.bfloat16 if name.startswith("backbone_llm.") else torch.float32


def redae_policy(name: str):
    return torch.bfloat16 if name.startswith("encoder.") else torch.float32


def main() -> None:
    src_root = Path(sys.argv[1])
    dst_root = Path(sys.argv[2])
    if not src_root.is_dir():
        raise SystemExit(f"missing source dir: {src_root}")

    jobs = [
        (src_root / "fireredtts3_base" / "model.safetensors", dst_root / "fireredtts3_base", core_policy),
        (src_root / "fireredtts3_instruct" / "model.safetensors", dst_root / "fireredtts3_instruct", core_policy),
        (src_root / "redae" / "model.safetensors", dst_root / "redae", redae_policy),
    ]
    for src, dst_dir, policy in jobs:
        dst_dir.mkdir(parents=True, exist_ok=True)
        if not src.is_file():
            print(f"skip (not downloaded): {src.name} in {src.parent.name}")
            continue
        t0 = time.time()
        converted, kept = convert_file(src, dst_dir / "model.safetensors", policy)
        cfg_path = src.parent / "config.json"
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        cfg["dtype"] = "bfloat16"
        (dst_dir / "config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
        gb = (dst_dir / "model.safetensors").stat().st_size / 2**30
        print(f"{dst_dir.name}: {converted} bf16 / {kept} fp32 tensors -> {gb:.2f} GiB ({time.time() - t0:.0f}s)")

    for small in ("campp", "text_tokenizer"):
        src = src_root / small
        if src.is_dir():
            shutil.copytree(src, dst_root / small, dirs_exist_ok=True)
            print(f"copied {small}/")

    ft = dst_root / "fasttext" / "lid.176.ftz"
    if not ft.is_file():
        ft.parent.mkdir(parents=True, exist_ok=True)
        print("downloading lid.176.ftz ...")
        urllib.request.urlretrieve(FASTTEXT_URL, str(ft))
    print("DONE")


if __name__ == "__main__":
    main()
