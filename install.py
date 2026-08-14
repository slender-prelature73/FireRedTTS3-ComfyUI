"""Dependency check/install helper for FireRedTTS3-ComfyUI.

Never installs, upgrades, or removes torch / torchaudio / transformers / numpy
or any other heavyweight ComfyUI runtime package. Works with pip and uv.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys


# Hard requirements that must already be provided by the ComfyUI environment.
# They are checked, never installed by this helper.
CRITICAL_IMPORTS = ["torch", "torchaudio", "transformers", "numpy"]

# Lightweight, torch-free dependencies installed when missing.
LIGHTWEIGHT_IMPORTS = {
    "huggingface_hub": "huggingface-hub",
    "safetensors": "safetensors",
    "tokenizers": "tokenizers",
    "accelerate": "accelerate",
    "regex": "regex",
    "tqdm": "tqdm",
}

# Optional frontend goodies: zh/en text normalization and fasttext language-id.
# Failure to install these only disables optional features.
OPTIONAL_IMPORTS = {
    "wetext": "wetext",
    "fasttext_predict": "fasttext-predict",
}


def _missing(mapping: dict[str, str]) -> list[str]:
    return [package for module, package in mapping.items() if importlib.util.find_spec(module) is None]


def _install(packages: list[str]) -> int:
    """Install with pip when available, otherwise uv. Never touches torch et al."""
    if importlib.util.find_spec("pip") is not None:
        cmd = [sys.executable, "-m", "pip", "install", *packages]
    else:
        uv = shutil.which("uv")
        if uv is None:
            print("Neither pip nor uv is available in this environment; install manually:", ", ".join(packages))
            return 1
        cmd = [uv, "pip", "install", "--python", sys.executable, *packages]
    print("Running:", " ".join(cmd))
    return subprocess.call(cmd)


def main() -> int:
    missing_critical = [name for name in CRITICAL_IMPORTS if importlib.util.find_spec(name) is None]
    if missing_critical:
        print("Missing ComfyUI/runtime dependency:", ", ".join(missing_critical))
        print("Install this nodepack inside a working ComfyUI environment; this helper will not modify torch, torchaudio, transformers, or numpy.")
        return 1

    missing = _missing(LIGHTWEIGHT_IMPORTS)
    if missing:
        print("Installing missing lightweight dependencies:", ", ".join(missing))
        print("Torch/torchaudio/transformers are not modified by this installer.")
        if _install(missing) != 0:
            print("Failed to install required dependencies:", ", ".join(missing))
            return 1

    missing_optional = _missing(OPTIONAL_IMPORTS)
    if missing_optional:
        print("Installing optional frontend dependencies:", ", ".join(missing_optional))
        if _install(missing_optional) != 0:
            print("Optional dependencies could not be installed; text normalization / language auto-detect may be limited.")
            print("FireRedTTS3 will still run without them.")

    if not missing and not missing_optional:
        print("FireRedTTS3-ComfyUI dependencies are already present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
