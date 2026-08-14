"""FireRedTTS3 model loading, downloads, and ComfyUI/AIMDO memory registration."""

from __future__ import annotations

import gc
import importlib.util
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

from . import native
from .frontend import FASTTEXT_FILENAME, FASTTEXT_URL, FastTextLangDetector, TextFrontend
from .tokenizer import load_text_tokenizer

logger = logging.getLogger("FireRedTTS3-ComfyUI")

MODEL_FOLDER_NAME = "fireredtts3"
OFFICIAL_REPO_ID = "FireRedTeam/FireRedTTS3"
BF16_REPO_ID = "drbaph/FireRedTTS3-bf16"
OFFICIAL_REPO_LABEL = "FireRedTTS3 fp32 - FireRedTeam (auto-download)"
BF16_REPO_LABEL = "FireRedTTS3 bf16 - drbaph (auto-download)"
REPO_CHOICES = {BF16_REPO_LABEL: BF16_REPO_ID, OFFICIAL_REPO_LABEL: OFFICIAL_REPO_ID}
HF_ENDPOINT = "https://huggingface.co"

VARIANTS = ["fireredtts3_base", "fireredtts3_instruct"]
SHARED_PATTERNS = ["redae/*", "campp/*", "text_tokenizer/*"]
DTYPE_OPTIONS = ["auto", "bf16", "fp32"]
DEVICE_OPTIONS = ["auto", "cuda", "cpu"]
ATTENTION_OPTIONS = ["auto", "sdpa", "flash_attention", "sageattention"]

_ACTIVE_BUNDLE: "FireRedBundle | None" = None
_ACTIVE_LOAD_KEY: tuple[Any, ...] | None = None


@dataclass
class FireRedBundle:
    variant: str  # "base" or "instruct"
    core: torch.nn.Module
    redae: torch.nn.Module
    campp: torch.nn.Module | None
    tokenizer: Any
    frontend: Any
    model_dir: Path
    device: torch.device
    dtype_name: str
    attention: str
    patchers: list[Any] = field(default_factory=list)


try:
    import comfy.model_patcher as _model_patcher

    _ComfyCorePatcher = _model_patcher.CoreModelPatcher
    del _model_patcher
except Exception:
    _ComfyCorePatcher = None


def _empty_accelerator_cache() -> None:
    try:
        import comfy.model_management as mm

        mm.soft_empty_cache()
        return
    except Exception:
        pass
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if hasattr(torch, "xpu") and torch.xpu.is_available():
        torch.xpu.empty_cache()


def model_dir() -> Path:
    try:
        import folder_paths

        base = Path(folder_paths.models_dir) / MODEL_FOLDER_NAME
    except Exception:
        base = Path(__file__).resolve().parent / "models" / MODEL_FOLDER_NAME
    base.mkdir(parents=True, exist_ok=True)
    return base


def register_model_folder() -> None:
    try:
        import folder_paths

        base = str(model_dir())
        if MODEL_FOLDER_NAME not in folder_paths.folder_names_and_paths:
            folder_paths.add_model_folder_path(MODEL_FOLDER_NAME, base)
        logger.info("FireRedTTS3 model folder registered: %s", base)
    except Exception:
        pass


def _safe_repo_name(repo_id: str) -> str:
    return repo_id.replace("/", "_").replace("\\", "_").replace(":", "_")


def _has_component_files(path: Path, variant: str) -> bool:
    return (
        (path / variant / "model.safetensors").is_file()
        and (path / variant / "config.json").is_file()
        and (path / "redae" / "model.safetensors").is_file()
        and (path / "redae" / "config.json").is_file()
        and (path / "text_tokenizer" / "tokenizer.json").is_file()
        and (variant != "fireredtts3_base" or (path / "campp" / "campplus_voxceleb.bin").is_file())
    )


def get_repo_choices() -> list[str]:
    choices = list(REPO_CHOICES)
    try:
        for entry in sorted(model_dir().iterdir()):
            if entry.is_dir() and (entry / "redae").is_dir() and (entry / "text_tokenizer").is_dir():
                label = f"local: {entry.name}"
                if label not in choices:
                    choices.append(label)
    except OSError:
        pass
    return choices


def _resolve_repo_dir(repo_choice: str) -> tuple[Path, str | None]:
    """Returns (directory, repo_id or None for a local folder)."""
    if repo_choice in REPO_CHOICES:
        return model_dir() / _safe_repo_name(REPO_CHOICES[repo_choice]), REPO_CHOICES[repo_choice]
    if repo_choice.startswith("local: "):
        return model_dir() / repo_choice[len("local: "):], None
    # Bare folder name or repo id typed into the combo.
    candidate = model_dir() / repo_choice
    if candidate.is_dir():
        return candidate, None
    return model_dir() / _safe_repo_name(repo_choice), repo_choice


def _download_model_files(repo_id: str, variant: str, dest: Path) -> None:
    from huggingface_hub import snapshot_download

    logger.info("Downloading FireRedTTS3 %s weights from %s to %s. This is a large download.", variant, repo_id, dest)
    dest.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(dest),
        allow_patterns=SHARED_PATTERNS + [f"{variant}/*"],
        endpoint=HF_ENDPOINT,
    )


def resolve_model_dir(repo_choice: str, variant: str, download_if_missing: bool) -> Path:
    repo_dir, repo_id = _resolve_repo_dir(repo_choice)
    if not _has_component_files(repo_dir, variant):
        if repo_id is None:
            raise FileNotFoundError(f"Local FireRedTTS3 folder is missing {variant}/redae/tokenizer files: {repo_dir}")
        if not download_if_missing:
            raise FileNotFoundError(
                f"FireRedTTS3 files for {variant} not found in {repo_dir}. Enable download_if_missing."
            )
        _download_model_files(repo_id, variant, repo_dir)
    if not _has_component_files(repo_dir, variant):
        raise RuntimeError(f"Download finished but FireRedTTS3 files are still incomplete in {repo_dir}.")
    return repo_dir


def ensure_fasttext_model(download_if_missing: bool) -> Path | None:
    dest = model_dir() / "fasttext" / FASTTEXT_FILENAME
    if dest.is_file():
        return dest
    if not download_if_missing:
        return None
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        from huggingface_hub import hf_hub_download

        hf_hub_download(repo_id=BF16_REPO_ID, filename=f"fasttext/{FASTTEXT_FILENAME}",
                        local_dir=str(model_dir()), endpoint=HF_ENDPOINT)
    except Exception:
        try:
            import urllib.request

            logger.info("Downloading FastText lid.176 from %s", FASTTEXT_URL)
            urllib.request.urlretrieve(FASTTEXT_URL, str(dest))
        except Exception as exc:
            logger.warning("FastText lid.176 download failed; auto language detection falls back to heuristics: %s", exc)
            return None
    return dest if dest.is_file() else None


def resolve_device(device_name: str) -> torch.device:
    if device_name == "auto":
        try:
            import comfy.model_management as mm

            return torch.device(mm.get_torch_device())
        except Exception:
            pass
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was selected, but torch.cuda is not available.")
    return device


def resolve_dtype_mode(dtype_name: str, device: torch.device) -> str:
    """Returns 'bf16' (mixed official precision) or 'fp32' (full precision)."""
    if device.type == "cpu":
        if dtype_name == "bf16":
            logger.warning("bf16 is not practical on CPU; using fp32.")
        return "fp32"
    if dtype_name == "auto":
        if device.type == "cuda" and torch.cuda.is_available():
            try:
                return "bf16" if torch.cuda.is_bf16_supported() else "fp32"
            except Exception:
                return "fp32"
        return "bf16"
    if dtype_name == "bf16":
        if device.type == "cuda" and torch.cuda.is_available():
            try:
                if not torch.cuda.is_bf16_supported():
                    raise RuntimeError("bf16 was selected, but this CUDA device does not report bf16 support. Use dtype=auto.")
            except RuntimeError:
                raise
            except Exception:
                pass
        return "bf16"
    if dtype_name == "fp32":
        return "fp32"
    raise ValueError(f"Unsupported dtype: {dtype_name}")


def resolve_attention(attention: str) -> str:
    """Returns the transformers attn_implementation; sage runs as an sdpa patch."""
    if attention in {"auto", "sdpa", "sageattention"}:
        if attention == "sageattention" and importlib.util.find_spec("sageattention") is None:
            raise ImportError("sageattention was selected, but sageattention is not installed.")
        return "sdpa"
    if attention == "flash_attention":
        if importlib.util.find_spec("flash_attn") is None:
            raise ImportError("flash_attention was selected, but flash_attn is not installed.")
        return "flash_attention_2"
    raise ValueError(f"Unsupported attention mode: {attention}")


def dynamic_vram_active(device: torch.device) -> bool:
    if torch.device(device).type == "cpu":
        return False
    try:
        import comfy.memory_management

        if not bool(comfy.memory_management.aimdo_enabled):
            return False
        try:
            import comfy_aimdo.control
            import comfy_aimdo.host_buffer
            import comfy_aimdo.model_vbar

            return (
                comfy_aimdo.control.lib is not None
                and comfy_aimdo.host_buffer.lib is not None
                and comfy_aimdo.model_vbar.lib is not None
            )
        except Exception:
            return False
    except Exception:
        return False


def _register_many_with_comfy(patchers: list[Any]) -> None:
    patchers = [p for p in patchers if p is not None and p.load_device.type != "cpu"]
    if not patchers:
        return
    try:
        import comfy.model_management as mm

        already_loaded = {
            id(loaded.model) for loaded in mm.current_loaded_models if loaded.model is not None
        }
        to_load = [p for p in patchers if id(p) not in already_loaded]
        if not to_load:
            return
        mm.load_models_gpu(to_load)
        for patcher in to_load:
            logger.info(
                "Loaded %s through ComfyUI%s memory management.",
                patcher.model.__class__.__name__,
                "/AIMDO" if patcher.is_dynamic() else "",
            )
    except Exception as exc:
        raise RuntimeError("Could not load model through ComfyUI memory management.") from exc


def _unregister_from_comfy(patcher: Any) -> None:
    try:
        import comfy.model_management as mm

        survivors = []
        for loaded in mm.current_loaded_models:
            if loaded.model is patcher:
                try:
                    if loaded.model_finalizer is not None:
                        loaded.model_finalizer.detach()
                    loaded.model_finalizer = None
                    loaded.real_model = None
                except Exception:
                    pass
                try:
                    finalizer = getattr(loaded, "_patcher_finalizer", None)
                    if finalizer is not None:
                        finalizer.detach()
                    loaded._patcher_finalizer = None
                except Exception:
                    pass
                continue
            survivors.append(loaded)
        mm.current_loaded_models[:] = survivors
    except Exception:
        pass


def _set_module_device_if_writable(module: torch.nn.Module, device: torch.device) -> None:
    try:
        module.device = torch.device(device)
    except (AttributeError, RuntimeError, TypeError):
        pass


def _ensure_writable_device_property(module: torch.nn.Module) -> None:
    cls = module.__class__
    prop = getattr(cls, "device", None)
    if not isinstance(prop, property) or prop.fset is not None:
        return
    if getattr(module, "_firered_writable_device_property", False):
        return

    def _get_device(self):
        runtime_device = self.__dict__.get("_firered_runtime_device")
        if runtime_device is not None:
            return runtime_device
        return prop.fget(self)

    def _set_device(self, value):
        self.__dict__["_firered_runtime_device"] = torch.device(value)

    writable_cls = type(
        cls.__name__,
        (cls,),
        {
            "device": property(_get_device, _set_device),
            "_firered_device_base_class": cls,
            "__module__": cls.__module__,
        },
    )
    module.__class__ = writable_cls
    module._firered_writable_device_property = True


def register_runtime_module(module: torch.nn.Module, device: torch.device, *, dynamic: bool | None = None) -> Any:
    device = torch.device(device)
    module._firered_runtime_device = torch.device(device)
    _ensure_writable_device_property(module)
    if _ComfyCorePatcher is None or device.type == "cpu":
        module.to(device)
        return None

    import comfy.model_patcher as model_patcher

    use_dynamic = dynamic_vram_active(device) and dynamic is not False
    patcher_class = model_patcher.ModelPatcherDynamic if use_dynamic else model_patcher.ModelPatcher
    patcher = patcher_class(module, load_device=device, offload_device=torch.device("cpu"))
    module.model_loaded_weight_memory = 0
    _register_many_with_comfy([patcher])
    if not patcher.is_dynamic():
        _set_module_device_if_writable(module, device)
    logger.info(
        "Registered %s with ComfyUI%s memory management.",
        module.__class__.__name__,
        "/AIMDO" if patcher.is_dynamic() else "",
    )
    return patcher


def resume_runtime_module(patcher: Any, device: torch.device) -> None:
    del device
    if patcher is not None:
        _register_many_with_comfy([patcher])


def unload_runtime_module(patcher: Any, *, hard: bool = True) -> None:
    if patcher is None:
        return
    _unregister_from_comfy(patcher)
    try:
        patcher.detach()
    except Exception:
        pass


def resume_bundle_to_device(bundle: FireRedBundle) -> None:
    for patcher in bundle.patchers:
        resume_runtime_module(patcher, bundle.device)


def unload_firered_bundle(bundle: FireRedBundle | None, reason: str = "manual unload", hard: bool = True) -> None:
    global _ACTIVE_BUNDLE, _ACTIVE_LOAD_KEY
    if bundle is None:
        return
    logger.info("Unloading FireRedTTS3 bundle (%s).", reason)
    for patcher in list(bundle.patchers):
        unload_runtime_module(patcher, hard=hard)
    modules = [bundle.core, bundle.redae] + ([bundle.campp] if bundle.campp is not None else [])
    if not hard:
        for module in modules:
            try:
                module.to("cpu")
            except Exception:
                pass
    for module in modules:
        try:
            module.model_loaded_weight_memory = 0
            if hasattr(module, "dynamic_vbars"):
                module.dynamic_vbars.clear()
            if hard and hasattr(module, "to_empty"):
                module.to_empty(device=torch.device("meta"))
        except Exception:
            pass
    bundle.patchers.clear()
    if hard:
        bundle.core = None
        bundle.redae = None
        bundle.campp = None
        bundle.tokenizer = None
        bundle.frontend = None
    gc.collect()
    _empty_accelerator_cache()
    if _ACTIVE_BUNDLE is bundle:
        _ACTIVE_BUNDLE = None
        _ACTIVE_LOAD_KEY = None


def _core_dtype_policy(mode: str):
    def policy(name: str) -> torch.dtype:
        if mode == "bf16" and name.startswith("backbone_llm."):
            return torch.bfloat16
        return torch.float32

    return policy


def _redae_dtype_policy(mode: str):
    def policy(name: str) -> torch.dtype:
        if mode == "bf16" and name.startswith("encoder."):
            return torch.bfloat16
        return torch.float32

    return policy


def _apply_runtime_dtype_tags(bundle_modules: dict[str, torch.nn.Module], mode: str) -> None:
    core = bundle_modules["core"]
    redae = bundle_modules["redae"]
    campp = bundle_modules.get("campp")
    if mode == "bf16":
        native.set_runtime_dtype(core.backbone_llm, torch.bfloat16)
        for sub in (core.patch_encoder, core.dit, core.dit_head, core.stop_head):
            native.set_runtime_dtype(sub, torch.float32)
        for extra in ("spk_proj_llm", "spk_proj_dit"):
            if hasattr(core, extra):
                native.set_runtime_dtype(getattr(core, extra), torch.float32)
        native.set_runtime_dtype(redae.encoder, torch.bfloat16)
        native.set_runtime_dtype(redae.decoder, torch.float32)
    else:
        for module in bundle_modules.values():
            if module is not None:
                native.set_runtime_dtype(module, torch.float32)
    if campp is not None and mode == "bf16":
        native.set_runtime_dtype(campp, torch.float32)


def _build_modules(repo_dir: Path, variant: str, attn_impl: str):
    core_config = native.read_config(repo_dir / variant)
    redae_config = native.read_config(repo_dir / "redae")
    if variant == "fireredtts3_base":
        core = native.FireRedTTS3BaseCore(core_config, attn_impl)
    else:
        core = native.FireRedTTS3InstructCore(core_config, attn_impl)
    redae = native.RedAE(redae_config, attn_impl)
    return core, redae


def load_firered_bundle(
    repo_choice: str,
    variant: str,
    dtype_name: str,
    device_name: str,
    attention: str,
    download_if_missing: bool,
) -> FireRedBundle:
    global _ACTIVE_BUNDLE, _ACTIVE_LOAD_KEY

    if variant not in VARIANTS:
        raise ValueError(f"Unknown FireRedTTS3 variant: {variant}")

    register_model_folder()
    runtime_dir = resolve_model_dir(repo_choice, variant, download_if_missing)
    device = resolve_device(device_name)
    dtype_mode = resolve_dtype_mode(dtype_name, device)
    attn_impl = resolve_attention(attention)
    variant_short = "base" if variant == "fireredtts3_base" else "instruct"

    model_file = runtime_dir / variant / "model.safetensors"
    load_key = (
        str(runtime_dir.resolve()),
        variant,
        model_file.stat().st_mtime_ns,
        str(device),
        dtype_mode,
        attn_impl,
    )
    if _ACTIVE_BUNDLE is not None and _ACTIVE_LOAD_KEY == load_key:
        resume_bundle_to_device(_ACTIVE_BUNDLE)
        return _ACTIVE_BUNDLE
    if _ACTIVE_BUNDLE is not None:
        unload_firered_bundle(_ACTIVE_BUNDLE, reason="load settings changed")

    logger.info(
        "Loading FireRedTTS3 %s from %s on %s with dtype=%s attention=%s",
        variant, runtime_dir, device, dtype_mode, attn_impl,
    )

    try:
        from accelerate import init_empty_weights

        with init_empty_weights():
            core, redae = _build_modules(runtime_dir, variant, attn_impl)
    except ImportError:
        core, redae = _build_modules(runtime_dir, variant, attn_impl)
    # CAM++ is tiny (29 MB); build it eagerly so load_state_dict lands on real tensors.
    campp = native.CamppEmbedding() if variant == "fireredtts3_base" else None

    try:
        native.load_safetensors_into(core, runtime_dir / variant, dtype_policy=_core_dtype_policy(dtype_mode),
                                     ignore_missing=("lm_head.weight",))
        native.tie_core_weights(core)
        native.load_safetensors_into(redae, runtime_dir / "redae", dtype_policy=_redae_dtype_policy(dtype_mode))
        if campp is not None:
            campp_path = runtime_dir / "campp" / "campplus_voxceleb.bin"
            campp.model.load_state_dict(torch.load(campp_path, weights_only=True, map_location="cpu"))
            campp.eval()
        native.convert_modules_for_comfy(core)
        native.convert_modules_for_comfy(redae)
        if campp is not None:
            native.convert_modules_for_comfy(campp)
        modules = {"core": core, "redae": redae, "campp": campp}
        _apply_runtime_dtype_tags(modules, dtype_mode)
        autocast = dtype_mode == "bf16"
        core.autocast_bf16 = autocast
        redae.autocast_bf16 = autocast
        tokenizer = load_text_tokenizer(runtime_dir / "text_tokenizer")

        patchers: list[Any] = []
        try:
            use_dynamic = dynamic_vram_active(device)
            if use_dynamic:
                logger.info("AIMDO DynamicVRAM is active; using dynamic patchers for FireRedTTS3 modules.")
            else:
                logger.info("AIMDO not active; using static ComfyUI memory management.")
            for module in (core, redae) + ((campp,) if campp is not None else ()):
                patcher = register_runtime_module(module, device, dynamic=use_dynamic)
                if patcher is not None:
                    patchers.append(patcher)
        except Exception:
            for patcher in list(patchers):
                unload_runtime_module(patcher, hard=True)
            for module in (core, redae) + ((campp,) if campp is not None else ()):
                try:
                    module.model_loaded_weight_memory = 0
                    if hasattr(module, "dynamic_vbars"):
                        module.dynamic_vbars.clear()
                    if hasattr(module, "to_empty"):
                        module.to_empty(device=torch.device("meta"))
                except Exception:
                    pass
            gc.collect()
            _empty_accelerator_cache()
            raise

        fasttext_path = ensure_fasttext_model(download_if_missing)
        detector = FastTextLangDetector(fasttext_path) if fasttext_path is not None else None
        frontend = TextFrontend(use_wetext=True, fasttext_detector=detector)

        bundle = FireRedBundle(
            variant=variant_short,
            core=core,
            redae=redae,
            campp=campp,
            tokenizer=tokenizer,
            frontend=frontend,
            model_dir=runtime_dir,
            device=device,
            dtype_name=dtype_mode,
            attention=attention,
            patchers=patchers,
        )
        _ACTIVE_BUNDLE = bundle
        _ACTIVE_LOAD_KEY = load_key
        install_comfy_unload_hook()
        _empty_accelerator_cache()
        return bundle
    except Exception:
        del core, redae, campp
        gc.collect()
        _empty_accelerator_cache()
        raise


def unload_active_bundle() -> None:
    unload_firered_bundle(_ACTIVE_BUNDLE, reason="active unload")


def install_comfy_unload_hook() -> None:
    """Patch ComfyUI unload calls so the active native bundle hard-releases."""
    try:
        import comfy.model_management as mm
    except Exception:
        return

    if getattr(mm, "_fireredtts3_unload_hook_installed", False):
        return

    original_unload_all_models = mm.unload_all_models

    def unload_all_models_with_firered(*args, **kwargs):
        try:
            return original_unload_all_models(*args, **kwargs)
        finally:
            unload_firered_bundle(_ACTIVE_BUNDLE, reason="ComfyUI unload_all_models")

    mm.unload_all_models = unload_all_models_with_firered

    original_unload_model_and_clones = getattr(mm, "unload_model_and_clones", None)
    if original_unload_model_and_clones is not None:
        def unload_model_and_clones_with_firered(model, *args, **kwargs):
            try:
                return original_unload_model_and_clones(model, *args, **kwargs)
            finally:
                if _ACTIVE_BUNDLE is not None and model is not None:
                    owned = list(_ACTIVE_BUNDLE.patchers) + [
                        m for m in (_ACTIVE_BUNDLE.core, _ACTIVE_BUNDLE.redae, _ACTIVE_BUNDLE.campp)
                        if m is not None
                    ]
                    if any(existing is model or existing is getattr(model, "model", None)
                           for existing in owned if existing is not None):
                        unload_firered_bundle(_ACTIVE_BUNDLE, reason="ComfyUI unload_model_and_clones")

        mm.unload_model_and_clones = unload_model_and_clones_with_firered

    mm._fireredtts3_unload_hook_installed = True
    logger.debug("Installed FireRedTTS3 unload hook for ComfyUI native unload.")
