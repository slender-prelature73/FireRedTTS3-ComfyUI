"""Full validation of the FireRedTTS3 INT8 ConvRot conversion (stages 2-7 + perf).

Run from anywhere with the ComfyUI venv python:
    python tools/validate_int8_convrot.py

Stages:
  2  per-layer weight roundtrip (official quantize -> official dequantize, from disk)
  3  real-activation comparison (hooks on the BF16 model, ck.int8_linear vs F.linear)
  4  on-disk checkpoint structure validation
  5  runtime proof (loader + generation reach comfy_kitchen.int8_linear)
  6  BF16 vs INT8 internal comparison (patch counts, latents, waveforms)
  7  end-to-end audio (EN/ZH/JA, whisper + CAM++ speaker similarity)
  +  performance / memory report
"""

from __future__ import annotations

import importlib.util
import json
import statistics
import sys
import time
from pathlib import Path

import torch
import torchaudio

sys.path.insert(0, r"C:\Users\drbaph\Documents\ComfyUI")

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(r"C:\Users\drbaph\AppData\Local\Temp\opencode\firered_int8_validation")
OUT.mkdir(parents=True, exist_ok=True)

FP32_REPO = r"C:\Users\drbaph\Documents\ComfyUI\models\fireredtts3\FireRedTeam_FireRedTTS3"
INT8_REPO = r"C:\Users\drbaph\Documents\ComfyUI\models\fireredtts3\FireRedTTS3-int8-convrot"
VARIANT = "fireredtts3_base"
PROMPT_WAV = r"C:\Users\drbaph\Documents\ComfyUI\input\Narrator_USA_deep-voice.wav"
GROUP = 256

spec = importlib.util.spec_from_file_location("fireredtts3_nodepack", ROOT / "__init__.py",
                                              submodule_search_locations=[str(ROOT)])
pack = importlib.util.module_from_spec(spec)
sys.modules["fireredtts3_nodepack"] = pack
spec.loader.exec_module(pack)

from fireredtts3_nodepack import int8 as int8rt  # noqa: E402
from fireredtts3_nodepack import loader, native, whisper  # noqa: E402
from fireredtts3_nodepack.nodes import _generate_clone_audio  # noqa: E402

DEV = torch.device("cuda")
VERDICT: dict[str, bool] = {}


def gate(name: str, ok: bool, detail: str = "") -> None:
    VERDICT[name] = bool(ok)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}{(' - ' + detail) if detail else ''}")


def original_weights(names):
    from safetensors import safe_open

    src = Path(FP32_REPO) / VARIANT / "model.safetensors"
    with safe_open(str(src), framework="pt", device="cpu") as f:
        for name in names:
            yield name, f.get_tensor(name)


def stage2_layer_roundtrip():
    from comfy_kitchen.tensor import TensorWiseINT8Layout
    from safetensors import safe_open

    src = Path(FP32_REPO) / VARIANT / "model.safetensors"
    int8_file = Path(INT8_REPO) / VARIANT / "model.safetensors"
    results = []
    worst_rel, worst_cos, worst_name = 0.0, 1.0, ""
    with safe_open(str(int8_file), framework="pt", device="cpu") as fq, safe_open(str(src), framework="pt", device="cpu") as fo:
        quant_names = [k[:-len(".comfy_quant")] for k in fq.keys() if k.endswith(".comfy_quant")]
        for prefix in quant_names:
            q = fq.get_tensor(f"{prefix}.weight").to(DEV)
            scale = fq.get_tensor(f"{prefix}.weight_scale").to(DEV)
            w = fo.get_tensor(f"{prefix}.weight").to(DEV, torch.float32)
            params = TensorWiseINT8Layout.Params(scale=scale, orig_dtype=torch.float32,
                                                 orig_shape=tuple(w.shape), is_weight=True,
                                                 convrot=True, convrot_groupsize=GROUP)
            w_dq = TensorWiseINT8Layout.dequantize(q, params)
            rel = ((w_dq - w).norm() / w.norm()).item()
            cos = torch.nn.functional.cosine_similarity(w_dq.flatten(), w.flatten(), dim=0).item()
            results.append({"layer": prefix, "rel_l2": rel, "cosine": cos,
                            "max_abs": (w_dq - w).abs().max().item()})
            if rel > worst_rel:
                worst_rel, worst_name = rel, prefix
            worst_cos = min(worst_cos, cos)
    (OUT / "validation_layers.json").write_text(json.dumps({
        "layers": results,
        "worst_rel_l2": worst_rel, "worst_cosine": worst_cos, "worst_layer": worst_name,
    }, indent=2))
    finite = all(torch.isfinite(torch.tensor([r["rel_l2"], r["cosine"]])).all() for r in results)
    gate("stage2_layer_roundtrip", finite and worst_rel <= 0.02 and worst_cos >= 0.999,
         f"worst rel_l2={worst_rel:.5f} ({worst_name}) worst cos={worst_cos:.6f} over {len(results)} layers")
    return results


def load_bundle(repo_label, dtype="auto"):
    return loader.load_firered_bundle(repo_choice=repo_label, variant=VARIANT, dtype_name=dtype,
                                      device_name="auto", attention="auto", download_if_missing=False)


def gen_clone(bundle, text, language, prompt_audio, transcript, seed=1234):
    return _generate_clone_audio(bundle, text=text, language=language, prompt_text=transcript,
                                 prompt_audio=prompt_audio, n_timesteps=10, inference_cfg=2.0,
                                 stop_threshold=0.5, seed=seed, max_audio_seconds=30.0,
                                 do_tn=True, do_split=True, cross_fade_ms=50.0)


def stage3_real_activations():
    captured = {}

    def make_hook(name):
        def hook(module, args):
            if name not in captured:
                x = args[0].detach()
                captured[name] = x.reshape(-1, x.shape[-1])[:256].float().cpu()
        return hook

    wanted_regions = ["backbone_llm.layers.0.", "backbone_llm.layers.14.", "backbone_llm.layers.27.",
                      "patch_encoder.blocks.0.", "patch_encoder.blocks.7.",
                      "dit.blocks.0.", "dit.blocks.5.", "dit.blocks.10."]
    bundle = load_bundle("FireRedTTS3 fp32 - FireRedTeam (auto-download)")
    handles = []
    for name, module in bundle.core.named_modules():
        if isinstance(module, int8rt.ConvRotInt8Linear):
            continue
        if not isinstance(module, torch.nn.Linear):
            continue
        if any(name.startswith(r) for r in wanted_regions) and name.split(".")[-1] in (
                "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
                "to_q", "to_k", "to_v", "ff"):
            if name.endswith(".ff"):
                continue
            handles.append(module.register_forward_pre_hook(make_hook(name)))
            if len(handles) >= 14:
                break

    wav, sr = torchaudio.load(PROMPT_WAV)
    audio_dict = {"waveform": wav.unsqueeze(0).float(), "sample_rate": int(sr)}
    transcript = whisper.transcribe_audio(audio_dict, "whisper-large-v3-turbo (auto-download)", "auto", "auto",
                                          "transcribe", 30, False)
    gen_clone(bundle, "The quick brown fox jumps over the lazy dog.", "auto", audio_dict, transcript)
    for h in handles:
        h.remove()
    loader.unload_active_bundle()

    import comfy_kitchen
    from safetensors import safe_open

    int8_file = Path(INT8_REPO) / VARIANT / "model.safetensors"
    results = []
    worst_rel, worst_cos, worst_name = 0.0, 1.0, ""
    with safe_open(str(int8_file), framework="pt", device="cpu") as fq, safe_open(str(Path(FP32_REPO) / VARIANT / "model.safetensors"), framework="pt", device="cpu") as fo:
        for name, x in captured.items():
            q = fq.get_tensor(f"{name}.weight").to(DEV)
            scale = fq.get_tensor(f"{name}.weight_scale").to(DEV)
            w = fo.get_tensor(f"{name}.weight").to(DEV, torch.float32)
            b = fo.get_tensor(f"{name}.bias").to(DEV, torch.float32) if f"{name}.bias" in fo.keys() else None
            x = x.to(DEV)
            y_ref = torch.nn.functional.linear(x, w, b)
            y_q = comfy_kitchen.int8_linear(x, q, scale, b, out_dtype=torch.float32,
                                            convrot=True, convrot_groupsize=GROUP)
            rel = ((y_ref - y_q).norm() / y_ref.norm()).item()
            cos = torch.nn.functional.cosine_similarity(y_ref.flatten(), y_q.flatten(), dim=0).item()
            results.append({"layer": name, "rows": int(x.shape[0]), "rel_l2": rel, "cosine": cos,
                            "nan_inf": int((~torch.isfinite(y_q)).sum())})
            if rel > worst_rel:
                worst_rel, worst_name = rel, name
            worst_cos = min(worst_cos, cos)
    (OUT / "validation_activations.json").write_text(json.dumps({
        "layers": results, "worst_rel_l2": worst_rel, "worst_cosine": worst_cos, "worst_layer": worst_name,
    }, indent=2))
    clean = all(r["nan_inf"] == 0 for r in results)
    gate("stage3_real_activations", clean and worst_rel <= 0.05 and worst_cos >= 0.995,
         f"{len(results)} layers, worst rel_l2={worst_rel:.5f} ({worst_name}) worst cos={worst_cos:.6f}")
    return transcript, audio_dict


def stage4_structure(manifest):
    from safetensors import safe_open

    int8_file = Path(INT8_REPO) / VARIANT / "model.safetensors"
    src_file = Path(FP32_REPO) / VARIANT / "model.safetensors"
    problems = []
    conv_keys = 0
    with safe_open(str(int8_file), framework="pt", device="cpu") as fq, safe_open(str(src_file), framework="pt", device="cpu") as fo:
        int8_keys = set(fq.keys())
        quant_prefixes = {k[:-len(".comfy_quant")] for k in int8_keys if k.endswith(".comfy_quant")}
        for prefix in quant_prefixes:
            w = fq.get_tensor(f"{prefix}.weight")
            s = fq.get_tensor(f"{prefix}.weight_scale")
            meta = json.loads(fq.get_tensor(f"{prefix}.comfy_quant").numpy().tobytes())
            if w.dtype != torch.int8 or w.shape != tuple(fo.get_tensor(f"{prefix}.weight").shape):
                problems.append(f"{prefix}.weight bad")
            if s.dtype != torch.float32 or tuple(s.shape) != (w.shape[0], 1):
                problems.append(f"{prefix}.weight_scale bad")
            if not torch.isfinite(s).all() or not (s > 0).all():
                problems.append(f"{prefix}.weight_scale values bad")
            if meta.get("format") != "int8_tensorwise" or meta.get("convrot") is not True or meta.get("convrot_groupsize") != GROUP:
                problems.append(f"{prefix}.comfy_quant bad: {meta}")
            if w.shape[1] % GROUP != 0:
                problems.append(f"{prefix} divisibility")
        for key in fo.keys():
            if key not in int8_keys:
                problems.append(f"missing from int8 checkpoint: {key}")
            elif key.endswith(".weight") and key[:-7] in quant_prefixes:
                pass  # verified above
            elif key.endswith(".weight") or key.endswith(".bias"):
                a, b = fo.get_tensor(key), fq.get_tensor(key)
                if a.dtype != b.dtype or a.shape != b.shape or not torch.equal(a, b):
                    problems.append(f"non-quantized tensor altered: {key}")
                if key.endswith(".weight") and a.dtype != torch.float32:
                    problems.append(f"unexpected float dtype in source: {key}")
        for key in int8_keys:
            if not (key in set(fo.keys()) or key.endswith((".weight_scale", ".comfy_quant"))):
                problems.append(f"unexpected new tensor key: {key}")
        conv_keys = sum(1 for k in fo.keys() if ".conv.block." in k)

    total_src_keys = 0
    with safe_open(str(src_file), framework="pt", device="cpu") as fo:
        total_src_keys = len(fo.keys())
    expected_selected = manifest["totals"]["layers_quantized"]
    gate("stage4_checkpoint_structure",
         not problems and len(quant_prefixes) == expected_selected and conv_keys > 0,
         f"{len(quant_prefixes)} quantized (manifest {expected_selected}), {total_src_keys} source keys preserved, "
         f"conv1d keys={conv_keys}" + (f", problems: {problems[:4]}" if problems else ""))
    (OUT / "validation_structure.json").write_text(json.dumps({
        "problems": problems, "quantized": len(quant_prefixes), "source_keys": total_src_keys,
    }, indent=2))


def stage5_runtime_proof():
    int8rt.reset_runtime_stats()
    bundle = load_bundle(loader.INT8_REPO_LABEL)
    qparams, qcount = int8rt.quantized_parameter_count(bundle.core)
    wav, sr = torchaudio.load(PROMPT_WAV)
    audio_dict = {"waveform": wav.unsqueeze(0).float(), "sample_rate": int(sr)}
    transcript = whisper.transcribe_audio(audio_dict, "whisper-large-v3-turbo (auto-download)", "auto", "auto",
                                          "transcribe", 30, False)
    result = gen_clone(bundle, "Runtime proof sentence for int8 convrot.", "auto", audio_dict, transcript)
    result = gen_clone(bundle, "Second generation without reload.", "auto", audio_dict, transcript)
    stats = dict(int8rt.RUNTIME_STATS)
    still_int8, _ = int8rt.quantized_parameter_count(bundle.core)
    dtype_ok = all(m.weight.dtype == torch.int8 and m.weight_scale.dtype == torch.float32
                   for m in bundle.core.modules() if isinstance(m, int8rt.ConvRotInt8Linear))
    finite = bool(torch.isfinite(result["waveform"]).all())
    gate("stage5_runtime_reaches_ck_int8_linear",
         stats["calls"] > 0 and qcount > 0 and stats["weight_dtype"] == torch.int8
         and stats["groupsize"] == GROUP and still_int8 > 0 and dtype_ok and finite,
         f"modules={qcount} kernel_calls={stats['calls']} weight_dtype={stats['weight_dtype']} "
         f"groupsize={stats['groupsize']}")

    # unload -> reload through the node's ComfyUI memory management, then generate again
    loader.unload_active_bundle()
    bundle = load_bundle(loader.INT8_REPO_LABEL)
    _, reloaded = int8rt.quantized_parameter_count(bundle.core)
    reloaded_ok = all(m.weight.dtype == torch.int8 and m.weight_scale.dtype == torch.float32
                      and m.convrot_groupsize == GROUP
                      for m in bundle.core.modules() if isinstance(m, int8rt.ConvRotInt8Linear))
    result = gen_clone(bundle, "Third generation after unload and reload.", "auto", audio_dict, transcript)
    gate("stage5_offload_reload_preserves_quantization",
         reloaded == qcount and reloaded_ok and bool(torch.isfinite(result["waveform"]).all()),
         f"{reloaded} modules still int8/fp32-scale G={GROUP} after unload+reload")
    return bundle


def stage6_internal_comparison(transcript, audio_dict):
    def run_once(bundle, text, language, seed):
        latents_info = {}
        original_gen = bundle.core.generate

        def counting_generate(*args, **kwargs):
            latents = original_gen(*args, **kwargs)
            latents_info["patches"] = int(latents.shape[1] // bundle.core.patch_size)
            latents_info["latent_rms"] = float(latents.float().pow(2).mean().sqrt())
            latents_info["latent_max"] = float(latents.float().abs().max())
            latents_info["finite"] = bool(torch.isfinite(latents).all())
            return latents

        bundle.core.generate = counting_generate
        try:
            audio = gen_clone(bundle, text, language, audio_dict, transcript, seed=seed)
        finally:
            bundle.core.generate = original_gen
        return audio, latents_info

    text = "The quick brown fox jumps over the lazy dog, and then wonders why it bothered."
    bf16_bundle = load_bundle(loader.OFFICIAL_REPO_LABEL)
    torch.cuda.reset_peak_memory_stats()
    bf16_audio, bf16_info = run_once(bf16_bundle, text, "auto", 1234)
    bf16_peak = torch.cuda.max_memory_allocated() / 2**30
    loader.unload_active_bundle()

    int8_bundle = load_bundle(loader.INT8_REPO_LABEL)
    torch.cuda.reset_peak_memory_stats()
    int8_audio, int8_info = run_once(int8_bundle, text, "auto", 1234)
    int8_peak = torch.cuda.max_memory_allocated() / 2**30

    a = bf16_audio["waveform"][0, 0]
    b = int8_audio["waveform"][0, 0]
    n = min(a.numel(), b.numel())
    cos = torch.nn.functional.cosine_similarity(a[:n].flatten(), b[:n].flatten(), dim=0).item() if n else 0.0
    dur_a = a.numel() / bf16_audio["sample_rate"]
    dur_b = b.numel() / int8_audio["sample_rate"]
    patch_diff = abs(bf16_info["patches"] - int8_info["patches"])
    gate("stage6_internal_comparison",
         bf16_info["finite"] and int8_info["finite"]
         and int8_info["latent_max"] < 100.0
         and 4 <= int8_info["patches"] <= 400
         and patch_diff <= 3
         and dur_b > 1.0,
         f"patches bf16={bf16_info['patches']} int8={int8_info['patches']} (diff {patch_diff}), "
         f"dur {dur_a:.2f}s/{dur_b:.2f}s, waveform cos={cos:.4f}, "
         f"latent_max int8={int8_info['latent_max']:.2f}")

    (OUT / "validation_internal.json").write_text(json.dumps({
        "bf16": {**bf16_info, "duration": dur_a, "peak_vram_gib": bf16_peak},
        "int8": {**int8_info, "duration": dur_b, "peak_vram_gib": int8_peak, "waveform_cosine": cos},
    }, indent=2))
    return int8_bundle, {"bf16_peak": bf16_peak, "int8_peak": int8_peak}


def stage7_end_to_end(transcript, audio_dict):
    cases = [
        ("english", "english", "The weather is beautiful today, let us take a walk in the park.", "park", True),
        ("chinese", "chinese", "今天天气很好，我们一起去公园散步吧。", "公园", True),
        ("japanese", "japanese", "今日はとても良い天気ですね。散歩に行きましょう。", "散歩", False),
    ]

    def run_cases(bundle):
        ref_emb = native.speaker_embedding(bundle, *native.comfy_audio_to_tensor(audio_dict)).squeeze(0)
        out = []
        for label, whisper_lang, text, needle, _abs in cases:
            audio = gen_clone(bundle, text, "auto", audio_dict, transcript, seed=1234)
            wave = audio["waveform"][0, 0]
            dur = wave.numel() / audio["sample_rate"]
            txt = whisper.transcribe_audio({"waveform": wave.view(1, 1, -1), "sample_rate": audio["sample_rate"]},
                                           "whisper-large-v3-turbo (auto-download)", "auto", whisper_lang,
                                           "transcribe", 30, False)
            gen_emb = native.speaker_embedding(bundle, wave, audio["sample_rate"]).squeeze(0)
            out.append({
                "lang": label, "duration": dur,
                "finite": bool(torch.isfinite(wave).all()),
                "nonzero": float(wave.abs().max()) > 0.01,
                "no_clip": float(wave.abs().max()) < 0.999,
                "plausible": 1.0 < dur < 40.0,
                "asr": txt[:90], "asr_ok": needle.lower() in txt.lower(),
                "speaker_sim": float(torch.nn.functional.cosine_similarity(ref_emb, gen_emb, dim=0)),
            })
            torchaudio.save(str(OUT / f"{{model}}_{label}.wav".replace("{model}", bundle.dtype_name)), wave.unsqueeze(0), audio["sample_rate"])
        return out

    int8_res = run_cases(load_bundle(loader.INT8_REPO_LABEL))
    loader.unload_active_bundle()
    bf16_res = run_cases(load_bundle(loader.OFFICIAL_REPO_LABEL))
    loader.unload_active_bundle()

    ok_all = True
    for case, i8, b16 in zip(cases, int8_res, bf16_res):
        absolute = case[4]
        asr_ok = i8["asr_ok"] if absolute else (i8["asr_ok"] or not b16["asr_ok"])
        case_ok = (i8["finite"] and i8["nonzero"] and i8["no_clip"] and i8["plausible"]
                   and asr_ok and i8["speaker_sim"] >= b16["speaker_sim"] - 0.1)
        ok_all = ok_all and case_ok
        print(f"    {case[0]}: int8 dur={i8['duration']:.2f}s asr_ok={i8['asr_ok']} spk={i8['speaker_sim']:.4f} | "
              f"bf16 asr_ok={b16['asr_ok']} spk={b16['speaker_sim']:.4f} | transcript={i8['asr'][:60]}")
    gate("stage7_end_to_end_audio", ok_all, f"{len(cases)} languages (int8 vs bf16)")
    (OUT / "validation_e2e.json").write_text(json.dumps(
        {"int8": int8_res, "bf16": bf16_res}, indent=2, ensure_ascii=False))
    return int8_res, bf16_res


def perf_report(peaks):
    def timed(bundle, text, transcript, audio_dict):
        gen_clone(bundle, text, "auto", audio_dict, transcript)  # warmup
        times = []
        for _ in range(3):
            torch.cuda.synchronize()
            t0 = time.time()
            gen_clone(bundle, text, "auto", audio_dict, transcript)
            torch.cuda.synchronize()
            times.append(time.time() - t0)
        return statistics.median(times)

    wav, sr = torchaudio.load(PROMPT_WAV)
    audio_dict = {"waveform": wav.unsqueeze(0).float(), "sample_rate": int(sr)}
    transcript = whisper.transcribe_audio(audio_dict, "whisper-large-v3-turbo (auto-download)", "auto", "auto",
                                          "transcribe", 30, False)
    text = "Performance measurement sentence with a reasonable length for timing."
    t0 = time.time()
    bf16_bundle = load_bundle(loader.OFFICIAL_REPO_LABEL)
    bf16_load = time.time() - t0
    bf16_time = timed(bf16_bundle, text, transcript, audio_dict)
    loader.unload_active_bundle()
    t0 = time.time()
    int8_bundle = load_bundle(loader.INT8_REPO_LABEL)
    int8_load = time.time() - t0
    int8_time = timed(int8_bundle, text, transcript, audio_dict)

    sizes = {
        "fp32_core_gib": (Path(FP32_REPO) / VARIANT / "model.safetensors").stat().st_size / 2**30,
        "int8_core_gib": (Path(INT8_REPO) / VARIANT / "model.safetensors").stat().st_size / 2**30,
    }
    report = {
        "checkpoint": sizes,
        "load_time_s": {"bf16": round(bf16_load, 1), "int8": round(int8_load, 1)},
        "generation_time_s_median": {"bf16": round(bf16_time, 2), "int8": round(int8_time, 2)},
        "peak_vram_gib": peaks,
    }
    (OUT / "perf_report.json").write_text(json.dumps(report, indent=2))
    print(f"    checkpoint {sizes['fp32_core_gib']:.2f} -> {sizes['int8_core_gib']:.2f} GiB | "
          f"gen {bf16_time:.2f}s -> {int8_time:.2f}s | load {bf16_load:.1f}s -> {int8_load:.1f}s | "
          f"peak VRAM {peaks['bf16_peak']:.2f} -> {peaks['int8_peak']:.2f} GiB")
    gate("perf_memory_improved", sizes["int8_core_gib"] < sizes["fp32_core_gib"] and peaks["int8_peak"] < peaks["bf16_peak"])


def main():
    manifest = json.loads((Path(INT8_REPO) / VARIANT / "quantization_manifest.json").read_text(encoding="utf-8"))
    stage2_layer_roundtrip()
    transcript, audio_dict = stage3_real_activations()
    stage4_structure(manifest)
    stage5_runtime_proof()
    loader.unload_active_bundle()
    _, peaks = stage6_internal_comparison(transcript, audio_dict)
    loader.unload_active_bundle()
    stage7_end_to_end(transcript, audio_dict)
    perf_report(peaks)

    print("\n==== VERDICT ====")
    for name, ok in VERDICT.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    complete = all(VERDICT.values())
    print(f"\nINT8 CONVROT VALIDATION: {'PASS' if complete else 'INCOMPLETE'}")


if __name__ == "__main__":
    main()
