"""Spike: XPU graph-captured code-predictor for Qwen3-TTS.

Stack: autotuned inductor (talker + code-predictor) + ONE XPUGraph replaying the
full 15-step code-predictor sequence per main token (greedy, static buffers).

Correctness: two seeded runs (graph on / graph off) must produce matching audio.
Usage: python q3x-graph.py
"""
import functools
import os
import time

import torch
import numpy as np
import soundfile as sf

# transformers 4.57.3 lacks merge_with_config_defaults (5.x name the fork's
# tokenizer prefers); inject a passthrough so the fork's primary import path works.
import transformers.utils.generic as _g

if not hasattr(_g, "merge_with_config_defaults"):
    def merge_with_config_defaults(func):
        @functools.wraps(func)
        def wrapper(self, *a, **k):
            return func(self, *a, **k)

        return wrapper

    _g.merge_with_config_defaults = merge_with_config_defaults
    print("SHIM: merge_with_config_defaults (passthrough) injected", flush=True)

from qwen_tts import Qwen3TTSModel
from qwen_tts.xpu_graph import GraphedCodePredictor

MODEL_DIR = "/models/Qwen3-TTS-12Hz-1.7B-CustomVoice"

TEXT = (
    "This is a throwaway test of the Qwen three text to speech model running "
    "on an Intel Arc Pro B70 graphics card. It synthesizes speech from text, "
    "which would make it suitable for reading chat replies aloud in a home "
    "server setup. The sentence is long enough to exercise the generator for "
    "several seconds of audio."
)

SHORT = "Compiling the kernels now. This short sentence absorbs the inductor warm-up."

stats = {}


def hook_module(mod, name):
    t0 = {}

    def pre(m, inp):
        torch.xpu.synchronize()
        t0["t"] = time.perf_counter()

    def post(m, inp, out):
        torch.xpu.synchronize()
        stats.setdefault(name, []).append(time.perf_counter() - t0["t"])

    mod.register_forward_pre_hook(pre)
    mod.register_forward_hook(post)


def timed_method(obj, attr, name):
    orig = getattr(obj, attr)

    def wrapper(*a, **k):
        torch.xpu.synchronize()
        t0 = time.perf_counter()
        out = orig(*a, **k)
        torch.xpu.synchronize()
        stats.setdefault(name, []).append(time.perf_counter() - t0)
        return out

    setattr(obj, attr, wrapper)


def patch_multinomial():
    orig = torch.multinomial

    def tm(*a, **k):
        torch.xpu.synchronize()
        t0 = time.perf_counter()
        out = orig(*a, **k)
        torch.xpu.synchronize()
        stats.setdefault("multinomial", []).append(time.perf_counter() - t0)
        return out

    torch.multinomial = tm


def ensure_on_xpu(wrapper):
    dev = torch.device("xpu")
    model = wrapper.model
    if next(model.parameters()).device.type != "xpu":
        model.to(dev)
    wrapper.device = dev
    st = model.speech_tokenizer
    if st is not None and getattr(st, "model", None) is not None:
        if next(st.model.parameters()).device.type != "xpu":
            st.model.to(dev)
        st.device = dev
    print("PLACEMENT: all components on xpu", flush=True)


def apply_compile(wrapper, compile_cp=False):
    """Compile talker.model (always) and optionally code_predictor.model.

    With the graph path, the code-predictor is captured EAGER into the XPUGraph:
    dynamo cannot fake-tensor-trace the custom in-place StaticKVCache, and the
    dispatch-bound cost is removed by the replay anyway.
    """
    kw = {"dynamic": True, "fullgraph": False, "backend": "inductor", "mode": "max-autotune"}
    talker = wrapper.model.talker
    talker.model = torch.compile(talker.model, **kw)
    if compile_cp:
        talker.code_predictor.model = torch.compile(talker.code_predictor.model, **kw)
        print("COMPILE: inductor max-autotune applied to talker.model + code_predictor.model", flush=True)
    else:
        print("COMPILE: inductor max-autotune applied to talker.model only (CP eager -> graph-captured)", flush=True)


def warmup_decode(wrapper):
    buckets = [64, 128, 192, 256, 320, 384, 448, 512]
    st = wrapper.model.speech_tokenizer
    device = next(st.model.parameters()).device
    n_q = getattr(st.model.config, "num_quantizers", 16)
    with torch.inference_mode():
        for bt in buckets:
            dummy = torch.zeros(bt, n_q, dtype=torch.long, device=device)
            st.decode([{"audio_codes": dummy}])
            torch.xpu.synchronize()
    print(f"WARMUP: Mimi decoder kernels primed T={buckets[0]}..{buckets[-1]}", flush=True)


def xpu_cleanup():
    import gc
    torch.xpu.synchronize()
    torch.xpu.empty_cache()
    gc.collect()


def rms(x):
    return float(np.sqrt(np.mean(np.square(np.asarray(x, dtype=np.float64)))))


def main():
    assert torch.xpu.is_available()

    wrapper = Qwen3TTSModel.from_pretrained(
        MODEL_DIR, device_map="xpu", dtype=torch.bfloat16, attn_implementation="sdpa",
    )
    ensure_on_xpu(wrapper)
    apply_compile(wrapper)
    warmup_decode(wrapper)

    # Attach the graph-captured code-predictor (captures the compiled cp body).
    talker = wrapper.model.talker
    talker._cp_graph = GraphedCodePredictor(talker, torch.device("xpu"), torch.bfloat16)

    # Absorb inductor + exercise the graph.
    t_c = time.monotonic()
    wrapper.generate_custom_voice(text=SHORT, language="English", speaker="Aiden",
                                  non_streaming_mode=True)
    torch.xpu.synchronize()
    print(f"COMPILE-ABSORB: {time.monotonic()-t_c:.1f}s", flush=True)
    xpu_cleanup()

    # ---- Correctness: seeded graph-on vs graph-off must match ----
    # do_sample=False AND subtalker_dosample=False on BOTH sides: the graph
    # path is structurally greedy (argmax baked in), so the eager path must
    # match it or the trajectories diverge (different codebooks -> different
    # talker feedback -> different EOS).
    # Token-level check: collect the per-step codec_ids (16 codebooks) from
    # the talker forward output and compare the sequences directly.
    codes_box = []

    def _collect(m, inp, out):
        try:
            hs = out.hidden_states
            if isinstance(hs, tuple) and len(hs) == 2 and hs[1] is not None:
                codes_box.append(hs[1].detach().reshape(-1).cpu())
        except Exception:
            pass

    talker.register_forward_hook(_collect)

    def gen(seed):
        torch.manual_seed(seed)
        torch.xpu.manual_seed_all(seed)
        codes_box.clear()
        wavs, sr = wrapper.generate_custom_voice(text=TEXT, language="English", speaker="Aiden",
                                                 non_streaming_mode=True,
                                                 do_sample=False, subtalker_dosample=False)
        torch.xpu.synchronize()
        codes = torch.cat(codes_box) if codes_box else torch.zeros(0, dtype=torch.long)
        return wavs[0], sr, codes

    wav_on, sr, codes_on = gen(42)
    xpu_cleanup()
    talker._cp_graph = None  # detach -> eager manual loop
    wav_off, _, codes_off = gen(42)
    talker._cp_graph = GraphedCodePredictor(talker, torch.device("xpu"), torch.bfloat16)
    xpu_cleanup()

    a, b = np.asarray(wav_on, dtype=np.float64), np.asarray(wav_off, dtype=np.float64)
    n = min(len(a), len(b))
    corr = float(np.corrcoef(a[:n], b[:n])[0, 1]) if n > 10 else float("nan")
    mad = float(np.mean(np.abs(a[:n] - b[:n])))
    print(f"CORRECTNESS: dur {len(a)/sr:.2f}s vs {len(b)/sr:.2f}s  corr={corr:.5f}  mean_abs_diff={mad:.2e}  "
          f"rms on/off = {rms(a):.4f}/{rms(b):.4f}", flush=True)
    n_min = min(codes_on.shape[0], codes_off.shape[0])
    if n_min:
        row_eq = (codes_on[:n_min].view(-1, 16) == codes_off[:n_min].view(-1, 16)).all(dim=1)
        first_bad = int((~row_eq).nonzero().flatten().clamp(max=n_min - 1)[0].item()) if (~row_eq).any() else -1
        print(f"CODES: on={codes_on.shape[0]} off={codes_off.shape[0]}  "
              f"prefix_rows_exact={row_eq.float().mean().item():.4f}  first_diverge_row={first_bad}", flush=True)
    else:
        print(f"CODES: on={codes_on.shape[0]} off={codes_off.shape[0]} (no overlap)", flush=True)

    # ---- Timed profile run (graph on) ----
    hook_module(talker.model, "talker")
    st = wrapper.model.speech_tokenizer
    timed_method(st, "decode", "mimi")
    timed_method(talker._cp_graph, "run", "cp_graph")
    patch_multinomial()

    torch.xpu.synchronize()
    t0 = time.perf_counter()
    wavs, sr = wrapper.generate_custom_voice(text=TEXT, language="English", speaker="Aiden",
                                             non_streaming_mode=True)
    torch.xpu.synchronize()
    gen_secs = time.perf_counter() - t0
    stats["_generate"] = [gen_secs]
    dur = len(wavs[0]) / sr
    print(f"AUDIO: {dur:.1f}s in {gen_secs:.1f}s -> RTF {gen_secs/dur:.2f}x", flush=True)

    print("=== PROFILE graph=True autotune=True ===", flush=True)
    tot = {}
    for name, xs in stats.items():
        tot[name] = sum(xs)
        print(f"{name:16s} n={len(xs):5d} total={tot[name]*1000:9.1f}ms "
              f"mean={tot[name]/len(xs)*1000:7.2f}ms max={max(xs)*1000:8.1f}ms", flush=True)
    gen = stats["_generate"][0]
    known = sum(v for k, v in tot.items() if k != "_generate")
    print(f"{'_generate':16s} total={gen*1000:9.1f}ms  glue(residual)={(gen-known)*1000:9.1f}ms", flush=True)
    print(f"PEAK MEM: {torch.xpu.max_memory_allocated('xpu')/1e9:.2f} GiB", flush=True)
    sf.write("/out/q3x-graph.wav", wavs[0], sr)
    print("SAVED /out/q3x-graph.wav", flush=True)


if __name__ == "__main__":
    main()
