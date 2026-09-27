"""Spike: XPU graph capture of BOTH the talker decode step and the code-predictor.

No torch.compile anywhere: the talker decode step is an XPUGraph replay with a
static per-layer KV cache + precomputed additive attention-mask table, and the
15-step code-predictor is a second XPUGraph. Expect: no inductor absorb, and
the talker's ~12 ms (autotuned) dropping toward the CP graph's ~15 ms floor.

Correctness: fully-greedy (do_sample=False, subtalker_dosample=False) graph-on
vs graph-off. Note: static-mask vs is_causal SDPA paths can differ in bf16
reduction order, so expect high (not perfect) correlation and near-equal
durations, per faster-qwen3-tts's parity notes.
Usage: python q3x-talkerg.py
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
from qwen_tts.xpu_graph import GraphedCodePredictor, GraphedTalkerStep

MODEL_DIR = "/models/Qwen3-TTS-12Hz-1.7B-CustomVoice"

TEXT = (
    "This is a throwaway test of the Qwen three text to speech model running "
    "on an Intel Arc Pro B70 graphics card. It synthesizes speech from text, "
    "which would make it suitable for reading chat replies aloud in a home "
    "server setup. The sentence is long enough to exercise the generator for "
    "several seconds of audio."
)

SHORT = "Compiling the kernels now. This short sentence absorbs the warm-up."

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
    warmup_decode(wrapper)

    talker = wrapper.model.talker
    real_backbone = talker.model  # keep for the graph-off side

    # Attach BOTH graphs. No torch.compile involved.
    talker.model = GraphedTalkerStep(real_backbone, max_len=2048)
    talker._cp_graph = GraphedCodePredictor(talker, torch.device("xpu"), torch.bfloat16)

    # Eager kernel warm-up (no inductor absorb expected).
    t_c = time.monotonic()
    wrapper.generate_custom_voice(text=SHORT, language="English", speaker="Aiden",
                                  non_streaming_mode=True)
    torch.xpu.synchronize()
    print(f"WARM-RUN: {time.monotonic()-t_c:.1f}s (no inductor)", flush=True)
    xpu_cleanup()

    # ---- Correctness: fully-greedy graph-on vs graph-off ----
    def gen(seed):
        torch.manual_seed(seed)
        torch.xpu.manual_seed_all(seed)
        wavs, sr = wrapper.generate_custom_voice(
            text=TEXT, language="English", speaker="Aiden", non_streaming_mode=True,
            do_sample=False, subtalker_dosample=False)
        torch.xpu.synchronize()
        return wavs[0], sr

    wav_on, sr = gen(42)
    xpu_cleanup()
    # Detach both graphs -> fully eager path.
    talker._cp_graph = None
    talker.model = real_backbone
    wav_off, _ = gen(42)
    # Re-attach for the timed run.
    talker.model = GraphedTalkerStep(real_backbone, max_len=2048)
    talker._cp_graph = GraphedCodePredictor(talker, torch.device("xpu"), torch.bfloat16)
    xpu_cleanup()

    a, b = np.asarray(wav_on, dtype=np.float64), np.asarray(wav_off, dtype=np.float64)
    n = min(len(a), len(b))
    corr = float(np.corrcoef(a[:n], b[:n])[0, 1]) if n > 10 else float("nan")
    mad = float(np.mean(np.abs(a[:n] - b[:n])))
    print(f"CORRECTNESS: dur {len(a)/sr:.2f}s vs {len(b)/sr:.2f}s  corr={corr:.5f}  "
          f"mean_abs_diff={mad:.2e}  rms on/off = {rms(a):.4f}/{rms(b):.4f}", flush=True)

    # ---- Timed profile run (both graphs on) ----
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

    print("=== PROFILE talker-graph=True cp-graph=True compile=False ===", flush=True)
    tot = {}
    for name, xs in stats.items():
        tot[name] = sum(xs)
        print(f"{name:16s} n={len(xs):5d} total={tot[name]*1000:9.1f}ms "
              f"mean={tot[name]/len(xs)*1000:7.2f}ms max={max(xs)*1000:8.1f}ms", flush=True)
    gen = stats["_generate"][0]
    known = sum(v for k, v in tot.items() if k != "_generate")
    print(f"{'_generate':16s} total={gen*1000:9.1f}ms  glue(residual)={(gen-known)*1000:9.1f}ms", flush=True)
    print(f"PEAK MEM: {torch.xpu.max_memory_allocated('xpu')/1e9:.2f} GiB", flush=True)
    sf.write("/out/q3x-talkerg.wav", wavs[0], sr)
    print("SAVED /out/q3x-talkerg.wav", flush=True)


if __name__ == "__main__":
    main()
