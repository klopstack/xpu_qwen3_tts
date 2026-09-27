"""Profile the Qwen3-TTS XPU pipeline: per-stage time breakdown.

Times (with XPU sync per call):
  - talker.model.forward   (the 1.7B transformer body, compiled or eager)
  - code_predictor.model.forward (x7 per main token)
  - speech_tokenizer.decode (Mimi decoder)
  - torch.multinomial       (sampling)
  - residual "glue" = total generate() minus the above

Env: Q3X_COMPILE=1 (default), Q3X_INT8=0, Q3X_COMPILE_MIMI=0
Usage: python q3x-profile.py
"""
import functools
import os
import time

import torch
import soundfile as sf

COMPILE = os.environ.get("Q3X_COMPILE", "1") == "1"
INT8 = os.environ.get("Q3X_INT8", "0") == "1"
COMPILE_MIMI = os.environ.get("Q3X_COMPILE_MIMI", "0") == "1"
DYNAMIC = os.environ.get("Q3X_DYNAMIC", "1") == "1"
AUTOTUNE = os.environ.get("Q3X_AUTOTUNE", "0") == "1"

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
    """Time a module's forward via pre/post hooks (works on compiled modules)."""
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


def apply_compile(wrapper):
    kw = {"dynamic": DYNAMIC, "fullgraph": False, "backend": "inductor"}
    if AUTOTUNE:
        kw["mode"] = "max-autotune"
    talker = wrapper.model.talker
    talker.model = torch.compile(talker.model, **kw)
    talker.code_predictor.model = torch.compile(talker.code_predictor.model, **kw)
    print(f"COMPILE: inductor applied (dynamic={DYNAMIC} autotune={AUTOTUNE})", flush=True)


def apply_int8(wrapper):
    """Port of the fork's weight-only INT8 (talker only)."""
    import torch.nn as nn
    import torch.nn.functional as F

    class _QL(nn.Module):
        def __init__(self, w8, scale, bias):
            super().__init__()
            self.register_buffer("weight_int8", w8)
            self.register_buffer("scale", scale)
            self.bias = nn.Parameter(bias.clone()) if bias is not None else None

        def forward(self, x):
            w = self.weight_int8.to(x.dtype) * self.scale
            return F.linear(x, w, self.bias)

    def walk(parent, min_numel=1024):
        n = 0
        for name, child in list(parent.named_children()):
            if isinstance(child, nn.Linear) and child.weight is not None and child.weight.numel() >= min_numel:
                w = child.weight.data.float()
                scale = (w.abs().max(dim=1, keepdim=True)[0] / 127.0).to(child.weight.dtype)
                w8 = (w / scale.float()).round().clamp(-128, 127).to(torch.int8)
                setattr(parent, name, _QL(w8, scale, child.bias))
                n += 1
            else:
                n += walk(child, min_numel)
        return n

    n = walk(wrapper.model.talker)
    print(f"INT8: {n} Linear layers quantized", flush=True)


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


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * p))]


def report(label):
    print(f"=== PROFILE {label} ===", flush=True)
    tot = {}
    for name, xs in stats.items():
        tot[name] = sum(xs)
        print(f"{name:16s} n={len(xs):5d} total={tot[name]*1000:9.1f}ms "
              f"mean={tot[name]/len(xs)*1000:7.2f}ms p50={pct(xs,.5)*1000:7.2f}ms "
              f"max={max(xs)*1000:8.1f}ms", flush=True)
    gen = stats.get("_generate", [0.0])[0]
    known = sum(v for k, v in tot.items() if k != "_generate")
    print(f"{'_generate':16s} total={gen*1000:9.1f}ms  glue(residual)={(gen-known)*1000:9.1f}ms", flush=True)
    print(flush=True)


def main():
    assert torch.xpu.is_available()

    wrapper = Qwen3TTSModel.from_pretrained(
        MODEL_DIR, device_map="xpu", dtype=torch.bfloat16, attn_implementation="sdpa",
    )
    ensure_on_xpu(wrapper)
    if INT8:
        apply_int8(wrapper)
    if COMPILE:
        apply_compile(wrapper)
    if COMPILE_MIMI:
        st = wrapper.model.speech_tokenizer
        st.model = torch.compile(st.model, dynamic=True, fullgraph=False, backend="inductor")
        print("COMPILE: inductor applied to Mimi decoder", flush=True)
    warmup_decode(wrapper)

    # Instrument AFTER compile so we time the compiled units as black boxes.
    talker = wrapper.model.talker
    hook_module(talker.model, "talker")
    hook_module(talker.code_predictor.model, "code_pred")
    st = wrapper.model.speech_tokenizer
    timed_method(st, "decode", "mimi")
    patch_multinomial()

    if COMPILE:
        t_c = time.monotonic()
        wrapper.generate_custom_voice(text=SHORT, language="English", speaker="Aiden",
                                      non_streaming_mode=True)
        torch.xpu.synchronize()
        print(f"COMPILE-ABSORB: {time.monotonic()-t_c:.1f}s", flush=True)
    xpu_cleanup()
    stats.clear()

    # Warm (timed) run.
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    wavs, sr = wrapper.generate_custom_voice(text=TEXT, language="English", speaker="Aiden",
                                             non_streaming_mode=True)
    torch.xpu.synchronize()
    gen_secs = time.perf_counter() - t0
    stats["_generate"] = [gen_secs]
    dur = len(wavs[0]) / sr
    print(f"AUDIO: {dur:.1f}s in {gen_secs:.1f}s -> RTF {gen_secs/dur:.2f}x", flush=True)
    report(f"compile={COMPILE} dyn={DYNAMIC} autotune={AUTOTUNE} int8={INT8} mimi={COMPILE_MIMI}")
    print(f"PEAK MEM: {torch.xpu.max_memory_allocated('xpu')/1e9:.2f} GiB", flush=True)
    sf.write("/out/q3x-profile.wav", wavs[0], sr)
    print("SAVED /out/q3x-profile.wav", flush=True)


if __name__ == "__main__":
    main()
