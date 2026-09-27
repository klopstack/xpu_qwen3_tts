"""Throwaway Phase-0 spike: Qwen3-TTS 1.7B CustomVoice on Intel XPU with the
allanmeng/ComfyUI-Qwen3TTS-XPU optimization stack:

  - vendored patched qwen_tts (manual code-predictor loop, Mimi bucket padding,
    transformers-5 compat) — active automatically via PYTHONPATH
  - explicit XPU placement (incl. speech_tokenizer inner model)
  - torch.compile(inductor) on talker.model + code_predictor.model
  - Mimi decoder kernel warmup (bucket sizes 64..512)
  - empty_cache between generations (fork's OOM mitigation)

Measures: load time, compile-absorb time, cold/warm RTF, peak VRAM.
Usage: python q3x-spike.py
"""
import gc
import os
import time

import functools

import torch
import soundfile as sf

COMPILE = os.environ.get("Q3X_COMPILE", "0") == "1"

# The fork's tokenizer does:
#   try: from transformers.utils.generic import merge_with_config_defaults as check_model_inputs
#   except ImportError: check_model_inputs = <4.x check_model_inputs>  (a FACTORY — breaks bare @decorator)
# On transformers 4.57.3 the 5.x name is absent, so the broken fallback fires.
# Bridge: provide a passthrough merge_with_config_defaults so the fork's primary path works.
import transformers.utils.generic as _g

if not hasattr(_g, "merge_with_config_defaults"):
    def merge_with_config_defaults(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            return func(self, *args, **kwargs)

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


def ensure_on_xpu(wrapper):
    """Port of the fork's _ensure_on_xpu: force main model + speech tokenizer to XPU."""
    dev = torch.device("xpu")
    model = wrapper.model
    cur = next(model.parameters()).device
    if cur.type != "xpu":
        print(f"PLACEMENT: main model on {cur}, moving to xpu", flush=True)
        model.to(dev)
    wrapper.device = dev
    st = getattr(model, "speech_tokenizer", None)
    if st is not None:
        st_model = getattr(st, "model", None)
        if st_model is not None:
            scur = next(st_model.parameters()).device
            if scur.type != "xpu":
                print(f"PLACEMENT: speech tokenizer on {scur}, moving to xpu", flush=True)
                st_model.to(dev)
        st.device = dev
    print("PLACEMENT: all components on xpu", flush=True)


def apply_compile(wrapper):
    """Port of the fork's _apply_torch_compile (inductor backend)."""
    kw = {"dynamic": True, "fullgraph": False, "backend": "inductor"}
    talker = wrapper.model.talker
    talker.model = torch.compile(talker.model, **kw)
    cp = talker.code_predictor
    cp.model = torch.compile(cp.model, **kw)
    print("COMPILE: inductor applied to talker.model + code_predictor.model", flush=True)


def warmup_decode(wrapper):
    """Port of the fork's _warmup_xpu_decode: pre-JIT Mimi kernels per bucket shape."""
    buckets = [64, 128, 192, 256, 320, 384, 448, 512]
    st = wrapper.model.speech_tokenizer
    device = next(st.model.parameters()).device
    n_q = getattr(st.model.config, "num_quantizers", 16)
    with torch.inference_mode():
        for bt in buckets:
            dummy = torch.zeros(bt, n_q, dtype=torch.long, device=device)
            st.decode([{"audio_codes": dummy}])
            torch.xpu.synchronize()
    print(f"WARMUP: Mimi decoder kernels primed for T={buckets[0]}..{buckets[-1]}", flush=True)


def xpu_cleanup():
    """Port of the fork's _xpu_cleanup."""
    torch.xpu.synchronize()
    torch.xpu.empty_cache()
    gc.collect()


def main():
    assert torch.xpu.is_available(), "XPU not available"

    t0 = time.monotonic()
    wrapper = Qwen3TTSModel.from_pretrained(
        MODEL_DIR,
        device_map="xpu",
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    )
    print(f"LOADED in {time.monotonic()-t0:.1f}s", flush=True)

    ensure_on_xpu(wrapper)
    if COMPILE:
        apply_compile(wrapper)
    warmup_decode(wrapper)

    torch.xpu.synchronize()
    print(f"MEM after load: {torch.xpu.memory_allocated('xpu')/1e9:.2f} GiB", flush=True)

    if COMPILE:
        # Absorb inductor trace + Triton kernel compilation with a short generation.
        t_c = time.monotonic()
        wrapper.generate_custom_voice(text=SHORT, language="English", speaker="Aiden",
                                      non_streaming_mode=True)
        torch.xpu.synchronize()
        print(f"COMPILE-ABSORB: {time.monotonic()-t_c:.1f}s", flush=True)
        xpu_cleanup()

    def run(tag):
        torch.xpu.synchronize()
        t_gen = time.monotonic()
        wavs, sr = wrapper.generate_custom_voice(
            text=TEXT, language="English", speaker="Aiden", non_streaming_mode=True,
        )
        torch.xpu.synchronize()
        gen_secs = time.monotonic() - t_gen
        dur = len(wavs[0]) / sr
        print(f"[{tag}] AUDIO: {dur:.1f}s in {gen_secs:.1f}s -> RTF {gen_secs/dur:.2f}x", flush=True)
        xpu_cleanup()
        return wavs[0], sr

    run("cold")
    wav, sr = run("warm")

    print(f"PEAK MEM: {torch.xpu.max_memory_allocated('xpu')/1e9:.2f} GiB", flush=True)
    sf.write("/out/q3x-spike.wav", wav, sr)
    print("SAVED /out/q3x-spike.wav", flush=True)


if __name__ == "__main__":
    main()
