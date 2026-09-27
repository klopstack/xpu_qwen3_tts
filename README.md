# xpu_qwen3_tts

Real-time [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) inference on **Intel Arc XPU**
(Arc Pro B70, Level Zero), ported out of ComfyUI.

**Result: RTF 0.41× — 2.4× faster than real-time** for the 1.7B CustomVoice model
(29.6 tokens/s vs the 12 tokens/s the 12 Hz codec needs), peak VRAM 5.0 GiB.

## What makes it fast

Three stacked optimizations, all standard PyTorch (no ComfyUI, no custom kernels):

1. **Manual code-predictor loop** — replaces HF `generate()` for the 15-step
   code-predictor sequence that runs per main token (eliminates logits-processor
   and stopping-criteria overhead).
2. **`torch.compile` (inductor, `max-autotune`, `dynamic=True`)** on the talker
   backbone — absorbs ~198 s of stock overhead into fused kernels.
3. **XPUGraph capture of the code-predictor** — the entire 15-step sequence
   (1 prefill + 15 decodes) is captured as **one `torch.xpu.XPUGraph`** with a
   static in-place KV cache and greedy sampling inside the graph. One replay per
   main token instead of ~500 kernel launches. This alone cuts the
   code-predictor from 33 ms to 14.7 ms (2.2×).

Plus: Mimi decoder bucket-padding warmup (T = 64..512), explicit XPU placement,
`empty_cache` between generations.

### Performance ladder (Arc Pro B70, 1.7B CustomVoice, bf16, sdpa, warm RTF)

| Stage | RTF | Notes |
|---|---|---|
| Stock `qwen-tts` | 2.62× | |
| + manual CP loop + Mimi bucket pad + warmup | 2.28× | |
| + inductor `dynamic=True` | 1.21× | |
| + `max-autotune` | 0.95× | real-time |
| + **XPUGraph code-predictor** | **0.41×** | 2.4× real-time |

Rejected variants: `dynamic=False` (2.05×, prefill retracing storm), INT8
weight-only (1.47×, unfused dequant casts), capturing the *compiled*
code-predictor (dynamo cannot fake-tensor-trace the in-place static cache —
capture the **eager** CP instead; the dispatch cost is what the replay removes).

## Repo layout

```
qwen_tts/            vendored qwen-tts package (transformers 4.57.x API) with:
  xpu_graph.py         StaticKVCache + GraphedCodePredictor (XPUGraph capture)
  core/models/
    modeling_qwen3_tts.py   manual CP loop (_manual_cp_loop) + _cp_graph dispatch
docker/Dockerfile    vllm-openai-xpu base + pinned transformers==4.57.3
scripts/
  q3x-spike.py         baseline + optimized RTF harness
  q3x-profile.py       per-stage profiler (talker / CP / Mimi / sampling / glue)
  q3x-graph.py         graph-capture spike with seeded correctness check
```

## Requirements

- Intel Arc GPU (tested: Arc Pro B70, 32 GB) with Level Zero drivers
- `torch >= 2.13.0+xpu` (image: `vllm/vllm-openai-xpu`)
- **`transformers==4.57.3`** — the vendored package targets the 4.57.x masking
  API (`input_embeds` + `cache_position` in `create_causal_mask`,
  `ROPE_INIT_FUNCTIONS['default']`). transformers 5.x breaks it.
- Model: `Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice` (the `speech_tokenizer/`
  subfolder ships inside the model dir — no separate download needed)

## Quick start

```bash
cd docker && docker build -t xpu-qwen3-tts .
docker run --rm --entrypoint python \
  --device /dev/dri -v /dev/dri/by-path:/dev/dri/by-path \
  -e ZE_AFFINITY_MASK=2 -e ONEAPI_DEVICE_SELECTOR=level_zero:gpu \
  -v /path/to/Qwen3-TTS-12Hz-1.7B-CustomVoice:/models/Qwen3-TTS-12Hz-1.7B-CustomVoice:ro \
  -v /tmp/xpu-qwen3-tts-out:/out \
  xpu-qwen3-tts /scripts/q3x-graph.py
```

First run absorbs ~16 min of inductor autotune (persist
`TORCHINDUCTOR_CACHE_DIR` to a volume to amortize across restarts).

## Attaching the graph (integration sketch)

```python
from qwen_tts import Qwen3TTSModel
from qwen_tts.xpu_graph import GraphedCodePredictor

wrapper = Qwen3TTSModel.from_pretrained(MODEL_DIR, device_map="xpu",
                                        dtype=torch.bfloat16,
                                        attn_implementation="sdpa")
# ... place on xpu, torch.compile(talker.model, mode="max-autotune", dynamic=True)
# ... warm up Mimi decode buckets ...
wrapper.model.talker._cp_graph = GraphedCodePredictor(
    wrapper.model.talker, torch.device("xpu"), torch.bfloat16)
```

The vendored talker dispatches to `_cp_graph.run(past_hidden, last_id_hidden)`
when the attribute is set, and falls back to the eager manual loop otherwise.

## Known caveats

- **Greedy code-predictor in the graph.** The captured sequence uses argmax for
  codebooks 1..15 (data-dependent sampling is not graph-safe). The talker's
  first codebook token is still sampled. Quality delta vs
  `subtalker_dosample=True` (top-k 50, T=0.9) is uncharacterized — listen to
  the samples.
- **Static shapes.** Batch size 1, ≤17 CP positions. Fine for single-stream
  TTS; batching would need per-bucket graphs.
- **Autotune absorb is cold-start pain.** ~16 min on first run per shape family.
- Power caps matter: on a 3-card box capped at 190 W/card, lifting the cap
  during TTS buys ~15–17 %.

## Attribution

- Model: [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) (Apache-2.0)
- Optimization techniques ported from
  [allanmeng/ComfyUI-Qwen3-TTS-XPU](https://github.com/allanmeng/ComfyUI-Qwen3TTS-XPU)
  (MIT, see `LICENSE-UPSTREAM-MIT`)
- Graph-capture design cross-checked against
  [andimarafioti/faster-qwen3-tts](https://github.com/andimarafioti/faster-qwen3-tts)
  (MIT) — they graph both talker and predictor on CUDA; we graph the predictor
  on XPU and keep the talker inductor-compiled.
