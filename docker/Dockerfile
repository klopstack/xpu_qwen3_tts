# Throwaway Phase-0 spike: Qwen3-TTS 1.7B on XPU with the allanmeng/ComfyUI-Qwen3TTS-XPU
# optimization stack (vendored patched qwen_tts package: manual code-predictor loop,
# Mimi bucket padding, transformers-5 compat) + torch.compile inductor + decode warmup.
# Base = production vLLM-XPU image (torch 2.13.0+xpu, transformers 5.15.0, gcc, triton 3.7.2).
FROM vllm/vllm-openai-xpu@sha256:f01e24f6c7ff01f1e0662234255a1372297d1dbd89d003cf13c8fad3eab1ba4f

# The fork's vendored package targets transformers 4.57.x (masking_utils
# input_embeds/cache_position API, ROPE_INIT_FUNCTIONS['default']). Pin it;
# vLLM import breaks in this image — irrelevant, vLLM is unused in the spike.
RUN pip install --no-cache-dir "transformers==4.57.3"

COPY qwen_tts /opt/q3x/qwen_tts

ENV PYTHONPATH=/opt/q3x

RUN python - <<'EOF'
import torch
major, minor = (int(x) for x in torch.__version__.split(".")[:2])
assert (major, minor) >= (2, 13), f"unexpected torch {torch.__version__}"
import qwen_tts
from qwen_tts import Qwen3TTSModel
print("torch OK:", torch.__version__, "| qwen_tts imported from patched package")
EOF
