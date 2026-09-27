# coding=utf-8
"""XPU graph-captured code-predictor for Qwen3-TTS.

Collapses the 15-step code-predictor sequence (1 prefill + 15 decodes, one per
main token) into a single torch.xpu.XPUGraph replay with static buffers.

Requirements satisfied by the model's structure:
  - every step has fixed shapes (batch=1, seq=1 decode / seq=2 prefill)
  - no data-dependent control flow (greedy argmax is graph-safe)
  - KV cache is tiny (<=17 positions) -> static buffer, in-place writes

Attach to the talker wrapper:  talker._cp_graph = GraphedCodePredictor(talker, ...)
The manual loop in Qwen3TTSTalkerForConditionalGeneration.forward checks for it.
"""
import torch
from transformers.cache_utils import Cache


class StaticKVCache(Cache):
    """Fixed-size KV cache with in-place writes (graph-capture safe).

    Subclasses transformers Cache (the CP model forward type-checks it) and
    implements the subset of the interface the model uses:
    update(key, value, layer_idx, cache_kwargs) -> (k_full, v_full) and
    get_seq_length(). All writes are in-place into preallocated buffers so
    the captured graph sees stable addresses.
    """

    def __init__(self, batch, max_len, n_kv_heads, head_dim, device, dtype):
        # Skip Cache.__init__ (requires layers/layer_class_to_replicate); only the
        # isinstance(...) check in the model forward relies on the subclass.
        self.k = torch.zeros(batch, n_kv_heads, max_len, head_dim, device=device, dtype=dtype)
        self.v = torch.zeros_like(self.k)
        self._len = 0
        self._max_len = max_len

    def update(self, key_states, value_states, layer_idx=None, cache_kwargs=None):
        # Layout: (batch, n_kv_heads, seq, head_dim)
        t = key_states.shape[-2]
        i = self._len
        self.k[:, :, i:i + t, :].copy_(key_states)
        self.v[:, :, i:i + t, :].copy_(value_states)
        self._len += t
        return self.k[:, :, :self._len, :], self.v[:, :, :self._len, :]

    def get_seq_length(self):
        return self._len

    def get_mask_sizes(self, cache_position, layer_idx: int = 0):
        # Called by create_causal_mask BEFORE the update(); report the
        # post-update length (matches DynamicCache semantics).
        return self._len + cache_position.shape[0], 0

    def get_max_cache_shape(self, layer_idx: int = 0):
        return self._max_len

    def get_max_batch_size(self, layer_idx: int = 0):
        return self.k.shape[0]

    def reset(self):
        self._len = 0


class GraphedCodePredictor:
    """Captures the full code-predictor sequence as one XPUGraph.

    Inputs (copied in per main token): past_hidden (1,1,H), last_id_hidden (1,1,H)
    Output: predicted codebooks 1..N-1 as (1, N-1) long tensor.
    """

    def __init__(self, talker, device, dtype, warmup_iters=3):
        self.talker = talker
        self.cp = talker.code_predictor
        self.device = device
        self.dtype = dtype
        self.n_steps = self.cp.config.num_code_groups - 1
        # Prefill input arrives in TALKER hidden space (past_hidden/last_id_hidden);
        # the small_to_mtp_projection to cp hidden happens inside cp.forward.
        hs_in = talker.config.hidden_size
        self.prefill_buf = torch.zeros(1, 2, hs_in, device=device, dtype=dtype)
        self.out_buf = torch.zeros(1, self.n_steps, dtype=torch.long, device=device)
        n_kv = self.cp.config.num_key_value_heads
        hd = getattr(self.cp.config, "head_dim",
                     self.cp.config.hidden_size // self.cp.config.num_attention_heads)
        self.kv = StaticKVCache(1, 2 + self.n_steps, n_kv, hd, device, dtype)
        self.graph = None
        self._capture(warmup_iters)

    def _seq(self):
        """Run the full 15-step sequence against the static buffers (trace or replay-prep)."""
        cp = self.cp
        kv = self.kv
        kv._len = 0
        out = self.out_buf
        o = cp(inputs_embeds=self.prefill_buf, use_cache=True, past_key_values=kv,
               output_hidden_states=False, output_attentions=False)
        tok = o.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        out[:, 0:1].copy_(tok)
        step = o.generation_steps
        for ci in range(1, self.n_steps):
            o = cp(input_ids=tok, past_key_values=kv, use_cache=True,
                   output_hidden_states=False, output_attentions=False, generation_steps=step)
            tok = o.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            out[:, ci:ci + 1].copy_(tok)
            step = o.generation_steps
        return out

    def _capture(self, warmup_iters):
        g = torch.xpu.XPUGraph()
        # Warm up on a side stream (standard graph-capture protocol) so that
        # allocations/compiles happen outside the capture.
        try:
            side = torch.xpu.Stream()
            side.wait_stream(torch.xpu.current_stream())
            with torch.xpu.stream(side):
                for _ in range(warmup_iters):
                    self._seq()
            torch.xpu.current_stream().wait_stream(side)
        except Exception as e:
            print(f"GRAPH: side-stream warmup unavailable ({type(e).__name__}); warming on current stream", flush=True)
            for _ in range(warmup_iters):
                self._seq()
        torch.xpu.synchronize()
        try:
            with torch.xpu.graph(g):
                self._seq()
        except TypeError:
            g.capture_begin()
            self._seq()
            g.capture_end()
        self.graph = g
        print(f"GRAPH: code-predictor captured ({self.n_steps} steps -> 1 replay/main token)", flush=True)

    def run(self, past_hidden, last_id_hidden):
        self.prefill_buf[:, 0:1].copy_(past_hidden)
        self.prefill_buf[:, 1:2].copy_(last_id_hidden)
        self.graph.replay()
        return self.out_buf
