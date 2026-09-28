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


class _PlaceholderLayer:
    """Stand-in for a StaticCache layer entry (see StaticKVCache.layers)."""
    is_compileable = False


_PLACEHOLDER_LAYER = _PlaceholderLayer()


class StaticKVCache(Cache):
    """Fixed-size KV cache with in-place writes (graph-capture safe).

    Subclasses transformers Cache (the CP model forward type-checks it) and
    implements the subset of the interface the model uses:
    update(key, value, layer_idx, cache_kwargs) -> (k_full, v_full) and
    get_seq_length(). All writes are in-place into preallocated buffers so
    the captured graph sees stable addresses.
    """

    def __init__(self, num_layers, batch, max_len, n_kv_heads, head_dim, device, dtype,
                 return_full=False):
        # Skip Cache.__init__ (requires layers/layer_class_to_replicate); only the
        # isinstance(...) check in the model forward relies on the subclass.
        # PER-LAYER buffers: each decoder layer owns its K/V rows. Sharing one
        # buffer across layers corrupts the cache from decode step 2 onward
        # (each layer's update overwrites the others' past positions).
        self.k = torch.zeros(num_layers, batch, n_kv_heads, max_len, head_dim,
                             device=device, dtype=dtype)
        self.v = torch.zeros_like(self.k)
        self._len = 0
        self._max_len = max_len
        # Explicit write position (talker graph: mutable cache_position buffer).
        # None -> derive from _len (CP graph: constant layout).
        self._write_pos = None
        # Incoming tokens for the in-flight forward (talker prefill: the mask
        # builder asks get_mask_sizes BEFORE update() runs).
        self._incoming = 0
        # return_full: True -> update() returns the FULL max_len buffer (talker
        # graph: an explicit additive mask spans the whole buffer). False ->
        # returns the growing :_len view (CP graph: create_causal_mask sizes the
        # mask to get_mask_sizes()==_len, so K/V length must equal _len).
        self.return_full = return_full

    def update(self, key_states, value_states, layer_idx=None, cache_kwargs=None):
        # Layout: (batch, n_kv_heads, seq, head_dim) per layer.
        # Writes at the CURRENT cache_position (mutable buffer for the talker
        # graph; equals _len for the constant-layout CP graph).
        t = key_states.shape[-2]
        i = self._write_pos if self._write_pos is not None else self._len
        self.k[layer_idx, :, :, i:i + t, :].copy_(key_states)
        self.v[layer_idx, :, :, i:i + t, :].copy_(value_states)
        self._len = max(self._len, i + t)
        self._incoming = 0
        if self.return_full:
            # Stable full-length buffers; future positions are hidden by the
            # caller's additive mask.
            return self.k[layer_idx], self.v[layer_idx]
        return (self.k[layer_idx, :, :, :self._len, :],
                self.v[layer_idx, :, :, :self._len, :])

    @property
    def layers(self):
        # transformers 4.57.3 touches this in two places:
        #  - _preprocess_mask_arguments: `layer_idx >= len(pkv.layers)` (length
        #    probe for not-yet-created DynamicCache layers)
        #  - Cache.is_compileable: `len(self.layers) == 0` then
        #    `all(layer.is_compileable ...)` (called by
        #    prepare_inputs_for_generation every decode step)
        # Report one placeholder per real layer. is_compileable=False keeps
        # prepare_inputs_for_generation from pre-building a 4D mask each step
        # (the graph supplies its own static mask).
        return [_PLACEHOLDER_LAYER] * self.k.shape[0]

    def get_seq_length(self):
        return self._len

    def get_mask_sizes(self, cache_position, layer_idx: int = 0):
        # Called by create_causal_mask BEFORE the update(); report the
        # post-update length (matches DynamicCache semantics). _incoming covers
        # the talker prefill, where the mask is built before any update() ran.
        return self._len + self._incoming, 0

    def get_max_cache_shape(self, layer_idx: int = 0):
        return self._max_len

    def get_max_batch_size(self, layer_idx: int = 0):
        return self.k.shape[1]

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
        self.kv = StaticKVCache(self.cp.config.num_hidden_layers, 1, 2 + self.n_steps,
                                n_kv, hd, device, dtype)
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


class GraphedTalkerStep(torch.nn.Module):
    """Drop-in replacement for the talker's inner model (talker.model).

    - Prefill (seq > 1): routed to the real backbone eagerly (variable length,
      HF DynamicCache), and the resulting KV is adopted into a static cache.
    - Decode (seq == 1): ONE captured XPUGraph replay per step with static
      per-layer KV, static I/O buffers, and a precomputed additive
      attention-mask table (passed as a dict so the model's forward skips
      create_causal_mask entirely).

    This removes torch.compile from the talker entirely: no inductor absorb,
    no dynamo guards, no retracing.

    Attach:  talker.model = GraphedTalkerStep(talker.model, max_len=2048)
    """

    def __init__(self, real_model, max_len=2048, warmup_iters=3):
        super().__init__()
        self.real = real_model  # registered submodule -> .to()/params still work
        cfg = real_model.config
        self.max_len = max_len
        device = next(real_model.parameters()).device
        dtype = next(real_model.parameters()).dtype
        self.dtype = dtype
        n_layers = cfg.num_hidden_layers
        n_kv = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
        hd = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        hs = cfg.hidden_size

        self.kv = StaticKVCache(n_layers, 1, max_len, n_kv, hd, device, dtype,
                                return_full=True)
        self.input_buf = torch.zeros(1, 1, hs, device=device, dtype=dtype)
        self.out_buf = torch.zeros(1, 1, hs, device=device, dtype=dtype)
        self.cache_pos_buf = torch.zeros(1, dtype=torch.long, device=device)
        # Float: the wrapper's decode path builds position_ids as
        # arange + rope_deltas (float), matching the eager path exactly.
        self.pos_ids_buf = torch.zeros(3, 1, 1, dtype=torch.float32, device=device)
        # Additive mask buffer: (1, 1, 1, max_len); copied from the table per step.
        self.attn_mask_buf = torch.zeros(1, 1, 1, max_len, device=device, dtype=dtype)
        self.mask_table = None
        self.graph = None
        self._ready = False
        self._prefill_mask = None
        self._capture(warmup_iters)

    # ------------------------------------------------------------------
    def _build_mask_table(self, attention_mask):
        """Precompute one additive (1,1,1,max_len) mask per decode position.

        mask[p][k] = 0 if k <= p and not padded, else finfo.min. Vectorized.
        """
        dev = self.attn_mask_buf.device
        dt = self.dtype
        minv = torch.finfo(dt).min
        key_pos = torch.arange(self.max_len, device=dev)
        allowed = key_pos.unsqueeze(0) <= key_pos.unsqueeze(1)  # (M, M): k <= p
        if attention_mask is not None:
            pad = attention_mask[0].to(torch.bool)[: self.max_len]
            if pad.shape[0] < self.max_len:
                pad = torch.cat([pad, torch.zeros(self.max_len - pad.shape[0],
                                                  dtype=torch.bool, device=dev)])
            allowed = allowed & pad.unsqueeze(0)
        self.mask_table = torch.where(
            allowed, torch.zeros((), dtype=dt, device=dev),
            torch.full((), minv, dtype=dt, device=dev),
        ).view(self.max_len, 1, 1, 1, self.max_len)
        self.attn_mask_buf.copy_(self.mask_table[0])

    def _decode_step(self):
        """The captured single-token decode (all inputs from static buffers).

        The inner talker model has no dict early-exit, so we pass a 4D additive
        mask tensor directly: create_causal_mask returns a 4D mask as-is.
        """
        out = self.real(
            input_ids=None,
            attention_mask=self.attn_mask_buf,
            position_ids=self.pos_ids_buf,
            past_key_values=self.kv,
            inputs_embeds=self.input_buf,
            use_cache=True,
            output_attentions=False,
            output_hidden_states=False,
            cache_position=self.cache_pos_buf,
        )
        self.out_buf.copy_(out.last_hidden_state)

    def _capture(self, warmup_iters):
        from transformers.modeling_outputs import BaseModelOutputWithPast

        # Warmup at position 0 with an all-ones mask (table not needed yet:
        # attn_mask_buf is already all-zero = attend-everywhere, which is a
        # superset of causal and exercises the same kernels).
        self.kv._len = 0
        self.kv._write_pos = None
        for _ in range(warmup_iters):
            self._decode_step()
        torch.xpu.synchronize()
        g = torch.xpu.XPUGraph()
        try:
            with torch.xpu.graph(g):
                self._decode_step()
        except TypeError:
            g.capture_begin()
            self._decode_step()
            g.capture_end()
        self.graph = g
        self._out = BaseModelOutputWithPast(
            last_hidden_state=self.out_buf, past_key_values=self.kv
        )
        print("GRAPH: talker decode step captured (1 replay/main token)", flush=True)

    # ------------------------------------------------------------------
    def _adopt(self, dyn_cache):
        """Copy an HF DynamicCache (from eager prefill) into the static cache."""
        self.kv._len = 0
        seq_len = 0
        for li in range(self.kv.k.shape[0]):
            k, v = dyn_cache.layers[li].keys, dyn_cache.layers[li].values
            seq_len = k.shape[2]
            self.kv.k[li, :, :, :seq_len, :].copy_(k)
            self.kv.v[li, :, :, :seq_len, :].copy_(v)
        self.kv._len = seq_len
        self._ready = True

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, use_cache=None,
                output_attentions=None, output_hidden_states=None,
                cache_position=None, **kwargs):
        # Prefill: variable length -> real backbone, eager.
        if inputs_embeds is not None and inputs_embeds.shape[1] > 1:
            if inputs_embeds.shape[1] > self.max_len:
                raise RuntimeError(
                    f"talker prefill {inputs_embeds.shape[1]} exceeds max_len {self.max_len}")
            self._prefill_mask = attention_mask
            self._ready = False
            if self.mask_table is None:
                self._build_mask_table(attention_mask)
            return self.real(
                input_ids=None, attention_mask=attention_mask, position_ids=position_ids,
                past_key_values=past_key_values, inputs_embeds=inputs_embeds,
                use_cache=use_cache, output_attentions=output_attentions,
                output_hidden_states=output_hidden_states, cache_position=cache_position,
                **kwargs,
            )
        # Decode: adopt prefill KV once, then replay.
        if not self._ready:
            self._adopt(past_key_values)
        self.input_buf.copy_(inputs_embeds)
        self.pos_ids_buf.copy_(position_ids)
        self.cache_pos_buf.copy_(cache_position)
        pos = int(cache_position[0].item())
        if pos >= self.max_len:
            raise RuntimeError(f"talker decode position {pos} exceeds max_len {self.max_len}")
        self.kv._write_pos = self.cache_pos_buf  # write at the live position
        self.attn_mask_buf.copy_(self.mask_table[pos])
        self.graph.replay()
        return self._out

    def __getattr__(self, name):
        # nn.Module.__getattr__ resolves _modules/_parameters/_buffers (incl.
        # 'real'); delegate anything else (config, rotary_emb, norm, layers,
        # gradient_checkpointing, ...) to the wrapped model.
        try:
            return super().__getattr__(name)
        except AttributeError:
            real = (self.__dict__.get("_modules") or {}).get("real")
            if real is not None:
                return getattr(real, name)
            raise
