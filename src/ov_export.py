"""Masked-LM forward as a static-shape graph for OpenVINO (CPU, integrated GPU, NPU): token ids [B, S] -> logits of the first int(0.3 S) positions
(the masked 30% of the evaluations; the position does not change the cost). Attention masks are baked in as constants (no padding: a latency
benchmark runs full sequences), the loop is unrolled to the requested depth and the shared blocks are one set of weights."""
import torch
from torch import nn

MASK = 0.3


class MaskedLogits(nn.Module):
    def __init__(self, model, seq, depth=None):
        super().__init__()
        self.model, self.depth, self.n = model.eval(), depth, int(seq * MASK)
        kind = model.config.model_type
        self.kind = "bort" if kind == "bort" else kind
        base = model.base_model
        self.has_masks = kind in ("bort", "modernbert")
        if kind in ("bort", "modernbert"):
            from transformers.masking_utils import create_bidirectional_mask, create_bidirectional_sliding_window_mask

            dummy = torch.zeros(1, seq, model.config.hidden_size)
            args = dict(config=model.config, inputs_embeds=dummy, attention_mask=torch.ones(1, seq, dtype=torch.long))
            masks = {"full_attention": create_bidirectional_mask(**args), "sliding_attention": create_bidirectional_sliding_window_mask(**args)}
            self.mask_kinds = [k for k, v in masks.items() if v is not None]  # a mask that is None (no masking needed) stays None
            for k in self.mask_kinds:
                self.register_buffer("mask_" + k, masks[k], persistent=False)  # buffers follow .to(device)
        self.base = base

    def encode(self, ids):
        kwargs = {} if self.depth is None else {"recurrence_depth": self.depth}
        if self.has_masks:
            masks = {k: getattr(self, "mask_" + k) if k in self.mask_kinds else None for k in ("full_attention", "sliding_attention")}
            return self.base(input_ids=ids, attention_mask=masks, **kwargs).last_hidden_state
        return self.base(input_ids=ids, attention_mask=torch.ones_like(ids)).last_hidden_state

    def forward(self, ids):
        h = self.encode(ids)[:, : self.n]
        m, kind = self.model, self.kind
        if kind in ("bort", "modernbert"):
            return m.decoder(m.head(h))
        if kind == "roberta":
            return m.lm_head(h)
        if kind == "distilbert":
            return m.vocab_projector(m.vocab_layer_norm(m.activation(m.vocab_transform(h))))
        if kind == "deberta":
            return m.cls(h)
        raise ValueError(kind)
