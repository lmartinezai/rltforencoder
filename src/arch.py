# Cell 1

# List of things I assumed I would need but didn't end up needing that are kept in to slow down your imports.

import torch
import math
from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F
from transformers import AutoConfig, AutoModel, AutoModelForMaskedLM
from transformers.masking_utils import create_bidirectional_mask
from transformers.modeling_outputs import ModelOutput
from transformers.masking_utils import create_bidirectional_mask, create_bidirectional_sliding_window_mask
from transformers.modeling_outputs import BaseModelOutput

from transformers.models.modernbert.configuration_modernbert import ModernBertConfig
from transformers.models.modernbert.modeling_modernbert import(
    ModernBertEmbeddings,       ModernBertEncoderLayer,
    ModernBertPredictionHead,   ModernBertPreTrainedModel,
    ModernBertRotaryEmbedding, ModernBertModel, ModernBertForMaskedLM
)
from transformers import PreTrainedModel

# Cell 2

class BortConfig(ModernBertConfig):
    model_type = "bort"

    hidden_size: int = 1024
    intermediate_size: int = 2624
    num_attention_heads: int = 16
    recurrence_depth: int = 1 # autohalt not in yet.

    # The blocks are a prelude, a looped core run `recurrence_depth` times, and a coda; num_hidden_layers is their sum (2 + 4 + 2 = 8 by default).
    prelude_layers: int = 2
    looped_layers: int = 4
    coda_layers: int = 2
    num_hidden_layers: int = 8
    merge_alpha: float = 0.1
    use_merge: bool = True # False: the looped core gets the plain state back each iteration (no gated merge)
    merge_base: str = "state" # what the gated feedback is added to: "state" (the loop state, the original Bort) | "anchor" (the prelude output: RLT's form, input injection)

    def __post_init__(self, **kwargs):
        if min(self.prelude_layers, self.looped_layers, self.coda_layers) < 1:
            raise ValueError("prelude, looped core and coda each need at least one block.")
        self.num_hidden_layers = self.prelude_layers + self.looped_layers + self.coda_layers
        if self.layer_types is None:
            # Full attention on the first block, on both ends of the looped core and on the last block (0, 2, 5, 7 for 2 + 4 + 2).
            full = {0, self.prelude_layers, self.prelude_layers + self.looped_layers - 1, self.num_hidden_layers - 1}
            self.layer_types = ["full_attention" if i in full else "sliding_attention" for i in range(self.num_hidden_layers)]

        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("layer_types must name every block.")
        
        super().__post_init__(**kwargs)


def MakeBortConfig(checkpoint="answerdotai/ModernBERT-large", prelude_layers=2, looped_layers=4, coda_layers=2):
    values = ModernBertConfig.from_pretrained(checkpoint).to_dict()
    
    values.pop("model_type", None)
    return BortConfig(**(values | {"prelude_layers": prelude_layers, "looped_layers": looped_layers, "coda_layers": coda_layers, "layer_types": None}))

# Cell 3 

# Yifan Zhang, recurrent looped transformer
class GatedMerge(nn.Module):
    def __init__(self, width, alpha, base="state"):
        super().__init__()

        if base not in ("state", "anchor"):
            raise ValueError("merge_base must be state or anchor")
        self.norm = nn.RMSNorm(width)
        self.gate = nn.Linear(2 * width, width)
        self.proj = nn.Linear(width, width, bias=False)
        self.alpha = alpha
        self.base = base

    def forward(self, anchor, state):
        normalized = self.norm(state)
        gate = self.gate(torch.cat((anchor, normalized), dim=-1)).sigmoid()
        # "anchor" is RLT's u = e + alpha * g * W_s RMSNorm(s): the input is re-injected every iteration, the state is the gated feedback
        return (anchor if self.base == "anchor" else state) + self.alpha * gate * self.proj(normalized)
		
# Cell 4

# Universal Transformers
class StepEncoding(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.width = width  # freq is computed in forward: a non-persistent buffer is not in checkpoints and from_pretrained leaves it uninitialised

    def forward(self, step, state):
        phase = step * 10000 ** (-torch.arange(0, self.width, 2, device=state.device).float() / self.width)
        return torch.cat((phase.sin(), phase.cos()))[:state.shape[-1]].to(state.dtype)
		
# Cell 5

class BortModel(ModernBertModel):
    config_class = BortConfig

    def __init__(self, config):
        super().__init__(config)

        self.merge = GatedMerge(config.hidden_size, config.merge_alpha, getattr(config, "merge_base", "state")) if config.use_merge else None
        self.step = StepEncoding(config.hidden_size)
        self.halting_head = nn.Linear(config.hidden_size, 1).requires_grad_(False) # Trained @ a later date.

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, inputs_embeds=None, recurrence_depth=None, **kwargs):
        depth = self.config.recurrence_depth if recurrence_depth is None else recurrence_depth
        if type(depth) is not int or depth < 1:
            raise ValueError("Depth must be positive & integral")

        x = self.embeddings(input_ids=input_ids, inputs_embeds=inputs_embeds)
        if position_ids is None:
            position_ids = torch.arange(x.size(1), device=x.device)[None, :]

        args = dict(config=self.config, inputs_embeds=x, attention_mask=attention_mask)
        the2masks = {}
        if isinstance(attention_mask, dict):
            the2masks = attention_mask
        else:
            the2masks = {"full_attention": create_bidirectional_mask(**args), "sliding_attention": create_bidirectional_sliding_window_mask(**args)};

        rope = {kind: self.rotary_emb(x, position_ids, kind) for kind in set(self.config.layer_types)}

        # Helper that reduces LOC by abstracting away the high level ML concept of, "running through layers"
        def run(s, e, state):
            for i in range(s, e):
                layer = self.layers[i] 
                kind = layer.attention_type
                state = layer(state, attention_mask=the2masks[kind], position_embeddings=rope[kind], **kwargs)
            return state

        prelude, core = self.config.prelude_layers, self.config.prelude_layers + self.config.looped_layers
        x = run(0, prelude, x)
        
        a = x
        for i in range(depth):
            x = run(prelude, core, (x if self.merge is None else self.merge(a, x)) + self.step(i, x))

        x = run(core, self.config.num_hidden_layers, x)
        return BaseModelOutput(last_hidden_state = self.final_norm(x))
		
# Cell 6

class BortForMaskedLM(ModernBertForMaskedLM):
    config_class = BortConfig

    def __init__(self, config):
        ModernBertPreTrainedModel.__init__(self, config)

        self.model = BortModel(config)
        self.head = ModernBertPredictionHead(config)
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=config.decoder_bias)
        self.sparse_prediction = config.sparse_prediction 
        self.sparse_pred_ignore_index = config.sparse_pred_ignore_index

        self.post_init()

    def get_input_embeddings(self):
        return self.model.get_input_embeddings()
		
# Cell 7 

def StealEmbeddingsFromModernBert(noble_robinhood_model, bourgeoisie_model, freeze=True):
    target = noble_robinhood_model.model.get_input_embeddings().weight
    source = bourgeoisie_model.get_input_embeddings().weight

    if target.shape != source.shape:
        raise ValueError("incompatible.")

    with torch.no_grad():
        target.copy_(source)

    target.requires_grad_(not freeze)
    noble_robinhood_model.tie_weights()