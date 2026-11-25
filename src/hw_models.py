"""The models of the hardware study, in one place: name -> (kind, path, family, looped). kind is hf | bort | flat (as src/crossmodel_eval.py).
Hardware cost (FLOPs, memory, latency) depends on the architecture and not on the trained values, so one checkpoint per configuration is measured
(seed 1); quantisation robustness (src/quant_eval.py) is the one measurement that needs trained weights."""
from pathlib import Path

MODELS = {
    "teacher": ("hf", "answerdotai/ModernBERT-large", "reference", False),
    "roberta-base": ("hf", "FacebookAI/roberta-base", "reference", False),
    "deberta-base": ("hf", "microsoft/deberta-base", "reference", False),
    "distilbert-cased": ("hf", "distilbert/distilbert-base-cased", "reference", False),
    "A8": ("bort", "runs/arm_a_merge/checkpoint-12208", "8 blocks", True),
    "B8": ("bort", "runs/arm_b_nomerge/checkpoint-12208", "8 blocks", True),
    "C8": ("flat", "runs/arm_c_noloop/checkpoint-12208", "8 blocks", False),
    "a_s1": ("bort", "runs/small/a_s1/checkpoint-3072", "4 blocks", True),
    "b_s1": ("bort", "runs/small/b_s1/checkpoint-3072", "4 blocks", True),
    "c_s1": ("flat", "runs/small_c/c_s1/checkpoint-3072", "4 blocks", False),
    "d_s1": ("flat", "runs/iso_flop/d_s1/checkpoint-3072", "10 blocks (iso-FLOP control)", False),
    "e_s1": ("bort", "runs/small/e_s1/checkpoint-3072", "4 blocks", True),
}


def available(root="."):
    """The models whose checkpoints exist."""
    return {n: m for n, m in MODELS.items() if m[0] == "hf" or (Path(root) / m[1]).exists()}
