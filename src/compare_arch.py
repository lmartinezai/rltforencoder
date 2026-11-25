"""Non-recurrent baseline for Bort: the same 8 ModernBERT blocks (same widths, same full/sliding layout, frozen tied teacher
embeddings, the same `--init`), each run once, with no loop, merge or step encoding. It is distilled by distill.py's own
pipeline, so it sees the same data, masks, teacher outputs, eval set and schedule as a Bort run with the same flags:

    uv run python src/compare_arch.py --output_dir runs/flat100m --teacher_addr <ip>:5555 --compile_layers \
        --lr_restart_step 1000 --max_steps 6200 --stream_buffer 2048

Everything else is distill.py's command line. There is no recurrence depth: depth is fixed at 1 and the only eval is
`eval_R1_*` (the name is kept so the numbers line up with Bort's `eval_R1/R4/R8`)."""
import sys

from transformers import HfArgumentParser
from transformers.models.modernbert.configuration_modernbert import ModernBertConfig
from transformers.models.modernbert.modeling_modernbert import ModernBertForMaskedLM

import distill
from arch import MakeBortConfig, StealEmbeddingsFromModernBert


class FlatForMaskedLM(ModernBertForMaskedLM):
    """ModernBertForMaskedLM that ignores `recurrence_depth`, so distill_step runs it unchanged."""

    def forward(self, *args, recurrence_depth=None, **kwargs):
        return super().forward(*args, **kwargs)


def flat_config(teacher="answerdotai/ModernBERT-large", config=None, layers=(2, 4, 2), layer_types="bort"):
    """Bort's config minus the recurrence settings: ModernBERT-large's widths with the same number of blocks (8: full attention at 0, 2, 5, 7)."""
    values = (config or MakeBortConfig(teacher, *layers)).to_dict()
    for key in ("model_type", "recurrence_depth", "merge_alpha", "use_merge", "merge_base", "prelude_layers", "looped_layers", "coda_layers"):
        values.pop(key, None)
    if layer_types == "modernbert":
        values.pop("layer_types", None)  # derived from global_attn_every_n_layers: full attention every 3rd block, as in the teacher
    config = ModernBertConfig(**values)
    config.sparse_prediction = True  # HF's own path: logits only at labelled positions, as in distill.build_student
    return config


def build_flat_student(args, embedding_weight=None, config=None):
    """Flat student with as many blocks as Bort (8 by default). The token embeddings come from the teacher (copied, frozen, tied); the rest is random, or
    with `--init teacher_layers` the nearest compatible teacher layers, exactly as distill.build_student does for Bort."""
    weight = distill.teacher_decoder_params(args)[0] if embedding_weight is None else embedding_weight
    student = FlatForMaskedLM(config or flat_config(args.teacher, layers=(args.prelude_layers, args.looped_layers, args.coda_layers), layer_types=args.flat_layer_types))
    StealEmbeddingsFromModernBert(student, distill._EmbeddingSource(weight), freeze=True)
    if args is not None and args.init == "teacher_layers":
        print(f"[init] student layers start from teacher layers {distill.init_from_teacher(student, args)}", flush=True)
    trainable = sum(p.numel() for p in student.parameters() if p.requires_grad)
    print(f"[compare_arch] flat {student.config.num_hidden_layers}-layer ModernBERT student, no recurrence: {trainable / 1e6:.1f}M trainable parameters", flush=True)
    return student


def main(argv=None):
    sys.stdout.reconfigure(line_buffering=True)
    (args,) = HfArgumentParser(distill.DistillArguments).parse_args_into_dataclasses(argv)
    if args.stage != "train":
        raise SystemExit("compare_arch.py only trains; extraction is architecture-independent (use distill.py --stage extract)")
    args.max_depth, args.eval_depths = 1, [1]  # no recurrence: one fixed depth, one eval

    return distill.train(args, build=build_flat_student)


if __name__ == "__main__":
    main()
