"""Score against compute in FLOPs (not blocks): fixed-depth Bort (every depth), the flat 8-layer baseline, and what the halting heads picked.

    uv run python src/flops_score.py --models A=runs/arm_a_merge/checkpoint-12208,runs/halt_a_seq/halting_head.pt B=runs/arm_b_nomerge/checkpoint-12208,runs/halt_b_seq/halting_head.pt \
        --flat C=runs/arm_c_noloop/checkpoint-12208

FLOPs are counted by torch's FlopCounterMode on a real forward of one 512-token sequence (30% of positions labelled, as in the evaluation), so
full vs sliding attention, the masked-position-only decoder and everything else is what the model actually does. Reported per sequence:
`encoder` = embeddings through the final norm (what recurrence changes), `total` = plus the prediction head and decoder at the masked positions.
FLOPs of a Bort forward are linear in the loop count (FLOPs(k) = FLOPs(0) + k x per-iteration), so an adaptive policy is costed from its mean loops.
Score = the distillation loss of src/halting_eval.py (fixed depths and exits) on the same held-out text. Lower is better.
Also prints, for each sequence-level head, how its chosen depths are distributed and whether they beat a random mix of depths."""
import argparse
import json
import sys

import torch
from torch.utils.flop_counter import FlopCounterMode

import distill
from arch import BortForMaskedLM
from bench_heldout import heldout_batches, load_teacher
from compare_arch import FlatForMaskedLM
from halting_eval import DEPTHS, collect, first_true


def count_flops(model, depth=None, seq_len=512, mask_prob=0.3):
    """-> (encoder GFLOPs, total GFLOPs) of one forward of a seq_len sequence with mask_prob of the positions labelled."""
    ids = torch.randint(5, 50000, (1, seq_len), device="cuda")
    labels = torch.full_like(ids, -100)
    labels[0, : int(seq_len * mask_prob)] = ids[0, : int(seq_len * mask_prob)]
    kwargs = {} if depth is None else {"recurrence_depth": depth}
    with torch.no_grad(), FlopCounterMode(display=False) as total:
        model(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels, **kwargs)
    with torch.no_grad(), FlopCounterMode(display=False) as encoder:
        model.model(input_ids=ids, attention_mask=torch.ones_like(ids), **kwargs)
    return encoder.get_total_flops() / 1e9, total.get_total_flops() / 1e9


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", required=True, help="NAME=checkpoint[,sequence_head.pt]")
    p.add_argument("--flat", nargs="*", default=[], help="NAME=checkpoint")
    p.add_argument("--eval_json", default="runs/halting_eval.json", help="halting_eval.py's output (policy points and the flat loss)")
    p.add_argument("--samples", type=int, default=128)
    a = p.parse_args(argv)
    sys.stdout.reconfigure(line_buffering=True)
    distill.configure_cuda()
    args = distill.DistillArguments(output_dir="runs/_flops", report_to="none")
    batches = heldout_batches(a.samples)
    teacher = load_teacher()
    args.bias = teacher.decoder.bias.float()
    with open(a.eval_json) as f:
        saved = json.load(f)

    for spec in a.models:
        name, _, rest = spec.partition("=")
        path, _, head_path = rest.partition(",")
        student = BortForMaskedLM.from_pretrained(path).cuda().eval()
        flops = {d: count_flops(student, d) for d in range(1, DEPTHS + 1)}
        per_iter = flops[2][0] - flops[1][0]
        print(f"[{name}] per sequence of 512 tokens: encoder GFLOPs by depth " + " ".join(f"{flops[d][0]:.0f}" for d in range(1, 9))
              + f"; one loop iteration {per_iter:.1f}; head+decoder {flops[1][1] - flops[1][0]:.1f}; total at depth 1 {flops[1][1]:.0f}")
        seq_head = torch.load(head_path, map_location="cpu") if head_path else None
        for k, b in batches.items():
            data = collect(student, teacher, args.bias, b, args)
            fixed = data["loss"].mean(1).tolist()
            print(f"[{k}] {name} fixed depth: " + "  ".join(f"d{d}: {fixed[d - 1]:.3f} @ {flops[d][1]:.0f}G" for d in range(1, 9)))
            if seq_head is not None:
                rows, loss = data["rows"], data["loss"]
                n_seq = int(rows.max()) + 1
                seq_sum = torch.zeros(DEPTHS, n_seq).index_add_(1, rows, loss)
                logit = data["pooled"] @ seq_head["weight"].squeeze(0) + seq_head["bias"]
                for theta in (0.5, 0.8, 0.9):
                    exit_depth = first_true(logit.sigmoid() >= theta)
                    hist = torch.bincount(exit_depth, minlength=DEPTHS + 1)[1:].tolist()
                    chosen = seq_sum[exit_depth - 1, torch.arange(n_seq)].sum().item() / loss.shape[1]
                    # a random mix with the same depth histogram (what a head with no information would give), averaged over shuffles
                    g = torch.Generator().manual_seed(0)
                    mixes = [seq_sum[exit_depth[torch.randperm(n_seq, generator=g)] - 1, torch.arange(n_seq)].sum().item() / loss.shape[1] for _ in range(200)]
                    print(f"[{k}] {name} seq head p>={theta}: chosen-depth histogram (1..8) {hist}; loss {chosen:.4f} vs {sum(mixes) / len(mixes):.4f} for a random assignment of the same depths "
                          f"(std {torch.tensor(mixes).std():.4f})")
        del student
        torch.cuda.empty_cache()
    for spec in a.flat:
        name, _, path = spec.partition("=")
        flat = FlatForMaskedLM.from_pretrained(path).cuda().eval()
        enc, tot = count_flops(flat)
        for k in batches:
            print(f"[{k}] {name} (no loop): loss {saved[k][name]['flat_loss_8_blocks']:.3f} @ {tot:.0f}G total ({enc:.0f}G encoder)")


if __name__ == "__main__":
    main()
