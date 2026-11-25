"""Train Bort's halting head on a frozen, already distilled checkpoint.

The head is `BortModel.halting_head`, a Linear(hidden, 1) that reads a masked position's state after each loop iteration (the output of
block 5, before the coda) and says whether another iteration is worth it. Nothing else is trained: the backbone is frozen.

Target. For every masked token and every depth d = 1..8 the student's distillation loss L_d (0.6 CE + 0.4 T^2 KL at T=2: the training
objective, per token) is measured by running the real depth-d forward. Token t "can stop" at depth d when L_d(t) <= min_j L_j(t) + tol,
i.e. stopping there costs at most `halt_tolerance` more than the best depth for that token. The head is a logistic regression for that
label on the state after iteration d (the state after iteration d does not depend on the total depth: one hook per run captures it).
Inputs are standardised with statistics from the first batch; the standardisation is folded into the weights when the head is saved,
so `halting_head.pt` applies to the raw state.

    uv run python src/halting.py --student runs/arm_a_merge/checkpoint-12208 --output_dir runs/halt_a \
        --teacher_addr <teacher-host>:5555 --data_seed 4242

It reuses distill.py's whole pipeline (teacher stream or local teacher, masks, logging, Trainer); only the model and the loss differ.
Evaluate with src/halting_eval.py."""
import sys
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch.nn import functional as F
from transformers import HfArgumentParser

import distill
from arch import BortForMaskedLM
from distill import DistillArguments, DistillTrainer
from store import split_targets
from targets import decode, make_targets, target_kl, teacher_hidden


def loop_bounds(student):
    """(first, last) block index of the looped core (2, 5 for the default 2 + 4 + 2): the output of `last` is the state after an iteration."""
    config = student.config
    return config.prelude_layers, config.prelude_layers + config.looped_layers - 1


@dataclass
class HaltArguments(DistillArguments):
    student: str = ""  # the Bort checkpoint whose halting head is trained
    halt_level: str = "token"  # "token": the head reads each masked token's state | "sequence": it reads the mean state of a sequence's masked tokens
    halt_tolerance: float = 0.1  # token level: a token "can stop" at depth d when its loss there is within this of its best depth's loss
    halt_seq_tolerance: float = 0.02  # sequence level: the same for a sequence's mean loss over its masked tokens
    # a probe on a frozen model: a handful of steps, one parameter vector, nothing to resume
    output_dir: str = "runs/halt"
    max_steps: int = 300
    learning_rate: float = 3e-3
    weight_decay: float = 0.0
    warmup_steps: float = 20
    eval_steps: int = 50
    save_strategy: str = "no"


@contextmanager
def loop_states(student):
    """Capture the output of the last looped block on every call: after the run, `states[i]` is the full [B, L, H] state after iteration i+1."""
    states = []
    handle = student.model.layers[loop_bounds(student)[1]].register_forward_hook(lambda _m, _inp, out: states.append(out))
    try:
        yield states
    finally:
        handle.remove()


@contextmanager
def block_inputs(student):
    """Capture what the loop's blocks are called with (attention masks, rotary embeddings), by attention type: the coda needs the same."""
    seen, handles = {}, []

    def capture(module, _args, kwargs):
        seen.setdefault(module.attention_type, kwargs)

    first, last = loop_bounds(student)
    for layer in student.model.layers[first:last + 1]:
        handles.append(layer.register_forward_pre_hook(capture, with_kwargs=True))
    try:
        yield seen
    finally:
        for h in handles:
            h.remove()


def decode_state(student, state, kwargs, selected):
    """Coda + final norm + prediction head + decoder on a loop state [B, L, H] -> logits [N_selected, V]: what the model does after its loop."""
    x = state
    for layer in student.model.layers[loop_bounds(student)[1] + 1:]:
        x = layer(x, **kwargs[layer.attention_type])
    return student.decoder(student.head(student.model.final_norm(x)[selected]))


def teacher_distribution(stored, W, bias, args):
    """The teacher's tempered log-probabilities [N, V] and probabilities, decoded ONCE per batch (every depth is compared with the same
    teacher). Only for dense targets (the hidden state), which is what the teacher server and the local teacher produce."""
    if stored.ids.shape[0] != 0:
        raise ValueError("teacher_distribution needs dense (hidden-state) targets")
    logp = F.log_softmax(decode(stored.hidden, W, bias) / args.temperature, -1)
    return logp.exp(), logp


def token_losses(logits, labels, stored, W, bias, args, teacher=None):
    """Per-masked-token distillation loss (the training objective: ce_weight CE + kd_weight T^2 KL at T) -> [N].
    `teacher` = teacher_distribution(...) to skip re-decoding the teacher at every depth."""
    ce = F.cross_entropy(logits.float(), labels, reduction="none")
    if teacher is None:
        kl = target_kl(logits, stored, W, bias, args.temperature)
    else:
        p, logp = teacher
        kl = (p * (logp - F.log_softmax(logits.float() / args.temperature, -1))).sum(-1)
    return args.ce_weight * ce + args.kd_weight * args.temperature**2 * kl


def run_loop(student, input_ids, attention_mask, depth):
    """One pass through the loop to `depth` -> (the [B, L, H] state after every iteration, the blocks' keyword inputs for decode_state). The
    state after iteration d does not depend on the total depth, so this one pass serves every depth up to `depth`."""
    with loop_states(student) as states, block_inputs(student) as kwargs:
        student.model(input_ids=input_ids, attention_mask=attention_mask, recurrence_depth=depth)
    return states, kwargs


def logits_by_depth(student, input_ids, attention_mask, selected, depths):
    """{depth: logits [N_selected, V]} for every depth in `depths` from ONE pass through the loop (each iteration's state decoded through the coda)."""
    states, kwargs = run_loop(student, input_ids, attention_mask, max(depths))
    return {d: decode_state(student, states[d - 1], kwargs, selected) for d in depths}


def _table(per_depth, labels, stored, W, bias, args, signals, confidence):
    """(state [N, H], logits [N, V]) per depth -> the stacked per-depth tensors depth_table and depth_table_fast return."""
    teacher = teacher_distribution(stored, W, bias, args)
    states, losses, kls, changes, entropies, top = [], [], [], [], [], []
    prev_logp = prev_state = None
    for state, logits in per_depth:
        states.append(state)
        losses.append(token_losses(logits, labels, stored, W, bias, args, teacher))
        if signals or confidence:
            logp = F.log_softmax(logits.float(), -1)
        if confidence:
            entropies.append(-(logp.exp() * logp).sum(-1))
            top.append(logp.max(-1).values.exp())
        if signals:  # inf at the first depth: nothing to compare with
            kls.append((logp.exp() * (logp - prev_logp)).sum(-1) if prev_logp is not None else torch.full_like(losses[-1], torch.inf))
            changes.append(1 - F.cosine_similarity(state, prev_state, dim=-1) if prev_state is not None else torch.full_like(losses[-1], torch.inf))
            prev_logp, prev_state = logp, state
    out = (torch.stack(states), torch.stack(losses))
    if signals:
        out += (torch.stack(kls), torch.stack(changes))
    if confidence:
        out += (torch.stack(entropies), torch.stack(top))
    return out


@torch.no_grad()
def depth_table(student, batch, stored, bias, args, depths, signals=False, confidence=False):
    """Run the frozen student at every depth on one batch (a full forward per depth) -> (states [D, N, H], losses [D, N]) at the masked positions.
    `batch` is the model's inputs (input_ids, attention_mask, labels); `stored` the teacher's targets for the same positions.
    signals=True also returns the two parameter-free "has it converged" signals between consecutive depths, [D, N] each: kl = KL(p_d || p_{d-1})
    of the output distributions (Huginn), change = 1 - cos(s_d, s_{d-1}) of the loop states (CALM's state saturation).
    confidence=True also returns the entropy and top probability of each depth's output distribution (what a deployed head could read; needs the
    decoder at every depth), [D, N] each, after the signals."""
    selected = batch["labels"] != -100

    def per_depth():
        for d in depths:
            with loop_states(student) as seen:
                out = student(**batch, recurrence_depth=d)
            yield seen[-1][selected].float(), out.logits

    return _table(per_depth(), batch["labels"][selected], stored, student.decoder.weight, bias, args, signals, confidence)


@torch.no_grad()
def depth_table_fast(student, batch, stored, bias, args, depths, signals=False, confidence=False):
    """depth_table in ONE pass (run_loop): every iteration's state is kept and pushed through the coda. 4 + 4D + 2D blocks instead of
    sum(4 + 4d): 52 instead of 176 for D = 8 (3.4x less). Identical results to depth_table (tested)."""
    selected = batch["labels"] != -100
    states, kwargs = run_loop(student, batch["input_ids"], batch["attention_mask"], max(depths))
    per_depth = ((states[d - 1][selected].float(), decode_state(student, states[d - 1], kwargs, selected)) for d in depths)
    return _table(per_depth, batch["labels"][selected], stored, student.decoder.weight, bias, args, signals, confidence)


def first_true(condition, dim=0):
    """bool with a depth axis `dim` -> the 1-based first depth where it holds; the last depth if never (the loop ends there anyway)."""
    condition = condition.clone()
    condition.select(dim, -1).fill_(True)
    return condition.float().argmax(dim) + 1


def interpolate(fixed, loops):
    """Loss of the fixed-depth model at a (fractional) mean loop count, linear between neighbouring depths (= what picking one of the two depths
    at random per sequence gives). fixed: list of floats, fixed[0] is depth 1."""
    lo = min(max(int(loops), 1), len(fixed) - 1)
    return fixed[lo - 1] + (loops - lo) * (fixed[lo] - fixed[lo - 1])


def pool_by_sequence(values, rows, n_seq):
    """Mean of `values` [D, N, ...] over the masked tokens of each sequence (rows [N] = the sequence of each token) -> [D, n_seq, ...]."""
    counts = torch.bincount(rows, minlength=n_seq).clamp_min(1).to(values.dtype)
    summed = values.new_zeros(values.shape[0], n_seq, *values.shape[2:]).index_add_(1, rows, values)
    return summed / counts.reshape(1, n_seq, *([1] * (values.dim() - 2)))


def can_stop(losses, tolerance):
    """[D, N] bool: depth d is good enough for token t (within `tolerance` of its best depth). Every token has at least one."""
    return losses <= losses.min(0).values + tolerance


def halting_loss(head, states, losses, tolerance, mean, std):
    """BCE of the head on the standardised states against `can_stop`. -> (loss, parts)."""
    target = can_stop(losses, tolerance).float()
    logits = head((states - mean) / std).squeeze(-1)
    loss = F.binary_cross_entropy_with_logits(logits, target)
    return loss, {"bce": loss.detach(), "stop_rate": target.mean(), "acc": ((logits > 0) == (target > 0.5)).float().mean()}


def pooled_halting_loss(head, states, losses, rows, n_seq, tolerance, mean, std):
    """Sequence level: BCE of the head on each sequence's mean (standardised) state at each depth against "the sequence's mean loss at this
    depth is within `tolerance` of its best depth". -> (loss, parts)."""
    pooled, seq_loss = pool_by_sequence(states, rows, n_seq), pool_by_sequence(losses, rows, n_seq)
    return halting_loss(head, pooled, seq_loss, tolerance, mean, std)


def fold_standardisation(head, mean, std):
    """-> (weight [H], bias []) of the equivalent head on raw states: w.(x-mu)/s + b = (w/s).x + (b - w.mu/s)."""
    w = head.weight.detach().float().squeeze(0)
    return w / std, head.bias.detach().float().squeeze() - (w * mean / std).sum()


class HaltTrainer(DistillTrainer):
    """DistillTrainer whose loss is the halting head's BCE. The student is frozen except the head."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.mean = self.std = None  # standardisation of the head's input, taken from the first batch

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        batch, stored = split_targets(dict(inputs)) if "kd_ids" in inputs else (dict(inputs), None)
        if stored is None:  # local teacher, called outside autocast like the server does (its hidden state must stay bf16)
            with torch.autocast(batch["input_ids"].device.type, enabled=False):
                stored = make_targets(teacher_hidden(self.teacher, batch), model.decoder.weight, self.teacher_bias,
                                      self.args.temperature, 0, 1.0)
        states, losses = depth_table(model, batch, stored, self.teacher_bias, self.args, range(1, self.args.max_depth + 1))
        sequence = self.args.halt_level == "sequence"
        rows = (batch["labels"] != -100).nonzero()[:, 0]
        if sequence:  # the head's input is a sequence's mean state: standardise that
            features, n_seq = pool_by_sequence(states, rows, len(batch["labels"])), len(batch["labels"])
        else:
            features = states
        if self.mean is None:
            flat = features.reshape(-1, features.shape[-1])
            self.mean, self.std = flat.mean(0), flat.std(0).clamp_min(1e-6)
        head = model.model.halting_head
        with torch.autocast(states.device.type, enabled=False):  # a 1k-parameter probe: fp32
            if sequence:
                loss, parts = pooled_halting_loss(head, states, losses, rows, n_seq, self.args.halt_seq_tolerance, self.mean, self.std)
            else:
                loss, parts = halting_loss(head, states, losses, self.args.halt_tolerance, self.mean, self.std)
        if model.training:
            for k, v in parts.items():
                self._stats[k].append(v)
        return loss

    def evaluate(self, *args, **kwargs):  # one evaluation (this loss has no depth)
        return super(DistillTrainer, self).evaluate(*args, **kwargs)

    def save_head(self, path):
        """Save the trained head, standardisation folded in, so it applies to raw states."""
        weight, bias = fold_standardisation(self.model.model.halting_head, self.mean, self.std)
        sequence = self.args.halt_level == "sequence"
        torch.save({"weight": weight.cpu().unsqueeze(0), "bias": bias.cpu().reshape(1), "level": self.args.halt_level,
                    "tolerance": self.args.halt_seq_tolerance if sequence else self.args.halt_tolerance, "student": self.args.student}, path)


def load_halting_head(model, path):
    """Put a saved head (raw-state weights) into `model.model.halting_head`."""
    saved = torch.load(path, map_location="cpu")
    model.model.halting_head.load_state_dict({"weight": saved["weight"], "bias": saved["bias"]})
    return model


def build_halting_student(args, embedding_weight=None):
    """The checkpoint with everything frozen except a zero-initialised halting head (p = 0.5 until it learns)."""
    student = BortForMaskedLM.from_pretrained(args.student)
    student.requires_grad_(False)
    head = student.model.halting_head
    head.requires_grad_(True)
    torch.nn.init.zeros_(head.weight)
    torch.nn.init.zeros_(head.bias)
    return student


def main(argv=None):
    sys.stdout.reconfigure(line_buffering=True)
    (args,) = HfArgumentParser(HaltArguments).parse_args_into_dataclasses(argv)
    if args.stage != "train" or not args.student:
        raise SystemExit("halting.py trains a head: pass --student <checkpoint dir>")
    if args.halt_level not in ("token", "sequence"):
        raise SystemExit("--halt_level is token or sequence")

    print(f"[halting] frozen {args.student}; training the {args.halt_level}-level head, tolerance "
          f"{args.halt_seq_tolerance if args.halt_level == 'sequence' else args.halt_tolerance}")
    trainer = distill.train(args, trainer_class=HaltTrainer, build=build_halting_student)
    trainer.save_head(f"{args.output_dir}/halting_head.pt")
    print(f"[halting] wrote {args.output_dir}/halting_head.pt")
    return trainer


if __name__ == "__main__":
    main()
