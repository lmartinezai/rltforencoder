"""Halting head: targets, state capture, loss, folding of the standardisation, and what is trainable. Tiny random models, CPU."""
from types import SimpleNamespace

import pytest
import torch
from transformers.models.modernbert.configuration_modernbert import ModernBertConfig
from transformers.models.modernbert.modeling_modernbert import ModernBertForMaskedLM

import distill
import freeze
import halting
from arch import BortConfig, BortForMaskedLM
from targets import make_targets, teacher_hidden

V, H = 120, 32
ARGS = SimpleNamespace(temperature=2.0, ce_weight=0.6, kd_weight=0.4)


@pytest.fixture(scope="module")
def models():
    torch.manual_seed(0)
    config = BortConfig(hidden_size=H, intermediate_size=64, num_attention_heads=2, vocab_size=V, pad_token_id=0, bos_token_id=1,
                        eos_token_id=2, cls_token_id=1, sep_token_id=2, local_attention=8, global_rope_theta=1000.0,
                        local_rope_theta=1000.0)
    config.sparse_prediction = True
    student = BortForMaskedLM(config).eval()
    teacher = ModernBertForMaskedLM(ModernBertConfig(hidden_size=H, intermediate_size=64, num_attention_heads=2, num_hidden_layers=2,
                                                     vocab_size=V, pad_token_id=0, local_attention=8)).eval()
    return student, teacher


def batch_and_targets(student, teacher, n=3, length=24):
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(3, V, (n, length), generator=g)
    labels = torch.full_like(ids, -100)
    masked = torch.rand(n, length, generator=g) < 0.3
    masked[:, 0] = True
    labels[masked] = ids[masked]
    ids = ids.masked_fill(masked, 4)
    batch = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels}
    stored = make_targets(teacher_hidden(teacher, batch), student.decoder.weight, teacher.decoder.bias, 2.0, 0, 1.0)
    return batch, stored


def test_state_after_an_iteration_does_not_depend_on_the_total_depth(models):
    """The head reads one capture per run: that is only valid if iteration i's state is the same in a depth-i and a depth-8 run."""
    student, teacher = models
    batch, _ = batch_and_targets(student, teacher)
    with torch.no_grad():
        with halting.loop_states(student) as deep:
            student(**batch, recurrence_depth=8)
        assert len(deep) == 8
        for d in (1, 3, 8):
            with halting.loop_states(student) as shallow:
                student(**batch, recurrence_depth=d)
            assert len(shallow) == d and torch.equal(shallow[-1], deep[d - 1])


def test_depth_table_losses_are_the_training_objective(models):
    """Mean per-token loss at depth d == distill_step's loss for that depth (same student, teacher, batch)."""
    student, teacher = models
    batch, stored = batch_and_targets(student, teacher)
    args = distill.DistillArguments(output_dir="runs/test")
    states, losses = halting.depth_table(student, batch, stored, teacher.decoder.bias, args, range(1, 9))
    n = int((batch["labels"] != -100).sum())
    assert states.shape == (8, n, H) and losses.shape == (8, n)
    for d in (1, 4, 8):
        with torch.no_grad():
            want, _ = distill.distill_step(student, teacher, teacher.decoder.bias.float(), dict(batch), d, args)
        assert torch.allclose(losses[d - 1].mean(), want, atol=1e-4), d


def test_can_stop_is_within_tolerance_of_the_best_depth_and_never_empty():
    losses = torch.tensor([[3.0, 1.0, 5.0], [2.0, 1.05, 4.0], [2.5, 1.2, 4.2]])  # [D=3, N=3]: rows are depths, columns tokens
    ok = halting.can_stop(losses, 0.1)
    assert ok.tolist() == [[False, True, False], [True, True, True], [False, False, False]]
    assert ok.any(0).all()
    assert halting.can_stop(losses, 0.0).sum(0).tolist() == [1, 1, 1]  # exact ties aside, only the argmin


def test_folding_the_standardisation_gives_the_same_logits():
    torch.manual_seed(0)
    head = torch.nn.Linear(H, 1)
    states = torch.randn(5, 7, H) * 30 + 4
    mean, std = states.reshape(-1, H).mean(0), states.reshape(-1, H).std(0)
    w, b = halting.fold_standardisation(head, mean, std)
    assert torch.allclose(head((states - mean) / std).squeeze(-1), states @ w + b, atol=1e-4)


def test_only_a_zero_initialised_halting_head_is_trainable(models, tmp_path):
    student, _ = models
    student.save_pretrained(tmp_path)
    built = halting.build_halting_student(SimpleNamespace(student=str(tmp_path)))
    trainable = {n for n, p in built.named_parameters() if p.requires_grad}
    assert trainable == {"model.halting_head.weight", "model.halting_head.bias"}
    assert not built.model.halting_head.weight.any() and not built.model.halting_head.bias.any()


def test_head_learns_a_separable_target_and_saved_head_loads_into_the_model(models, tmp_path):
    student, teacher = models
    torch.manual_seed(0)
    batch, stored = batch_and_targets(student, teacher, n=8)
    args = distill.DistillArguments(output_dir="runs/test")
    states, losses = halting.depth_table(student, batch, stored, teacher.decoder.bias, args, range(1, 9))
    flat = states.reshape(-1, H)
    mean, std = flat.mean(0), flat.std(0).clamp_min(1e-6)
    head = torch.nn.Linear(H, 1)
    torch.nn.init.zeros_(head.weight), torch.nn.init.zeros_(head.bias)
    opt = torch.optim.Adam(head.parameters(), lr=0.05)
    first = None
    for _ in range(200):
        loss, _ = halting.halting_loss(head, states, losses, 0.1, mean, std)
        first = loss.item() if first is None else first
        opt.zero_grad(), loss.backward(), opt.step()
    assert abs(first - 0.6931) < 1e-3 and loss.item() < first - 0.01
    w, b = halting.fold_standardisation(head, mean, std)
    path = tmp_path / "halting_head.pt"
    torch.save({"weight": w.unsqueeze(0), "bias": b.reshape(1), "tolerance": 0.1, "student": "x"}, path)
    halting.load_halting_head(student, path)
    with torch.no_grad():
        assert torch.allclose(student.model.halting_head(states).squeeze(-1), head((states - mean) / std).squeeze(-1), atol=1e-3)


def test_exit_rule_and_interpolation():
    import halting_eval

    ok = torch.tensor([[False, True, False], [False, True, False], [True, True, False]])  # [D=3, B=3]
    assert halting_eval.first_true(ok).tolist() == [3, 1, 3]  # first depth that holds; the last if never
    assert halting.first_true(ok.T, dim=1).tolist() == [3, 1, 3] and not ok[-1, 2]  # the depth axis can be either one; the input is not modified
    fixed = [3.0, 2.0, 1.5]
    assert halting_eval.interpolate(fixed, 1.0) == 3.0 and halting_eval.interpolate(fixed, 2.5) == pytest.approx(1.75)
    assert halting_eval.interpolate(fixed, 3.0) == 1.5
    sums = torch.tensor([[3.0, 1.0], [2.0, 1.5], [1.5, 2.0]])  # [D, B] summed losses, 1 token per sequence
    r = halting_eval.point(sums, torch.tensor([3, 1]), 2, fixed)
    assert r["loops"] == 2.0 and r["blocks"] == 12.0 and r["loss"] == pytest.approx(1.25) and r["fixed_loss"] == pytest.approx(2.0)
    assert r["gain"] == pytest.approx(0.75)


# ------------------------------------------------------------------ GPU: the real Trainer pipeline, real teacher, random student

@pytest.mark.gpu
def test_halt_trainer_trains_only_the_head_and_saves_a_head_that_applies_to_raw_states(tmp_path):
    import random

    from datasets import Dataset
    from transformers import AutoTokenizer

    from data import pack_dataset
    from distill import PositionalCollator, build_student, build_teacher, teacher_decoder_params

    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    random.seed(0)
    text = " ".join(random.choice(["the", "of", "and", "quick", "fox", "river", "mountain", "energy", "paper", "model"]) for _ in range(900))
    base = distill.DistillArguments(output_dir="x")
    tok = AutoTokenizer.from_pretrained(base.teacher)
    packed = pack_dataset(Dataset.from_dict({"text": [text] * 60}), tok, 96)
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(0)
        build_student(base).save_pretrained(tmp_path / "student")
    args = halting.HaltArguments(student=str(tmp_path / "student"), output_dir=str(tmp_path / "out"), max_steps=6, warmup_steps=1,
                                 per_device_train_batch_size=2, per_device_eval_batch_size=4, gradient_accumulation_steps=2,
                                 logging_steps=1, eval_steps=6, disable_tqdm=True, learning_rate=0.05)
    student = halting.build_halting_student(args).cuda()
    before = {n: p.detach().clone() for n, p in student.named_parameters()}
    teacher = build_teacher(args)
    bias = teacher_decoder_params(args)[1].cuda()
    trainer = halting.HaltTrainer(model=student, args=args, teacher=teacher, teacher_bias=bias, data_collator=PositionalCollator(tok, args, 0),
                                  processing_class=tok, train_dataset=packed, eval_dataset=packed.select(range(8)))
    trainer.train()
    changed = {n for n, p in student.named_parameters() if not torch.equal(p.cpu(), before[n].cpu())}
    assert changed == {"model.halting_head.weight", "model.halting_head.bias"}
    logs = {k for h in trainer.state.log_history for k in h}
    assert {"bce", "stop_rate", "acc", "eval_loss"} <= logs and not any(k.startswith("eval_R") for k in logs)
    path = tmp_path / "halting_head.pt"
    trainer.save_head(path)
    # the saved (folded) head on raw states == the trained head on standardised states
    batch = {k: v.cuda() for k, v in PositionalCollator(tok, args, 7)([packed[0], packed[1]]).items()}
    stored = make_targets(teacher_hidden(teacher, batch), student.decoder.weight, bias, 2.0, 0, 1.0)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        states, _ = halting.depth_table(student, batch, stored, bias, args, range(1, 9))
    want = student.model.halting_head((states - trainer.mean) / trainer.std).squeeze(-1)
    halting.load_halting_head(student, path)
    assert torch.allclose(student.model.halting_head(states).squeeze(-1), want, atol=1e-3)


# ------------------------------------------------------------------ freeze-and-continue forward (freeze.py)

def run_frozen(student, batch, rule, mode):
    voters = batch["labels"] != -100
    with torch.no_grad(), freeze.FreezeLoop(student, voters, rule, mode) as halt:
        out = student(**batch, recurrence_depth=8)
    return out.logits, halt


def fixed(student, batch, depth):
    with torch.no_grad():
        return student(**batch, recurrence_depth=depth).logits


def test_a_rule_that_never_fires_is_the_plain_depth_8_forward(models):
    student, teacher = models
    batch, _ = batch_and_targets(student, teacher)
    logits, halt = run_frozen(student, batch, freeze.never, "token")
    assert torch.equal(logits, fixed(student, batch, 8)) and halt.loops.tolist() == [8, 8, 8]


def test_sequence_mode_equals_the_fixed_depth_forward_of_each_sequence(models):
    """A sequence stopped after iteration k is exactly the depth-k forward: this is what lets sequence-level exits be read off per-depth runs."""
    student, teacher = models
    batch, _ = batch_and_targets(student, teacher)
    rows = (batch["labels"] != -100).nonzero()[:, 0]
    stop_at = torch.tensor([3, 8, 5])

    def rule(i, previous, state):
        return stop_at == i

    logits, halt = run_frozen(student, batch, rule, "sequence")
    assert halt.loops.tolist() == [3, 8, 5]
    for b, k in enumerate(stop_at.tolist()):
        assert torch.allclose(logits[rows == b], fixed(student, batch, k)[rows == b], atol=1e-5), b


def test_token_mode_freezes_stopped_tokens_and_ends_when_all_voters_stopped(models):
    student, teacher = models
    batch, _ = batch_and_targets(student, teacher)
    voters = batch["labels"] != -100

    def stop_all_voters_at_2(i, previous, state):
        return torch.full(voters.shape, i == 2, device=state.device)

    logits, halt = run_frozen(student, batch, stop_all_voters_at_2, "token")
    assert halt.loops.tolist() == [2, 2, 2] and halt.done.all()
    assert torch.allclose(logits, fixed(student, batch, 2), atol=1e-5)  # everything frozen at the state after iteration 2

    first = voters.float().argmax(1)  # one voter per sequence stops at iteration 2, the rest never do
    only_first = torch.zeros_like(voters)
    only_first[torch.arange(len(first)), first] = True

    def stop_first_voter(i, previous, state):
        return only_first & (i == 2)

    logits, halt = run_frozen(student, batch, stop_first_voter, "token")
    assert (voters.sum(1) > 1).all()
    assert halt.loops.tolist() == [8, 8, 8] and not halt.done.any()  # a sequence is done only when ALL its voters have stopped
    assert torch.equal(halt.halted, only_first)  # exactly the one voter per sequence that was told to stop
    assert not torch.allclose(logits, fixed(student, batch, 8), atol=1e-5)  # the frozen voter changed what the others attend to


def test_free_signal_rules_on_a_stationary_state():
    state = torch.randn(2, 5, 8)
    voters = torch.ones(2, 5, dtype=torch.bool)
    assert not freeze.change_rule(0.1, "token", voters)(1, None, state).any()  # nothing to compare at iteration 1
    assert freeze.change_rule(0.1, "token", voters)(2, state, state).all()  # unchanged -> stop
    assert freeze.change_rule(0.1, "sequence", voters)(2, state, state).tolist() == [True, True]
    assert not freeze.change_rule(0.1, "token", voters)(2, state, -state).any()  # opposite direction


def test_one_pass_depth_table_equals_the_per_depth_runs(models):
    """depth_table_fast (one loop run, every state through the coda) == depth_table (a full forward per depth): states, losses and both signals."""
    student, teacher = models
    batch, stored = batch_and_targets(student, teacher)
    args = distill.DistillArguments(output_dir="runs/test")
    slow = halting.depth_table(student, batch, stored, teacher.decoder.bias, args, range(1, 9), signals=True)
    fast = halting.depth_table_fast(student, batch, stored, teacher.decoder.bias, args, range(1, 9), signals=True)
    for name, a, b in zip(("states", "losses", "kl", "change"), slow, fast, strict=True):
        finite = torch.isfinite(a)
        assert torch.equal(finite, torch.isfinite(b)), name
        assert torch.allclose(a[finite], b[finite], atol=1e-4, rtol=1e-4), name
    a, b = halting.depth_table(student, batch, stored, teacher.decoder.bias, args, [2, 5]), halting.depth_table_fast(student, batch, stored, teacher.decoder.bias, args, [2, 5])
    assert torch.allclose(a[1], b[1], atol=1e-4) and torch.allclose(a[0], b[0], atol=1e-4)  # a subset of depths works too


def test_pooling_and_the_three_free_signals_from_depth_table(models):
    student, teacher = models
    batch, stored = batch_and_targets(student, teacher)
    args = distill.DistillArguments(output_dir="runs/test")
    _states, losses, kl, change = halting.depth_table(student, batch, stored, teacher.decoder.bias, args, range(1, 9), signals=True)
    assert torch.isinf(kl[0]).all() and torch.isinf(change[0]).all() and torch.isfinite(kl[1:]).all() and (kl[1:] >= -1e-6).all()
    rows = (batch["labels"] != -100).nonzero()[:, 0]
    pooled = halting.pool_by_sequence(losses, rows, 3)
    assert torch.allclose(pooled[:, 1], losses[:, rows == 1].mean(1), atol=1e-6)
