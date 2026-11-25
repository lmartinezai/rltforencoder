import random
from types import SimpleNamespace

import pytest
import torch
from datasets import Dataset
from torch.nn import functional as F
from transformers import AutoTokenizer

from arch import BortForMaskedLM, MakeBortConfig
from data import make_collator, pack_dataset
from distill import (
    DistillArguments,
    DistillTrainer,
    PositionalCollator,
    SkipFirst,
    StreamPosition,
    build_student,
    build_teacher,
    combine,
    depth_for_step,
    distill_step,
    teacher_decoder_params,
)
from store import TargetStore, collate_store, extract

ARGS = DistillArguments(output_dir="runs/test")
WORDS = ["the", "of", "and", "quick", "fox", "river", "mountain", "energy", "paper", "model", "green", "seven"]
random.seed(0)
TEXT = " ".join(random.choice(WORDS) for _ in range(900))  # high-entropy text: exercises dense positions


# ------------------------------------------------------------------ CPU

def test_combine_weights_and_temperature_scaling():
    ce, kl = torch.tensor(3.0), torch.tensor(0.5)
    assert torch.allclose(combine(ce, kl, ARGS), 0.6 * ce + 0.4 * 4 * kl)
    a = DistillArguments(output_dir="runs/test", temperature=4.0, ce_weight=0.3, kd_weight=0.7)
    assert torch.allclose(combine(ce, kl, a), 0.3 * ce + 0.7 * 16 * kl)


def test_depth_schedule_balanced_shuffled_and_stateless():
    D = ARGS.max_depth
    assert D == 8 and ARGS.eval_depths == [1, 4, 8]  # recurrence depth is sampled from 1..8 (mean 4.5 -> 22 virtual layers)
    draws = [depth_for_step(s, D) for s in range(8 * D)]
    for i in range(0, 8 * D, D):
        assert sorted(draws[i:i + D]) == list(range(1, D + 1))
    assert draws[:D] != list(range(1, D + 1)) and draws[:D] != draws[D:2 * D]
    assert draws == [depth_for_step(s, D) for s in range(8 * D)]  # pure function of step (resume-safe)


def test_extraction_requires_empty_explicit_destination(tmp_path):
    from distill import run_extract

    with pytest.raises(ValueError, match="targets_dir"):
        run_extract(DistillArguments(output_dir="x"))
    (tmp_path / "train").mkdir()
    with pytest.raises(FileExistsError):
        run_extract(DistillArguments(output_dir="x", targets_dir=str(tmp_path)))


def test_stream_workers_rejected_before_loading():
    from distill import train

    with pytest.raises(ValueError, match="dataloader_num_workers"):
        train(DistillArguments(output_dir="x", dataloader_num_workers=1))


def test_train_seeds_before_student_initialization(monkeypatch):
    import distill

    monkeypatch.setattr(distill.AutoTokenizer, "from_pretrained", lambda *_: None)
    monkeypatch.setattr(distill, "teacher_decoder_params", lambda _: (torch.ones(2, 2), torch.ones(2)))
    monkeypatch.setattr(distill, "build_teacher", lambda _: None)
    monkeypatch.setattr(distill, "make_collator", lambda *_: None)
    monkeypatch.setattr(distill, "make_train_dataset", lambda *_: [])
    monkeypatch.setattr(distill, "make_eval_dataset", lambda *_: [])
    monkeypatch.setattr(distill, "build_student", lambda *_: torch.rand(3))
    monkeypatch.setattr(distill, "DistillTrainer", lambda **kw: SimpleNamespace(
        model=kw["model"], train=lambda **_: None))
    args = DistillArguments(output_dir="x", seed=123)
    a = distill.train(args).model
    torch.rand(100)
    assert torch.equal(a, distill.train(args).model)


def test_store_metadata_rejected_before_training(monkeypatch):
    import distill
    from store import decoder_fingerprint

    W, b = torch.ones(2, 2), torch.ones(2)
    meta = dict(teacher="teacher", decoder_sha256=decoder_fingerprint(W, b), sparse=1, T=2.0,
                seq_len=16, mask_prob=0.3, topk=1, topp=0.9)
    monkeypatch.setattr(distill.AutoTokenizer, "from_pretrained", lambda *_: None)
    monkeypatch.setattr(distill, "teacher_decoder_params", lambda _: (W, b))
    monkeypatch.setattr(distill, "TargetStore", lambda _: SimpleNamespace(meta=meta))
    args = DistillArguments(output_dir="x", targets_dir="store", teacher="teacher", temperature=3.0)
    with pytest.raises(ValueError, match="temperature"):
        distill.train(args)
    meta["decoder_sha256"] = "wrong"
    with pytest.raises(ValueError, match="decoder"):
        distill.train(args)


@pytest.mark.gpu
def test_eval_seed_preserves_cuda_rng(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    from transformers import Trainer

    monkeypatch.setattr(Trainer, "evaluate", lambda *a, **kw: {})
    trainer = object.__new__(DistillTrainer)
    trainer.args = SimpleNamespace(eval_depths=[1, 2], seed=42)
    trainer.data_collator = None  # a real Trainer always has one; plain collators have no `at`
    torch.cuda.manual_seed_all(123)
    before = torch.cuda.get_rng_state().clone()
    trainer.evaluate()
    assert torch.equal(before, torch.cuda.get_rng_state())


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained(ARGS.teacher)


def test_packing_and_masking_rules(tok):
    ds = pack_dataset(Dataset.from_dict({"text": [TEXT] * 20}), tok, 64)
    assert len(ds) > 0
    assert all(len(r["input_ids"]) == 64 and r["input_ids"][0] == tok.cls_token_id
               and r["input_ids"][-1] == tok.sep_token_id for r in ds)
    torch.manual_seed(0)
    rows = [ds[i % len(ds)] for i in range(200)]
    batch = make_collator(tok, ARGS)(rows)
    ids, labels, am = batch["input_ids"], batch["labels"], batch["attention_mask"]
    orig = torch.tensor([r["input_ids"] for r in rows])
    sel = labels != -100
    assert set(batch) == {"input_ids", "attention_mask", "labels"}
    assert am.all()
    assert not sel[:, [0, -1]].any()                      # [CLS]/[SEP] never selected
    assert torch.equal(labels[sel], orig[sel])            # labels = original ids at selected positions
    assert torch.equal(ids[~sel], orig[~sel])             # everything else untouched
    n = sel.sum()
    assert abs(n / (orig.numel() - 2 * len(rows)) - 0.3) < 0.02
    assert abs((ids[sel] == tok.mask_token_id).sum() / n - 0.8) < 0.03
    assert abs((ids[sel] == labels[sel]).sum() / n - 0.1) < 0.03


def test_padding_excluded_from_masking(tok):
    enc = [tok(t, truncation=True, max_length=64) for t in (TEXT, "short one")]
    batch = make_collator(tok, ARGS)(enc)
    pad = batch["attention_mask"] == 0
    assert pad.any() and not (batch["labels"][pad] != -100).any()


def tiny_bort():
    cfg = MakeBortConfig(ARGS.teacher)
    cfg.hidden_size, cfg.intermediate_size, cfg.num_attention_heads, cfg.vocab_size = 64, 128, 4, 200
    cfg.pad_token_id = cfg.bos_token_id = cfg.eos_token_id = cfg.cls_token_id = cfg.sep_token_id = 0
    cfg.sparse_prediction = True
    return BortForMaskedLM(cfg).eval()


def test_execution_counts_sparse_logits_and_ce_targets():
    m = tiny_bort()
    calls = []
    for layer in m.model.layers:
        layer.register_forward_hook(lambda *_: calls.append(1))
    ids = torch.randint(5, 200, (2, 16))
    labels = torch.full_like(ids, -100)
    labels[0, 3], labels[1, 5], labels[1, 9] = 7, 8, 9
    for R in (1, 2, 6, 16):
        calls.clear()
        out = m(input_ids=ids, attention_mask=torch.ones_like(ids), labels=labels, recurrence_depth=R)
        assert len(calls) == 4 + 4 * R
        assert out.logits.shape == (3, 200)  # [N_masked, V], never [B, L, V]
        assert torch.allclose(out.loss, F.cross_entropy(out.logits.float(), labels[labels != -100]), atol=1e-6)


def test_tied_decoder_and_depth_validation():
    m = tiny_bort()
    assert m.decoder.weight is m.model.embeddings.tok_embeddings.weight
    ids = torch.randint(5, 200, (1, 8))
    for bad in (0, -1, 2.0):
        with pytest.raises(ValueError):
            m.model(input_ids=ids, recurrence_depth=bad)


# ------------------------------------------------------------------ real teacher, CUDA

@pytest.fixture(scope="module")
def teacher_resources():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    teacher = build_teacher(ARGS)
    return teacher, teacher_decoder_params(ARGS)[1].cuda()


@pytest.fixture()
def models(teacher_resources):
    teacher, bias = teacher_resources
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(0)
        student = build_student(ARGS)
    student = student.cuda()
    yield teacher, student, bias
    del student
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def batch(tok):
    ds = pack_dataset(Dataset.from_dict({"text": [TEXT] * 10}), tok, 128)
    torch.manual_seed(0)
    return {k: v.cuda() for k, v in make_collator(tok, ARGS)([ds[0], ds[1]]).items()}


@pytest.fixture(scope="module")
def store(teacher_resources, tok, tmp_path_factory):
    """A real discrete-pipeline store: teacher -> targets on disk (K=8, P=0.97 so both paths occur)."""
    teacher, _ = teacher_resources
    root = tmp_path_factory.mktemp("store")
    W, b = (t.cuda() for t in teacher_decoder_params(ARGS))
    args = DistillArguments(output_dir="runs/test", topk=8, topp=0.97)
    ds = pack_dataset(Dataset.from_dict({"text": [TEXT] * 20}), tok, 96)
    meta = extract(teacher, W, b, ds, make_collator(tok, args, seed=0), root, args, 16, batch_size=8)
    return root, meta, args


@pytest.mark.gpu
def test_only_embeddings_come_from_teacher(models):
    teacher, student, bias = models
    emb = "embeddings.tok_embeddings.weight"
    assert torch.equal(student.model.get_parameter(emb).cpu(), teacher.model.get_parameter(emb).cpu().float())
    assert torch.equal(bias.cpu(), teacher.decoder.bias.cpu().float())  # safetensors path == live teacher
    assert student.decoder.weight is student.model.get_parameter(emb) and not student.decoder.weight.requires_grad
    assert student.decoder.bias.requires_grad and all(p.requires_grad for p in student.head.parameters())
    for name in ("head.dense.weight", "model.layers.1.mlp.Wo.weight", "model.layers.0.attn.Wqkv.weight"):
        s, t = student.get_parameter(name).cpu(), teacher.get_parameter(name).cpu().float()
        assert not torch.equal(s, t), name  # the student's own random init
    assert not any(p.requires_grad for p in teacher.parameters()) and not teacher.training
    assert student.sparse_prediction


@pytest.mark.gpu
@pytest.mark.parametrize("ckpt", [False, True])
@pytest.mark.parametrize("R", [1, 16])
@pytest.mark.parametrize("mode", ["full", "compact"])
def test_bf16_step(models, batch, mode, R, ckpt):
    teacher, student, bias = models
    student.train()
    (student.gradient_checkpointing_enable if ckpt else student.gradient_checkpointing_disable)()
    student.zero_grad(set_to_none=True)
    args = DistillArguments(output_dir="runs/test", kd_mode=mode)
    with torch.autocast("cuda", torch.bfloat16):
        loss, parts = distill_step(student, teacher, bias, batch, R, args)
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(parts["kl"])
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in student.parameters() if p.requires_grad)
    assert student.model.embeddings.tok_embeddings.weight.grad is None
    assert all(p.grad is None for p in teacher.parameters())


@pytest.mark.gpu
def test_truncated_kl_is_a_lower_bound_of_full_kl(models, batch):
    teacher, student, bias = models
    student.eval()
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        kl = {m: distill_step(student, teacher, bias, batch, 4, DistillArguments(output_dir="x", kd_mode=m))[1]["kl"]
              for m in ("full", "compact")}
    assert kl["compact"] <= kl["full"] * 1.001 and kl["compact"] > 0.5 * kl["full"]


@pytest.mark.gpu
def test_stored_targets_give_the_same_loss_as_live_teacher(models, store):
    """The discrete pipeline (teacher -> disk -> student, teacher unloaded) is equivalent to the joint one."""
    teacher, student, bias = models
    root, meta, args = store
    assert meta["sparse"] > 0 and meta["dense"] > 0, meta  # both the top-K and the hidden-state path were stored
    ts = TargetStore(root)
    batch = {k: v.cuda() for k, v in collate_store([ts[i] for i in range(8)]).items()}
    live_args = DistillArguments(output_dir="x", kd_mode="compact", topk=args.topk, topp=args.topp)
    student.eval()
    with torch.no_grad(), torch.autocast("cuda", torch.bfloat16):
        stored_loss, stored = distill_step(student, None, bias, batch, 3, live_args)  # no teacher at all
        live_batch = {k: batch[k] for k in ("input_ids", "attention_mask", "labels")}
        live_loss, live = distill_step(student, teacher, bias, live_batch, 3, live_args)
    assert torch.allclose(stored["ce"], live["ce"]) and torch.allclose(stored["kl"], live["kl"], rtol=5e-3, atol=1e-3)
    assert torch.allclose(stored_loss, live_loss, rtol=5e-3, atol=1e-3)


def trainer_for(models, tok, tmp_path, train_ds, eval_ds, collator, teacher=True, **overrides):
    mdl, student, bias = models
    args = DistillArguments(output_dir=str(tmp_path), max_steps=4, warmup_steps=1, per_device_train_batch_size=2,
                            per_device_eval_batch_size=4, gradient_accumulation_steps=2, logging_steps=1,
                            save_steps=2, eval_steps=4, eval_depths=[1, 2], disable_tqdm=True, **overrides)
    return DistillTrainer(model=student, args=args, teacher=mdl if teacher else None, teacher_bias=bias,
                          data_collator=collator, processing_class=tok, train_dataset=train_ds, eval_dataset=eval_ds)


@pytest.fixture(scope="module")
def packed(tok):
    return pack_dataset(Dataset.from_dict({"text": [TEXT] * 60}), tok, 96)


@pytest.mark.gpu
def test_trainer_divides_loss_by_accumulation(models, tok, tmp_path, batch, packed):
    trainer = trainer_for(models, tok, tmp_path, packed, packed.select(range(8)), PositionalCollator(tok, ARGS, 0))
    assert trainer.model_accepts_loss_kwargs is False
    student, norms = trainer.model, {}
    student.train()
    for ga in (1, 4):
        trainer.current_gradient_accumulation_steps = ga
        student.zero_grad(set_to_none=True)
        trainer.training_step(student, dict(batch))
        norms[ga] = torch.sqrt(sum(p.grad.float().pow(2).sum() for p in student.parameters() if p.grad is not None))
    assert abs(norms[4] / norms[1] - 0.25) < 0.02


@pytest.mark.gpu
def test_trainer_runs_logs_resumes_and_eval_keeps_rng(models, tok, tmp_path, packed):
    mk = lambda p: trainer_for(models, tok, p, packed, packed.select(range(8)), PositionalCollator(tok, ARGS, 0))
    trainer = mk(tmp_path)
    trainer.train()
    logs = {k for h in trainer.state.log_history for k in h}
    assert {"loss", "ce", "kl", "depth", "grad_norm", "eval_R1_loss", "eval_R2_loss"} <= logs
    assert trainer.state.global_step == 4 and (tmp_path / "checkpoint-2").exists()
    torch.manual_seed(123)
    before = torch.get_rng_state()
    trainer.evaluate()
    assert torch.equal(torch.get_rng_state(), before)  # evaluating must not disturb the training RNG
    trainer2 = mk(tmp_path / "r")
    trainer2.train(resume_from_checkpoint=str(tmp_path / "checkpoint-2"))
    assert trainer2.state.global_step == 4
    assert all(torch.isfinite(p).all() for p in trainer2.model.parameters())


@pytest.mark.gpu
def test_discrete_training_from_store_without_teacher(models, tok, tmp_path, store):
    root = store[0]
    ds = TargetStore(root)
    trainer = trainer_for(models, tok, tmp_path, ds, ds, collate_store, teacher=False,
                          kd_mode="compact", topk=store[2].topk, topp=store[2].topp)
    trainer.train()
    assert trainer.teacher is None and trainer.state.global_step == 4
    assert all(torch.isfinite(torch.tensor(h["loss"])) for h in trainer.state.log_history if "loss" in h)
    assert "eval_R2_loss" in {k for h in trainer.state.log_history for k in h}


@pytest.mark.gpu
def test_streaming_checkpoint_records_an_exact_position_and_resume_replays_to_it(models, tok, tmp_path, packed):
    mk = lambda: packed.to_iterable_dataset(num_shards=4).shuffle(seed=0, buffer_size=8)
    stream = mk()
    trainer = trainer_for(models, tok, tmp_path, stream, packed.select(range(8)), PositionalCollator(tok, ARGS, 0))
    trainer.add_callback(StreamPosition(stream, seed=0))
    trainer.train()
    # checkpoint-2 = 2 steps x accum 2 x batch 2 = 8 sequences consumed
    position = StreamPosition.load(tmp_path / "checkpoint-2")
    assert position["rows_consumed"] == 8 and position["seed"] == 0 and position["global_step"] == 2
    full = [tuple(r["input_ids"]) for r in mk()]
    resumed = [tuple(r["input_ids"]) for r in SkipFirst(mk(), position["rows_consumed"])]
    assert resumed == full[8:]  # exact: not "skips at most the shuffle buffer", the very next sequences of the very same stream


def test_stream_position_is_written_with_the_schedule_and_drift_is_fatal(tmp_path):
    args = _args(tmp_path, per_device_train_batch_size=3, gradient_accumulation_steps=4)
    (tmp_path / "checkpoint-7").mkdir()
    state = SimpleNamespace(global_step=7)

    class Remote:  # a remote stream counts its own consumption
        rows_consumed = 100 + 5 * 12

    cb = StreamPosition(Remote(), seed=9, start_rows=100, eval_sha256="abc", stream_sha256="stream")
    cb.on_train_begin(args, SimpleNamespace(global_step=2), None)  # this run began at step 2 (a resume)
    cb.on_save(args, state, None)  # 100 + (7 - 2) steps x 3 x 4 = 160 == the stream's own count
    assert StreamPosition.load(tmp_path / "checkpoint-7") == {"rows_consumed": 160, "seed": 9, "global_step": 7, "eval_sha256": "abc", "stream_sha256": "stream"}
    Remote.rows_consumed += 4  # the Trainer's dataloader looks one micro-batch ahead: tolerated, the checkpoint says 160
    cb.on_save(args, state, None)
    assert StreamPosition.load(tmp_path / "checkpoint-7")["rows_consumed"] == 160
    for bad in (159, 160 + 12 + 1):  # a sequence lost, or more than a whole accumulation window ahead
        Remote.rows_consumed = bad
        with pytest.raises(RuntimeError, match="drifted"):
            cb.on_save(args, state, None)


def test_a_checkpoint_without_a_stream_position_cannot_be_resumed_exactly(tmp_path):
    with pytest.raises(ValueError, match="stream_position.json"):
        StreamPosition.load(tmp_path)


def test_remote_resume_rejects_changed_training_identity(monkeypatch, tmp_path):
    import distill

    position = {"seed": 42, "rows_consumed": 4, "eval_sha256": "eval", "stream_sha256": "original"}
    closed = []
    class Remote:
        def __init__(self, *args, **kwargs):
            self.meta = {"stream_sha256": "different"}
            self.eval_sha256 = "eval"
        def close(self):
            closed.append(True)
    monkeypatch.setattr(distill.StreamPosition, "load", lambda _: position)
    monkeypatch.setattr(distill, "configure_cuda", lambda: None)
    monkeypatch.setattr(distill.AutoTokenizer, "from_pretrained", lambda *_: None)
    monkeypatch.setattr(distill, "teacher_decoder_params", lambda _: (torch.ones(2, 2), torch.ones(2)))
    monkeypatch.setattr(distill, "RemoteTargets", Remote)
    monkeypatch.setattr(distill, "check_server_meta", lambda *_: None)
    args = DistillArguments(output_dir=str(tmp_path), teacher_addr="localhost:1", resume_from_checkpoint="checkpoint", seed=42)
    with pytest.raises(ValueError, match="training stream identity"):
        distill.train(args)
    assert closed == [True]


# ---------------------------------------------------------------- argument / server validation, NaN guard

def _args(tmp_path, **kw):
    return DistillArguments(output_dir=str(tmp_path), use_cpu=True, bf16=False, report_to="none", **kw)


def test_validate_args_rejects_contradictory_teacher_sources(tmp_path):
    from distill import validate_args

    validate_args(_args(tmp_path))  # joint
    validate_args(_args(tmp_path, teacher_addr="h:1"))
    validate_args(_args(tmp_path, targets_dir="s"))
    validate_args(_args(tmp_path, stage="extract", targets_dir="s"))
    with pytest.raises(ValueError, match="choose one teacher source"):
        validate_args(_args(tmp_path, teacher_addr="h:1", targets_dir="s"))
    with pytest.raises(ValueError, match="--teacher_addr does not apply"):
        validate_args(_args(tmp_path, stage="extract", targets_dir="s", teacher_addr="h:1"))
    with pytest.raises(ValueError, match="targets_dir is required"):
        validate_args(_args(tmp_path, stage="extract"))


def test_validate_args_rejects_server_owned_flags_in_remote_mode(tmp_path):
    from distill import validate_args

    with pytest.raises(ValueError, match=r"--holdout_pct, --seq_len.*teacher server owns"):
        validate_args(_args(tmp_path, teacher_addr="h:1", holdout_pct=5, seq_len=256))
    validate_args(_args(tmp_path, holdout_pct=5, seq_len=256))  # they are the trainer's own flags without a server
    validate_args(_args(tmp_path, teacher_addr="h:1", temperature=1.0))  # z carries no temperature: the trainer's


def test_server_meta_must_match_the_trainers_decoder(tmp_path):
    from distill import check_server_meta
    from store import decoder_fingerprint

    W, b = torch.randn(10, 4), torch.randn(10)
    args = _args(tmp_path, teacher="t")
    good = {"teacher": "t", "hidden": 4, "decoder_sha256": decoder_fingerprint(W, b)}
    check_server_meta(good, args, W, b)
    for bad in ({"teacher": "other"}, {"hidden": 8}, {"decoder_sha256": decoder_fingerprint(W + 1e-3, b)}, {"decoder_sha256": None}):
        with pytest.raises(SystemExit, match="does not match"):
            check_server_meta(good | bad, args, W, b)
    with pytest.raises(SystemExit):  # a server that does not send a fingerprint (old protocol) is refused too
        check_server_meta({k: v for k, v in good.items() if k != "decoder_sha256"}, args, W, b)


def test_non_finite_loss_stops_training_loudly(tmp_path):
    from distill import StopOnNonFinite

    cb, state = StopOnNonFinite(), SimpleNamespace(global_step=7)
    cb.on_log(None, state, None, logs={"loss": 1.5, "grad_norm": 2.0, "learning_rate": 1e-4})
    cb.on_log(None, state, None, logs={"eval_loss": float("nan")})  # only the training signals are guarded
    for bad in ({"loss": float("nan")}, {"loss": 1.0, "grad_norm": float("inf")}):
        with pytest.raises(FloatingPointError, match="step 7"):
            cb.on_log(None, state, None, logs=bad)


def test_configure_cuda_applies_everywhere_and_is_a_noop_without_cuda(monkeypatch):
    import distill

    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "set_per_process_memory_fraction", lambda f: calls.append(f))
    distill.configure_cuda()
    assert calls == []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch, "set_float32_matmul_precision", lambda p: calls.append(p))
    distill.configure_cuda()
    assert calls == [distill.MEMORY_FRACTION, "high"]


# ------------------------------------------------------------------ resume equivalence and accumulation weighting (GPU)

def _grads(student):
    return torch.cat([p.grad.float().flatten() for p in student.parameters() if p.grad is not None])


@pytest.mark.gpu
def test_accumulation_is_the_mean_of_micro_batch_means(models, tok, tmp_path, packed):
    """The intended weighting: each micro-batch contributes its own mean loss, equally. (Not a masked-token-weighted mean;
    with fixed-length packing the masked counts differ by ~1%, so the two coincide in training. This pins the choice.)"""
    trainer = trainer_for(models, tok, tmp_path, packed, packed.select(range(8)), PositionalCollator(tok, ARGS, 0))
    student = trainer.model
    student.train()
    torch.manual_seed(0)
    b1 = {k: v.cuda() for k, v in make_collator(tok, ARGS)([packed[0], packed[1]]).items()}
    sparse_args = SimpleNamespace(mask_prob=0.1)  # very different masked count (not TrainingArguments: that resets accelerate state)
    b2 = {k: v.cuda() for k, v in make_collator(tok, sparse_args)([packed[2], packed[3]]).items()}
    n1, n2 = int((b1["labels"] != -100).sum()), int((b2["labels"] != -100).sum())
    assert n1 > 1.5 * n2  # unequal micro-batches: the two weightings differ

    single = []
    trainer.current_gradient_accumulation_steps = 1
    for b in (b1, b2):
        student.zero_grad(set_to_none=True)
        trainer.training_step(student, dict(b))
        single.append(_grads(student).clone())
    student.zero_grad(set_to_none=True)
    trainer.current_gradient_accumulation_steps = 2
    for b in (b1, b2):
        trainer.training_step(student, dict(b))
    accumulated = _grads(student)

    mean_of_means = (single[0] + single[1]) / 2
    token_weighted = (n1 * single[0] + n2 * single[1]) / (n1 + n2)
    err = lambda ref: ((accumulated - ref).norm() / ref.norm()).item()
    assert err(mean_of_means) < 1e-3 and err(token_weighted) > 10 * err(mean_of_means)


@pytest.mark.gpu
@pytest.mark.parametrize("compiled", [False, True], ids=["eager", "compiled"])
def test_resumed_training_equals_uninterrupted_training(models, tok, tmp_path, packed, compiled):
    """Interrupt at step 2 and resume: parameters (so optimizer moments, scheduler, RNG and data order) and the logged
    losses at steps 3-4 must equal the uninterrupted run's. Compiled layers must survive the checkpoint round trip too."""
    from transformers import set_seed

    teacher, _, bias = models

    def run(path, resume=None):
        set_seed(0)
        student = build_student(ARGS).cuda()
        if compiled:
            for layer in student.model.layers:
                layer.compile()
        trainer = trainer_for((teacher, student, bias), tok, path, packed, packed.select(range(8)), PositionalCollator(tok, ARGS, 0))
        trainer.train(resume_from_checkpoint=resume)
        losses = {h["step"]: h["loss"] for h in trainer.state.log_history if "loss" in h}
        params = {k: v.detach().clone() for k, v in student.state_dict().items()}
        del trainer, student
        torch.cuda.empty_cache()
        return params, losses

    full, full_losses = run(tmp_path / "full")
    resumed, resumed_losses = run(tmp_path / "resumed", resume=str(tmp_path / "full" / "checkpoint-2"))
    assert all(resumed_losses[s] == pytest.approx(full_losses[s], rel=1e-4) for s in (3, 4))
    worst = max((resumed[k].float() - full[k].float()).abs().max().item() for k in full)
    assert worst < 1e-5, f"resumed run diverged from the uninterrupted one by {worst}"


def test_train_and_eval_stores_must_agree_on_their_metadata(monkeypatch):
    import distill
    from store import decoder_fingerprint

    W, b = torch.ones(2, 2), torch.ones(2)
    meta = dict(teacher="teacher", decoder_sha256=decoder_fingerprint(W, b), sparse=0, T=2.0,
                seq_len=16, mask_prob=0.3, topk=1, topp=0.9)
    monkeypatch.setattr(distill.AutoTokenizer, "from_pretrained", lambda *_: None)
    monkeypatch.setattr(distill, "teacher_decoder_params", lambda _: (W, b))
    monkeypatch.setattr(distill, "TargetStore", lambda path: SimpleNamespace(meta=meta if "train" in str(path) else meta | {"topk": 2}))
    with pytest.raises(ValueError, match="differ for topk"):
        distill.train(DistillArguments(output_dir="x", targets_dir="store", teacher="teacher"))


# ---------------------------------------------------------------- warm-restart learning-rate schedule

def _lrs(scheduler, optimizer, steps):
    out = []
    for _ in range(steps):
        out.append(optimizer.param_groups[0]["lr"])
        optimizer.step()
        scheduler.step()
    return out


def test_restart_cosine_is_two_hf_cosine_schedules_joined_at_the_restart():
    from transformers import get_cosine_schedule_with_warmup

    from distill import restart_cosine

    def hf(total):
        opt = torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=1.0)
        return _lrs(get_cosine_schedule_with_warmup(opt, 100, total), opt, total)

    first, second = hf(1000), hf(6200)
    got = [restart_cosine(s, 100, 1000, 6200) for s in range(6200)]
    assert got[:1000] == pytest.approx(first[:1000]) and got[1000:] == pytest.approx(second[1000:])
    assert got[999] < 1e-5 and got[1000] == pytest.approx(0.5 * (1 + torch.cos(torch.tensor(torch.pi * 900 / 6100)).item()))  # the jump
    assert [restart_cosine(s, 100, 0, 6200) for s in range(6200)] == pytest.approx(second)  # restart 0 = the plain schedule


def test_trainer_builds_the_restart_schedule_and_it_survives_a_checkpoint(tmp_path):
    from distill import restart_cosine

    args = _args(tmp_path, max_steps=300, warmup_steps=10, lr_restart_step=100, learning_rate=0.5)
    new = lambda: torch.optim.SGD([torch.nn.Parameter(torch.zeros(1))], lr=args.learning_rate)
    trainer = SimpleNamespace(args=args, lr_scheduler=None, optimizer=new())
    scheduler = DistillTrainer.create_scheduler(trainer, 300)
    assert trainer._created_lr_scheduler and DistillTrainer.create_scheduler(trainer, 300) is scheduler
    want = [0.5 * restart_cosine(s, 10, 100, 300) for s in range(300)]
    assert _lrs(scheduler, trainer.optimizer, 120) == pytest.approx(want[:120])
    state = scheduler.state_dict()  # resume mid-run: a fresh scheduler picks up at step 120
    again = SimpleNamespace(args=args, lr_scheduler=None, optimizer=new())
    resumed = DistillTrainer.create_scheduler(again, 300)
    resumed.load_state_dict(state)
    again.optimizer.load_state_dict(trainer.optimizer.state_dict())  # a real resume restores the optimizer (and its lr) too
    assert _lrs(resumed, again.optimizer, 180) == pytest.approx(want[120:])


def test_restart_step_is_validated(tmp_path):
    from distill import validate_args

    validate_args(_args(tmp_path, max_steps=1000, lr_restart_step=500))
    for bad in (dict(lr_restart_step=50), dict(lr_restart_step=1000), dict(lr_restart_step=2000), dict(lr_scheduler_type="linear", lr_restart_step=500)):
        with pytest.raises(ValueError, match="lr_restart_step"):
            validate_args(_args(tmp_path, max_steps=1000, **bad))
