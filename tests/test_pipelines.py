"""The entry points end to end: `train()` (local stream, remote stream, stored targets), `run_extract`, the server's pieces,
the CLI and the benchmark script. GPU tests use the real ModernBERT-large teacher; everything is tiny (a few steps)."""
import json
import socket
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from datasets import Dataset
from transformers import AutoTokenizer

import distill
import stream
import teacher_server
from distill import DistillArguments, StreamPosition, teacher_decoder_params, train
from store import TargetStore, decoder_fingerprint
from teacher_server import ServeArguments, Teacher, accept_loop, eval_frames

ROOT = Path(__file__).resolve().parent.parent
ARGS = DistillArguments(output_dir="runs/test")
SEQ = 512


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained(ARGS.teacher)


def packed_rows(tok, n, seed):
    g = np.random.default_rng(seed)
    return [{"input_ids": [tok.cls_token_id, *g.integers(1000, 20000, SEQ - 2).tolist(), tok.sep_token_id]} for _ in range(n)]


# ---------------------------------------------------------------- server pieces (CPU)

def test_teacher_builds_its_fingerprints_from_the_model_and_the_decoder(monkeypatch):
    model = SimpleNamespace(model=torch.nn.Linear(4, 4), head=torch.nn.Identity(), state_dict=lambda: {"w": torch.ones(3)})
    W, b = torch.randn(7, 4), torch.zeros(7)
    monkeypatch.setattr(teacher_server, "build_teacher", lambda _: model)
    monkeypatch.setattr(teacher_server, "teacher_decoder_params", lambda _: (W, b))
    t = Teacher(ServeArguments(output_dir="x", compile_teacher=False, use_cpu=True))
    assert t.fingerprint == decoder_fingerprint(W, b) and len(t.model_fingerprint) == 64
    assert t.W.shape == (7, 4) and isinstance(t.model.model, torch.nn.Linear)  # not compiled when disabled


def test_the_cli_entry_point_refuses_extraction_without_a_destination():
    p = subprocess.run([sys.executable, str(ROOT / "src" / "distill.py"), "--output_dir", "x", "--stage", "extract"],
                       capture_output=True, text=True, timeout=300, check=False)
    assert p.returncode != 0 and "targets_dir" in p.stderr


# ---------------------------------------------------------------- GPU: train() end to end

def tiny_args(tmp_path, **kw):
    return DistillArguments(output_dir=str(tmp_path), max_steps=2, warmup_steps=1, per_device_train_batch_size=2,
                            per_device_eval_batch_size=4, gradient_accumulation_steps=2, logging_steps=1, save_steps=1,
                            eval_strategy="no", eval_depths=[1], disable_tqdm=True, report_to="none", seq_len=SEQ, **kw)


@pytest.mark.gpu
def test_run_extract_builds_a_store_train_consumes_and_a_finished_run_can_be_re_resumed(tok, tmp_path, monkeypatch):
    rows = packed_rows(tok, 24, 2)
    monkeypatch.setattr(distill, "make_train_dataset", lambda *_: iter(rows))
    monkeypatch.setattr(distill, "make_eval_dataset", lambda *_: rows[:8])
    store_dir = tmp_path / "targets"
    args = DistillArguments(output_dir=str(tmp_path / "x"), stage="extract", targets_dir=str(store_dir), extract_tokens=16 * SEQ,
                            eval_samples=8, extract_batch=8, seq_len=SEQ, topk=0)
    meta = distill.run_extract(args)
    assert meta["n_seq"] == 16 and len(TargetStore(store_dir / "train")) == 16 and len(TargetStore(store_dir / "eval")) == 8
    with pytest.raises(FileExistsError):  # a second extraction never overwrites
        distill.run_extract(args)
    monkeypatch.setattr(distill, "make_train_dataset", lambda *_: iter(rows))  # the iterator was consumed
    args.resume_from_checkpoint = "1"  # resuming a finished extraction is a no-op that returns the same store
    assert distill.run_extract(args)["n_seq"] == 16
    trained = train(tiny_args(tmp_path / "run", targets_dir=str(store_dir), kd_mode="compact"))
    assert trained.state.global_step == 2 and trained.teacher is None


@pytest.mark.gpu
def test_local_streaming_run_resumes_exactly_through_train(tok, tmp_path, monkeypatch):
    rows = packed_rows(tok, 40, 3)
    mk = lambda *_: Dataset.from_list(rows).to_iterable_dataset(num_shards=2).shuffle(seed=0, buffer_size=8)
    monkeypatch.setattr(distill, "make_train_dataset", mk)
    monkeypatch.setattr(distill, "make_eval_dataset", lambda *_: Dataset.from_list(rows[:8]))
    full = train(tiny_args(tmp_path / "full"))
    assert full.state.global_step == 2
    position = StreamPosition.load(tmp_path / "full" / "checkpoint-1")
    assert position["rows_consumed"] == 4 and position["seed"] == 42
    resumed = train(tiny_args(tmp_path / "resumed", resume_from_checkpoint=str(tmp_path / "full" / "checkpoint-1")))
    final = lambda t: next(h["loss"] for h in reversed(t.state.log_history) if "loss" in h)
    assert resumed.state.global_step == 2 and final(resumed) == pytest.approx(final(full), rel=1e-4)
    with pytest.raises(ValueError, match="data seed"):
        train(tiny_args(tmp_path / "other", seed=5, resume_from_checkpoint=str(tmp_path / "full" / "checkpoint-1")))


@pytest.mark.gpu
def test_remote_run_through_train_resumes_exactly_and_refuses_a_changed_stream(tok, tmp_path, monkeypatch):
    """train(--teacher_addr) against a real in-process server whose encoder is a stand-in (real decoder), then resume."""
    weight, bias = teacher_decoder_params(ARGS)
    torch.manual_seed(1)
    emb, head = torch.nn.Embedding(weight.shape[0], weight.shape[1]), torch.nn.Identity()
    model = SimpleNamespace(model=lambda input_ids, attention_mask: SimpleNamespace(last_hidden_state=emb(input_ids) * 0.1), head=head)
    t = object.__new__(Teacher)
    t.args, t.device, t.model = SimpleNamespace(temperature=ARGS.temperature), "cpu", model
    t.W, t.b, t.fingerprint, t.stream_identity = weight, bias, decoder_fingerprint(weight, bias), "b" * 64
    deliveries = []
    class RecordingRemote(stream.RemoteTargets):
        def __init__(self, *args, **kwargs):
            kwargs["stall_timeout"] = min(kwargs.get("stall_timeout", 30), 30)
            kwargs["reconnect_s"] = min(kwargs.get("reconnect_s", 30), 30)
            kwargs.setdefault("connect_timeout", 15)
            super().__init__(*args, **kwargs)
            self.delivered = []
            deliveries.append(self.delivered)
        def __iter__(self):
            for row in super().__iter__():
                self.delivered.append((row["input_ids"].tobytes(), row["labels"].tobytes()))
                yield row
    monkeypatch.setattr(distill, "RemoteTargets", RecordingRemote)
    rows = packed_rows(tok, 80, 4)
    monkeypatch.setattr(teacher_server, "make_train_dataset", lambda *_: rows)
    monkeypatch.setattr(teacher_server, "make_eval_dataset", lambda *_: rows[:8])
    sargs = ServeArguments(output_dir="x", extract_batch=8, seq_len=SEQ, prefetch=2, handshake_s=60, join_grace_s=0, feed_stop_s=5, log_every=10_000)
    srv = socket.create_server(("127.0.0.1", 0))
    threading.Thread(target=accept_loop, args=(srv, t, tok, sargs, eval_frames(t, tok, sargs), lambda _: None), daemon=True).start()
    addr = f"127.0.0.1:{srv.getsockname()[1]}"
    try:
        full = train(tiny_args(tmp_path / "full", teacher_addr=addr))
        assert full.state.global_step == 2 and any("stream_fill" in h for h in full.state.log_history)
        checkpoint = str(tmp_path / "full" / "checkpoint-1")
        assert StreamPosition.load(checkpoint)["rows_consumed"] == 4
        full_loss = next(h["loss"] for h in reversed(full.state.log_history) if "loss" in h)
        full_params = {k: v.detach().cpu().clone() for k, v in full.model.state_dict().items()}
        full_optimizer = torch.load(tmp_path / "full" / "checkpoint-2" / "optimizer.pt", map_location="cpu", weights_only=True)
        full_scheduler = torch.load(tmp_path / "full" / "checkpoint-2" / "scheduler.pt", map_location="cpu", weights_only=True)
        del full
        torch.cuda.empty_cache()
        resumed = train(tiny_args(tmp_path / "resumed", teacher_addr=addr, resume_from_checkpoint=checkpoint))
        resumed_loss = next(h["loss"] for h in reversed(resumed.state.log_history) if "loss" in h)
        assert resumed_loss == pytest.approx(full_loss, rel=1e-3)
        assert len(deliveries[0]) >= 8 and len(deliveries[1]) >= 4
        assert deliveries[1][:4] == deliveries[0][4:8]
        for name, value in resumed.model.state_dict().items():
            torch.testing.assert_close(value.cpu(), full_params[name], rtol=1e-4, atol=1e-5, msg=lambda m, name=name: f"{name}: {m}")
        resumed_optimizer = torch.load(tmp_path / "resumed" / "checkpoint-2" / "optimizer.pt", map_location="cpu", weights_only=True)
        resumed_scheduler = torch.load(tmp_path / "resumed" / "checkpoint-2" / "scheduler.pt", map_location="cpu", weights_only=True)
        torch.testing.assert_close(resumed_optimizer, full_optimizer, rtol=1e-4, atol=1e-5)
        assert resumed_scheduler == full_scheduler
        del resumed, full_params, full_optimizer, resumed_optimizer
        torch.cuda.empty_cache()
        with pytest.raises(ValueError, match="data seed"):
            train(tiny_args(tmp_path / "s", teacher_addr=addr, data_seed=9, resume_from_checkpoint=checkpoint))
        position = tmp_path / "full" / "checkpoint-1" / StreamPosition.FILE
        saved = json.loads(position.read_text())
        position.write_text(json.dumps(saved | {"stream_sha256": "c" * 64}))
        with pytest.raises(ValueError, match="stream identity"):
            train(tiny_args(tmp_path / "i", teacher_addr=addr, resume_from_checkpoint=checkpoint))
        position.write_text(json.dumps(saved | {"eval_sha256": "d" * 64}))
        with pytest.raises(ValueError, match="eval set differs"):
            train(tiny_args(tmp_path / "e", teacher_addr=addr, resume_from_checkpoint=checkpoint))
    finally:
        srv.close()


@pytest.mark.gpu
def test_teacher_benchmark_script_runs_end_to_end(tmp_path, monkeypatch):
    import bench_teacher

    out = tmp_path / "bench.json"
    monkeypatch.setattr(sys, "argv", ["bench_teacher", "--batches", "4", "--impls", "sdpa", "--compile", "0", "--seq-len", "64",
                                      "--warmup", "1", "--iters", "2", "--out", str(out)])
    bench_teacher.main()
    report = json.loads(out.read_text())
    row = report["rows"][0]
    assert report["ceiling"] > 0 and row["tok_per_s"] > 0 and 0 <= row["tensor_core_kernel_share"] <= 1 and row["rel_l2_vs_ref"] < 1


@pytest.mark.gpu
def test_train_with_compiled_layers_runs_and_stays_finite(tok, tmp_path, monkeypatch):
    rows = packed_rows(tok, 40, 6)
    monkeypatch.setattr(distill, "make_train_dataset", lambda *_: Dataset.from_list(rows).to_iterable_dataset(num_shards=2))
    monkeypatch.setattr(distill, "make_eval_dataset", lambda *_: Dataset.from_list(rows[:8]))
    trainer = train(tiny_args(tmp_path, compile_layers=True))
    losses = [h["loss"] for h in trainer.state.log_history if "loss" in h]
    assert trainer.state.global_step == 2 and losses and all(np.isfinite(losses))
