"""The whole remote path on CPU with a stand-in teacher: real `serve` + `Feed` + `RemoteTargets` + real collator + a student step.

Everything between the dataset and the loss is the production code; only the teacher network is a tiny deterministic module.
"""
import socket
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from datasets import Dataset
from transformers import AutoTokenizer

import stream
import teacher_server
from arch import BortForMaskedLM, MakeBortConfig
from data import make_collator
from distill import DistillArguments, distill_step
from store import collate_store, decoder_fingerprint
from teacher_server import ServeArguments, Teacher, accept_loop, eval_frames, mask_seed

ARGS = DistillArguments(output_dir="runs/test")
H, SEQ, BATCH, SEED = 64, 16, 4, 7
pytestmark = pytest.mark.timeout(90)


@pytest.fixture(autouse=True)
def bounded_network(monkeypatch):
    original = stream.RemoteTargets
    class BoundedRemote(original):
        def __init__(self, *args, **kwargs):
            kwargs.setdefault("connect_timeout", 10)
            kwargs.setdefault("stall_timeout", 15)
            kwargs.setdefault("reconnect_s", 15)
            super().__init__(*args, **kwargs)
    monkeypatch.setattr(stream, "RemoteTargets", BoundedRemote)


@pytest.fixture(scope="module")
def tok():
    return AutoTokenizer.from_pretrained(ARGS.teacher)


@pytest.fixture(scope="module")
def student(tok):
    torch.manual_seed(0)
    cfg = MakeBortConfig(ARGS.teacher)
    cfg.hidden_size, cfg.intermediate_size, cfg.num_attention_heads, cfg.vocab_size = H, 128, 4, len(tok)
    cfg.sparse_prediction = True
    return BortForMaskedLM(cfg).eval()


def corpus(tok, n, seed):
    g = np.random.default_rng(seed)
    return Dataset.from_list([{"input_ids": [tok.cls_token_id, *g.integers(1000, 3000, SEQ - 2).tolist(), tok.sep_token_id]}
                              for _ in range(n)])


@pytest.fixture()
def teacher(student):
    """A deterministic stand-in encoder + head whose decoder is the student's frozen tied one (as in training)."""
    torch.manual_seed(1)
    emb, head = torch.nn.Embedding(student.decoder.weight.shape[0], H), torch.nn.Linear(H, H)
    model = SimpleNamespace(model=lambda input_ids, attention_mask: SimpleNamespace(last_hidden_state=emb(input_ids)), head=head)
    t = object.__new__(Teacher)
    t.args, t.device, t.model = SimpleNamespace(temperature=ARGS.temperature), "cpu", model
    t.W, t.b = student.decoder.weight.detach().float(), torch.zeros(student.decoder.weight.shape[0])
    t.fingerprint, t.stream_identity = decoder_fingerprint(t.W, t.b), "a" * 64
    return t


def start_server(tok, teacher, args, monkeypatch, train, evaluation, log=lambda _: None):
    """Serve `teacher` over a real socket (the production `serve` + `Feed`); returns (address, listening socket)."""
    monkeypatch.setattr(teacher_server, "make_train_dataset", lambda *_: train)
    monkeypatch.setattr(teacher_server, "make_eval_dataset", lambda *_: evaluation)
    evals = eval_frames(teacher, tok, args)
    srv = socket.create_server(("127.0.0.1", 0))
    threading.Thread(target=accept_loop, args=(srv, teacher, tok, args, evals, log), daemon=True).start()
    return f"127.0.0.1:{srv.getsockname()[1]}", srv


@pytest.fixture()
def server(tok, teacher, monkeypatch):
    train, evaluation = corpus(tok, 80, 11), corpus(tok, 8, 12)
    args = ServeArguments(output_dir="x", extract_batch=BATCH, seq_len=SEQ, prefetch=2, handshake_s=10, join_grace_s=0, feed_stop_s=2,
                          log_every=10_000)
    addr, srv = start_server(tok, teacher, args, monkeypatch, train, evaluation)
    yield addr, args, train
    srv.close()


def test_the_server_survives_trainers_that_leave_while_its_feed_is_blocked(tok, teacher, monkeypatch):
    """Several abandoned feeds in a row (each stuck inside the dataset) must not stop it serving the next trainer."""
    release, calls, logs = threading.Event(), [], []
    train = corpus(tok, 80, 11)
    args = ServeArguments(output_dir="x", extract_batch=BATCH, seq_len=SEQ, prefetch=1, handshake_s=10, join_grace_s=0, feed_stop_s=0.2,
                          log_every=10_000)
    addr, srv = start_server(tok, teacher, args, monkeypatch, train, corpus(tok, 8, 12), log=logs.append)

    def hung_then_fine(*_):
        calls.append(1)
        if len(calls) <= 3:
            release.wait(30)  # a hung dataset call
        return train

    monkeypatch.setattr(teacher_server, "make_train_dataset", hung_then_fine)
    try:
        for _ in range(3):  # three trainers connect and leave while their feed is stuck (they may share one stream: same seed, same position)
            stream.RemoteTargets(addr, seed=SEED, connect_timeout=10).close()
        for _ in range(100):
            if any("abandoning" in m for m in logs):
                break
            threading.Event().wait(0.1)
        assert any("abandoning" in m for m in logs) and len(calls) <= 3  # the stuck stream was dropped, not kept for the next trainer
        ds = stream.RemoteTargets(addr, capacity=16, seed=SEED)  # the fourth is served normally
        try:
            assert len(take(ds, 4)) == 4
        finally:
            ds.close()
    finally:
        release.set()
        srv.close()


def reference_rows(tok, teacher, args, train, n_batches, seed=SEED):
    """What the stream must deliver, computed directly: batch i = sequences [i*B, (i+1)*B) masked with mask_seed(seed, i)."""
    out = []
    for i in range(n_batches):
        rows = [train[j] for j in range(i * BATCH, (i + 1) * BATCH)]
        x = make_collator(tok, args, mask_seed(seed, i))(rows)
        arrays = teacher(x)
        masked = (x["labels"] != -100).sum(1).tolist()
        hidden = np.split(arrays["hidden"], np.cumsum(masked)[:-1])
        out += [(x["input_ids"][r].tolist(), x["labels"][r].tolist(), hidden[r]) for r in range(BATCH)]
    return out


def same(row, ref):
    return row["input_ids"].tolist() == ref[0] and row["labels"].tolist() == ref[1] and np.array_equal(row["kd_hidden"], ref[2])


def take(ds, n):
    it = iter(ds)
    return [next(it) for _ in range(n)]


def test_remote_stream_equals_the_direct_computation_across_a_disconnect(tok, teacher, server):
    addr, args, train = server
    ref = reference_rows(tok, teacher, args, train, 8)
    ds = stream.RemoteTargets(addr, capacity=16, seed=SEED, reconnect_s=30)
    try:
        it = iter(ds)
        got = [next(it) for _ in range(6)]
        ds.sock.shutdown(socket.SHUT_RDWR)  # the link drops mid-stream
        got += [next(it) for _ in range(18)]
        assert ds.reconnects >= 1
    finally:
        ds.close()
    assert len(got) == 24 and all(same(r, f) for r, f in zip(got, ref[:24], strict=True))  # nothing lost, repeated or reordered


def test_remote_stream_resumes_at_an_arbitrary_sequence(tok, teacher, server):
    addr, args, train = server
    ref = reference_rows(tok, teacher, args, train, 8)
    ds = stream.RemoteTargets(addr, capacity=16, seed=SEED, start_seq=10)  # inside batch 2
    try:
        got = take(ds, 12)
    finally:
        ds.close()
    assert all(same(r, f) for r, f in zip(got, ref[10:22], strict=True))


def test_a_different_seed_is_a_different_stream(tok, teacher, server):
    addr, args, train = server
    ds = stream.RemoteTargets(addr, capacity=16, seed=SEED + 1)
    try:
        got = take(ds, 4)
    finally:
        ds.close()
    ref = reference_rows(tok, teacher, args, train, 1)
    assert any(r["labels"].tolist() != f[1] for r, f in zip(got, ref, strict=True))  # masks (and so targets) differ


def test_remote_batch_trains_a_student_like_a_live_teacher(tok, teacher, student, server):
    """collate_store over network rows -> distill_step gives the loss a live teacher gives (fp16 rounding aside)."""
    addr, args, train = server
    ds = stream.RemoteTargets(addr, capacity=16, seed=SEED)
    try:
        batch = collate_store(take(ds, BATCH))
    finally:
        ds.close()
    x = make_collator(tok, args, mask_seed(SEED, 0))([train[j] for j in range(BATCH)])
    assert torch.equal(batch["input_ids"], x["input_ids"]) and torch.equal(batch["labels"], x["labels"])
    bias = torch.zeros(student.decoder.weight.shape[0])
    remote, parts = distill_step(student, None, bias, dict(batch), 2, ARGS)
    live, live_parts = distill_step(student, teacher.model, bias, dict(x), 2, ARGS)
    assert torch.isfinite(remote) and torch.allclose(remote, live, rtol=2e-2, atol=2e-2), (remote, live)
    assert torch.allclose(parts["ce"], live_parts["ce"])  # CE does not involve the teacher at all


@pytest.mark.gpu
@pytest.mark.timeout(300)
def test_real_bf16_teacher_over_the_wire_trains_like_a_live_teacher(tok, monkeypatch):
    """The production path on CUDA: the real bf16 teacher's hidden states cross the network as bf16 bit patterns and give the
    student the loss a live teacher gives."""
    from distill import build_student, build_teacher, teacher_decoder_params

    if not torch.cuda.is_available():
        pytest.skip("no CUDA")
    model = build_teacher(ARGS)
    weight, bias = teacher_decoder_params(ARGS)
    t = object.__new__(Teacher)
    t.args, t.device, t.model = SimpleNamespace(temperature=ARGS.temperature), "cuda", model
    t.W, t.b = weight.cuda(), bias.cuda()
    t.fingerprint, t.stream_identity = decoder_fingerprint(weight, bias), "a" * 64
    g = np.random.default_rng(5)
    rows = lambda n: Dataset.from_list([{"input_ids": [tok.cls_token_id, *g.integers(1000, 9000, 94).tolist(), tok.sep_token_id]}
                                        for _ in range(n)])
    train, evaluation = rows(16), rows(4)
    args = ServeArguments(output_dir="x", extract_batch=4, seq_len=96, prefetch=2, handshake_s=30, join_grace_s=0, feed_stop_s=2, log_every=10_000)
    addr, srv = start_server(tok, t, args, monkeypatch, train, evaluation)
    ds = None
    try:
        ds = stream.RemoteTargets(addr, capacity=16, seed=SEED)
        batch = collate_store(take(ds, 4))
    finally:
        if ds is not None:
            ds.close()
        srv.close()
    student = build_student(ARGS, weight).cuda().train()
    x = make_collator(tok, args, mask_seed(SEED, 0))([train[j] for j in range(4)])
    assert torch.equal(batch["input_ids"], x["input_ids"]) and torch.equal(batch["labels"], x["labels"])
    remote_batch = {k: v.cuda() for k, v in batch.items()}
    live_batch = {k: v.cuda() for k, v in x.items()}
    param = student.head.dense.weight
    before = param.detach().clone()
    optimizer = torch.optim.SGD([param], lr=0.01)
    torch.manual_seed(123)
    remote, _ = distill_step(student, None, bias.cuda(), remote_batch, 3, ARGS)
    remote.backward()
    remote_grads = {name: p.grad.detach().cpu().clone() for name, p in student.named_parameters() if p.grad is not None}
    assert remote_grads and all(torch.isfinite(g).all() for g in remote_grads.values())
    optimizer.step()
    remote_update = param.detach().clone()
    with torch.no_grad():
        param.copy_(before)
    student.zero_grad(set_to_none=True)
    torch.manual_seed(123)
    live, _ = distill_step(student, model, bias.cuda(), live_batch, 3, ARGS)
    live.backward()
    assert torch.isfinite(remote) and torch.allclose(remote, live, rtol=1e-3), (remote.item(), live.item())
    live_grads = {name: p.grad for name, p in student.named_parameters() if p.grad is not None}
    assert live_grads.keys() == remote_grads.keys()
    for name, grad in live_grads.items():
        torch.testing.assert_close(grad.cpu(), remote_grads[name], rtol=5e-3, atol=1e-5, msg=lambda m, name=name: f"{name}: {m}")
    optimizer.step()
    torch.testing.assert_close(param, remote_update, rtol=5e-3, atol=1e-6)


def test_pause_and_resume_throttle_the_real_server_without_losing_or_repeating_rows(tok, teacher, monkeypatch):
    logs = []
    train = corpus(tok, 80, 11)
    args = ServeArguments(output_dir="x", extract_batch=BATCH, seq_len=SEQ, prefetch=2, handshake_s=10, join_grace_s=0, feed_stop_s=2, log_every=10_000)
    addr, srv = start_server(tok, teacher, args, monkeypatch, train, corpus(tok, 8, 12), log=logs.append)
    ds = None
    try:
        ref = reference_rows(tok, teacher, args, train, 10)
        ds = stream.RemoteTargets(addr, capacity=8, high=0.5, low=0.25, seed=SEED)  # PAUSE after two batches
        for _ in range(100):
            if ds.buffer.paused:
                break
            threading.Event().wait(0.05)
        assert ds.buffer.paused and any("PAUSE" in m for m in logs)
        got = take(ds, 28)  # draining below the low-water mark RESUMEs the server
        assert any("RESUME" in m for m in logs)
    finally:
        if ds is not None:
            ds.close()
        srv.close()
    assert all(same(r, f) for r, f in zip(got, ref[:28], strict=True))


def test_batch_stream_refuses_a_position_beyond_the_dataset():
    from teacher_server import batch_stream

    with pytest.raises(EOFError, match="before the requested position"):
        next(batch_stream(lambda: [{"input_ids": [1]}] * 3, lambda i: (lambda rows: rows), 2, start_batch=5))


def test_serve_refuses_a_first_frame_that_is_not_start_and_a_malformed_request():
    from teacher_server import serve

    args = SimpleNamespace(handshake_s=2, extract_batch=4)
    a, b = socket.socketpair()
    stream.send_frame(a, stream.PAUSE)
    with pytest.raises(stream.ProtocolError, match="expected START"):
        serve(b, None, None, args, [], lambda _: None)
    for bad in (b'{"seed": -1}', b'{"seed": 1, "start_seq": -3}', b'{"seed": "x"}'):
        stream.send_frame(a, stream.START, bad)
        with pytest.raises(stream.ProtocolError, match="bad START"):
            serve(b, None, None, args, [], lambda _: None)
    a.close(), b.close()
