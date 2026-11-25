import itertools
import json
import queue
import socket
import struct
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import stream
from store import StoreWriter, TargetStore, batch_arrays, collate_store, rows_from_arrays
from targets import Targets

pytestmark = pytest.mark.timeout(60)


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


def make_batch(B=3, L=10, H=6, nbytes=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(5, 1000, (B, L), generator=g)
    labels = torch.where(torch.rand(B, L, generator=g) < 0.4, ids, torch.full_like(ids, -100))
    labels[:, 0] = ids[:, 0]  # every sequence has at least one masked position
    n = int((labels != -100).sum())
    e = torch.empty(0)
    tg = Targets(e.long().reshape(0, 0), e.float().reshape(0, 0), e.float(), torch.ones(n, dtype=torch.bool),
                 torch.randn(n, nbytes, generator=g).half())
    return ids, labels, tg


def test_frame_and_batch_roundtrip():
    a, b = socket.socketpair()
    arrays = {"x": np.arange(12, dtype=np.uint16).reshape(3, 4), "y": np.ones((0, 0), np.float32), "z": np.arange(5, dtype=np.int64)}
    stream.send_frame(a, stream.BATCH, *stream.encode_batch("train", arrays))
    kind, buf = stream.recv_frame(b)
    split, out = stream.decode_batch(buf)
    assert kind == stream.BATCH and split == "train"
    assert out.keys() == arrays.keys()
    for k, want in arrays.items():
        assert out[k].dtype == want.dtype and out[k].shape == want.shape and (out[k] == want).all()
    a.close(), b.close()


def test_network_rows_are_store_rows(tmp_path):
    """A batch that went through the wire collates to exactly what the same batch read back from a store does."""
    ids, labels, tg = make_batch()
    arrays = batch_arrays(ids, labels, tg)
    w = StoreWriter(tmp_path / "s", {"x": 1})
    w.add(ids, labels, tg)
    w.close()
    store = TargetStore(tmp_path / "s")
    a, b = socket.socketpair()
    stream.send_frame(a, stream.BATCH, *stream.encode_batch("train", arrays))
    _, wire = stream.decode_batch(stream.recv_frame(b)[1])
    rows = rows_from_arrays(wire)
    assert len(rows) == len(store) == 3
    a.close(), b.close()
    got = collate_store(rows)
    want = collate_store([store[i] for i in range(3)])
    assert got.keys() == want.keys()
    for k in want:
        assert torch.equal(got[k], want[k]), k


def test_buffer_hysteresis_pauses_at_95_resumes_at_70():
    events = []
    buf = stream.ReceiveBuffer(100, 0.95, 0.70, lambda: events.append("pause"), lambda: events.append("resume"))
    buf.put(range(94))
    assert events == []
    buf.put([94])  # 95%: pause, once
    buf.put(range(95, 99))
    assert events == ["pause"]
    for _ in range(28):  # 99 -> 71 rows
        buf.get()
    assert events == ["pause"]
    buf.get()  # 70 rows: resume, once
    assert events == ["pause", "resume"]
    buf.put(range(10))  # 80: below the high mark, no second pause
    assert events == ["pause", "resume"]
    buf.put(range(15))  # 95 again
    assert events == ["pause", "resume", "pause"]


def test_buffer_blocks_until_data_and_surfaces_errors():
    buf = stream.ReceiveBuffer(10)
    got = []
    t = threading.Thread(target=lambda: got.append(buf.get()))
    t.start()
    t.join(0.2)
    assert t.is_alive()  # starved: waits instead of returning garbage
    buf.put(["row"])
    t.join(2)
    assert got == ["row"] and buf.stall_s > 0
    buf.fail(ConnectionError("gone"))
    with pytest.raises(RuntimeError, match="teacher stream failed"):
        buf.get()


def test_buffer_bounds_inflight_frame_and_preserves_order():
    buf = stream.ReceiveBuffer(3, high=0.9, low=0.5)
    thread = threading.Thread(target=lambda: buf.put(list(range(8))))
    thread.start()
    got = [buf.get() for _ in range(8)]
    thread.join(2)
    assert not thread.is_alive() and got == list(range(8))
    assert buf.received == buf.consumed == 8


def test_buffer_close_wakes_blocked_producer():
    buf = stream.ReceiveBuffer(1)
    buf.put([1])
    errors = []
    def produce():
        try:
            buf.put([2])
        except RuntimeError as exc:
            errors.append(exc)
    thread = threading.Thread(target=produce)
    thread.start()
    buf.fail(ConnectionError("closed"))
    thread.join(2)
    assert not thread.is_alive() and errors


def test_protocol_rejects_wrong_kind_and_split():
    arrays = batch_arrays(*make_batch())
    frame = bytearray(b"".join(bytes(m.cast("B")) for m in map(memoryview, stream.encode_batch("train", arrays)) if m.nbytes))
    with pytest.raises(stream.ProtocolError, match="expected eval"):
        stream.RemoteTargets._batch(stream.BATCH, frame, "eval")
    with pytest.raises(stream.ProtocolError, match="expected train"):
        stream.RemoteTargets._batch(stream.TRAIN, frame, "train")


def test_handshake_timeout_is_enforced():
    ready, ports, done = threading.Event(), [], threading.Event()
    def server():
        with socket.create_server(("127.0.0.1", 0)) as srv:
            ports.append(srv.getsockname()[1])
            ready.set()
            conn, _ = srv.accept()
            with conn:
                done.wait(2)
    thread = threading.Thread(target=server)
    thread.start()
    assert ready.wait(2)
    try:
        with pytest.raises(TimeoutError):
            stream.RemoteTargets(f"127.0.0.1:{ports[0]}", connect_timeout=0.2, reconnect_s=0)
    finally:
        done.set()
        thread.join(2)


def test_remote_dataset_closed_if_teacher_metadata_invalid(monkeypatch):
    import distill

    closed = []
    class Remote:
        def __init__(self, *args, **kwargs):
            self.meta = {}
        def close(self):
            closed.append(True)
    monkeypatch.setattr(distill, "configure_cuda", lambda: None)
    monkeypatch.setattr(distill.AutoTokenizer, "from_pretrained", lambda *_: None)
    monkeypatch.setattr(distill, "teacher_decoder_params", lambda _: (torch.ones(2, 2), torch.ones(2)))
    monkeypatch.setattr(distill, "RemoteTargets", Remote)
    args = distill.DistillArguments(output_dir="x", teacher_addr="localhost:1")
    with pytest.raises(SystemExit):
        distill.train(args)
    assert closed == [True]


def test_trainer_dataloader_keeps_every_masked_position(tmp_path):
    """Regression: for an IterableDataset accelerate's default dispatch_batches=True slices *every* tensor of a batch to
    the batch size, which silently cut the per-position kd_* targets to one row per sequence (KL ~0.02 instead of ~4)."""
    from transformers import Trainer

    from distill import DistillArguments

    class Rows(torch.utils.data.IterableDataset):
        def __iter__(self):
            for seed in range(100):
                yield from rows_from_arrays(batch_arrays(*make_batch(B=4, seed=seed)))

    args = DistillArguments(output_dir=str(tmp_path), use_cpu=True, bf16=False, per_device_train_batch_size=3, eval_strategy="no", report_to="none")
    trainer = Trainer(model=torch.nn.Linear(1, 1), args=args, train_dataset=Rows(), data_collator=collate_store)
    batch = next(iter(trainer.get_train_dataloader()))
    n_masked = int((batch["labels"] != -100).sum())
    assert n_masked > len(batch["labels"])  # several masked positions per sequence
    assert batch["kd_hidden"].shape[0] == batch["kd_dense"].shape[0] == n_masked


def test_frame_arrays_are_8_byte_aligned():
    arrays = {"a": np.zeros(3, np.uint16), "b": np.zeros((2, 5), np.int32), "c": np.zeros(0, np.float32), "d": np.zeros(7, np.float16)}
    buf = bytearray(b"".join(bytes(memoryview(p).cast("B")) for p in stream.encode_batch("train", arrays)))
    _, out = stream.decode_batch(buf)
    assert all(v.flags.aligned and v.ctypes.data % 8 == 0 for v in out.values() if v.size), {k: v.ctypes.data % 8 for k, v in out.items()}


def test_oversized_frame_is_refused_not_allocated():
    a, b = socket.socketpair()
    a.sendall(stream.HEAD.pack(b"S", 1 << 60))
    with pytest.raises(stream.ProtocolError):
        stream.recv_frame(b)
    assert isinstance(stream.ProtocolError(), OSError)  # the server's per-connection handler catches it
    a.close(), b.close()


def test_silent_server_raises_instead_of_hanging():
    buf = stream.ReceiveBuffer(10, stall_timeout=0.3)
    with pytest.raises(TimeoutError, match="no teacher data"):
        buf.get()


# ---------------------------------------------------------------- consistency, protocol, handshake, reconnect

def frame_of(arrays, split="train"):
    return bytearray(b"".join(bytes(m.cast("B")) for m in map(memoryview, stream.encode_batch(split, arrays)) if m.nbytes))


def test_inconsistent_frames_are_refused_not_misaligned():
    good = batch_arrays(*make_batch(B=3))
    stream.RemoteTargets._batch(stream.BATCH, frame_of(good), "train")  # consistent: accepted
    broken = {
        "a dense flag missing": good | {"dense": good["dense"][:-1]},
        "counts disagree with labels": good | {"m": good["m"] + 1},
        "hidden rows missing": good | {"hidden": good["hidden"][:-1]},
        "wrong dtype": good | {"hidden": good["hidden"].astype("float32")},
        "missing array": {k: v for k, v in good.items() if k != "tail"},
    }
    for why, arrays in broken.items():
        with pytest.raises(stream.ProtocolError, match="inconsistent train batch"):
            stream.RemoteTargets._batch(stream.BATCH, frame_of(arrays), "train")
        assert why


def test_control_frames_are_size_limited():
    a, b = socket.socketpair()
    a.sendall(stream.HEAD.pack(b"P", stream.CONTROL_MAX + 1))
    with pytest.raises(stream.ProtocolError):
        stream.recv_frame(b, limit=stream.CONTROL_MAX)
    a.close(), b.close()


def test_server_drops_a_client_that_never_sends_start():
    from types import SimpleNamespace

    from teacher_server import serve

    client, server = socket.socketpair()  # connects and says nothing
    args = SimpleNamespace(handshake_s=0.3)
    t0 = time.monotonic()
    with pytest.raises(TimeoutError):
        serve(server, None, None, args, [], lambda _: None)
    assert time.monotonic() - t0 < 3
    client.close(), server.close()


# ---------------------------------------------------------------- fake teacher servers (the real handshake, deterministic data)

def rows_of(index, B):
    """The rows batch `index` holds in these tests: a pure function of the index, like the real stream."""
    return rows_from_arrays(batch_arrays(*make_batch(B=B, seed=index)))


def handshake(conn, batch, n_eval=2, **meta_over):
    """Server side of the handshake: read START, answer HELLO (+ eval frames) + TRAIN. -> (request, first batch index to send)."""
    kind, raw = stream.recv_frame(conn)
    assert kind == stream.START
    request = json.loads(bytes(raw))
    start_batch, drop = divmod(request["start_seq"], batch)
    evals = [stream.encode_batch("eval", batch_arrays(*make_batch(B=2, seed=1000 + i))) for i in range(n_eval)]
    meta = {"protocol": stream.PROTOCOL, "hidden": 6, "batch": batch, "seed": request["seed"], "start_batch": start_batch,
            "drop": drop, "eval_sha256": stream.frames_digest(evals), "stream_sha256": "a" * 64} | meta_over
    blob = json.dumps(meta).encode()
    stream.send_frame(conn, stream.HELLO, struct.pack("!I", len(blob)), blob)
    for frame in evals:
        stream.send_frame(conn, stream.BATCH, *frame)
    stream.send_frame(conn, stream.TRAIN)
    return request, start_batch


def train_frame(index, B):
    return stream.encode_batch("train", batch_arrays(*make_batch(B=B, seed=index)), index=index)


def fake_server(port_box, ready, log, B=8):
    """Pushes batches of 8 sequences as fast as it can until PAUSE; sends more after RESUME."""
    srv = socket.create_server(("127.0.0.1", 0))
    port_box.append(srv.getsockname()[1])
    ready.set()
    conn, _ = srv.accept()
    stream.tune(conn)
    _, index = handshake(conn, B)
    running, stop = threading.Event(), threading.Event()
    running.set()

    def control():
        try:
            while True:
                kind, _ = stream.recv_frame(conn)
                if kind == stream.PAUSE:
                    running.clear()
                    log.append(("pause", index))
                elif kind == stream.RESUME:
                    running.set()
                    log.append(("resume", index))
                elif kind == stream.BYE:
                    break
        except OSError:
            pass
        stop.set(), running.set()

    threading.Thread(target=control, daemon=True).start()
    while not stop.is_set():
        running.wait()
        if stop.is_set():
            break
        try:
            stream.send_frame(conn, stream.BATCH, *train_frame(index, B))
        except OSError:  # the client hung up
            break
        index += 1
    conn.close()
    srv.close()


def test_remote_targets_throttles_the_server_and_never_drops():
    port, ready, log = [], threading.Event(), []
    threading.Thread(target=fake_server, args=(port, ready, log), daemon=True).start()
    ready.wait(5)
    ds = stream.RemoteTargets(f"127.0.0.1:{port[0]}", capacity=64, high=0.95, low=0.70)
    assert len(ds.eval) == 4
    deadline = 50
    while not log and deadline:  # the server floods until the buffer crosses 95% (>= 61 of 64 sequences)
        threading.Event().wait(0.05)
        deadline -= 1
    assert log and log[0][0] == "pause" and ds.buffer.paused
    threading.Event().wait(0.3)  # in-flight batches have landed
    assert ds.buffer.fill >= 0.95 and ds.buffer.received == len(ds.buffer.rows) <= 64 + 8 * 4
    it = iter(ds)
    n_before = ds.buffer.received
    for _ in range(len(ds.buffer.rows) - 44):  # drain to 44/64 = 69% <= 70%
        next(it)
    for _ in range(100):
        if any(k == "resume" for k, _ in log):
            break
        threading.Event().wait(0.05)
    assert any(k == "resume" for k, _ in log)
    row = next(it)  # rows are TargetStore-format and decode
    assert row["input_ids"].dtype == np.uint16 and row["kd_hidden"].dtype == np.float16
    assert ds.buffer.received >= n_before  # nothing lost, the stream continues after RESUME
    ds.close()


def sequence(rows):
    return [tuple(r["input_ids"]) for r in rows]


def test_the_stream_arrives_in_order_and_resumes_inside_a_batch():
    """START asks for sequence 6 of batches of 4: batch 1 with its first 2 rows dropped, then batches 2, 3, ..."""
    port, ready, log = [], threading.Event(), []
    threading.Thread(target=fake_server, args=(port, ready, log), kwargs=dict(B=4), daemon=True).start()
    ready.wait(5)
    ds = stream.RemoteTargets(f"127.0.0.1:{port[0]}", capacity=64, start_seq=6)
    it = iter(ds)
    got = [next(it) for _ in range(10)]
    want = rows_of(1, 4)[2:] + rows_of(2, 4) + rows_of(3, 4)[:4]
    assert sequence(got) == sequence(want)
    assert ds.rows_consumed == 6 + 10  # absolute: earlier run + this one
    ds.close()


def one_shot_server(**meta_over):
    """Answers one connection with the handshake (meta overridden) and holds it until the client leaves."""
    srv = socket.create_server(("127.0.0.1", 0))

    def run():
        conn, _ = srv.accept()
        with conn:
            try:
                handshake(conn, 4, **meta_over)
                stream.recv_frame(conn)
            except OSError:
                pass
        srv.close()

    threading.Thread(target=run, daemon=True).start()
    return srv.getsockname()[1]


def test_protocol_version_mismatch_fails_at_the_handshake():
    with pytest.raises(stream.ProtocolError, match="speaks protocol"):
        stream.RemoteTargets(f"127.0.0.1:{one_shot_server(protocol=stream.PROTOCOL + 1)}", connect_timeout=2)
    with pytest.raises(stream.ProtocolError, match="protocol None"):
        stream.RemoteTargets(f"127.0.0.1:{one_shot_server(protocol=None)}", connect_timeout=2)  # an old server: no version


def test_server_answering_a_different_request_or_eval_set_is_refused():
    with pytest.raises(stream.ProtocolError, match="seed/position"):
        stream.RemoteTargets(f"127.0.0.1:{one_shot_server(seed=999)}", seed=1, connect_timeout=2)
    with pytest.raises(stream.ProtocolError, match="eval set"):
        stream.RemoteTargets(f"127.0.0.1:{one_shot_server(eval_sha256='0' * 64)}", connect_timeout=2)


def serve_frames(conn, indices, B=4):
    for i in indices:
        stream.send_frame(conn, stream.BATCH, *train_frame(i, B))


def test_a_gap_or_a_repeat_in_the_stream_is_an_error():
    for bad in ([0, 2], [0, 0]):
        srv = socket.create_server(("127.0.0.1", 0))

        def run(srv=srv, bad=bad):
            conn, _ = srv.accept()
            with conn:
                handshake(conn, 4)
                serve_frames(conn, bad)
                try:
                    stream.recv_frame(conn)
                except OSError:
                    pass
            srv.close()

        threading.Thread(target=run, daemon=True).start()
        ds = stream.RemoteTargets(f"127.0.0.1:{srv.getsockname()[1]}", reconnect_s=0)
        it = iter(ds)
        [next(it) for _ in range(4)]  # batch 0 is fine
        with pytest.raises(RuntimeError, match="teacher stream failed"):
            next(it)
        assert "stream position broken" in str(ds.buffer.error)
        ds.close()


def flaky_server(port_box, ready, requests, drop_after=3, expect_pause=None, B=4, session_meta=None):
    """Session 1 sends `drop_after` batches and drops; session 2 serves the stream from whatever position it is asked for."""
    srv = socket.create_server(("127.0.0.1", 0))
    port_box.append(srv.getsockname()[1])
    ready.set()
    for session in (1, 2):
        conn, _ = srv.accept()
        stream.tune(conn)
        request, index = handshake(conn, B, **((session_meta or {}).get(session, {})))
        requests.append(request)
        if session == 1:
            serve_frames(conn, range(index, index + drop_after), B)
            conn.close()  # the drop
            continue
        conn.settimeout(1.0)
        try:
            kind, _ = stream.recv_frame(conn)  # does the client re-assert PAUSE on the fresh connection?
            if expect_pause is not None:
                expect_pause.append(kind == stream.PAUSE)
        except OSError:
            if expect_pause is not None:
                expect_pause.append(False)
        conn.settimeout(None)
        try:
            serve_frames(conn, range(index, index + 3), B)
        except OSError:
            pass
        conn.close()
    srv.close()


def test_reconnect_resumes_at_the_next_batch_with_nothing_lost_or_repeated():
    port, ready, requests = [], threading.Event(), []
    threading.Thread(target=flaky_server, args=(port, ready, requests), daemon=True).start()
    ready.wait(5)
    ds = stream.RemoteTargets(f"127.0.0.1:{port[0]}", capacity=64, reconnect_s=20, seed=5)
    it = iter(ds)
    got = [next(it) for _ in range(6 * 4)]  # 3 batches before the drop + 3 after, 4 sequences each
    assert sequence(got) == sequence(list(itertools.chain.from_iterable(rows_of(i, 4) for i in range(6))))
    assert [r["start_seq"] for r in requests] == [0, 12] and {r["seed"] for r in requests} == {5}  # asked for exactly where it broke
    assert ds.reconnects == 1 and ds.stats()["stream_reconnects"] == 1 and ds.rows_consumed == 24
    ds.close()


def test_reconnect_carries_the_paused_state_over():
    port, ready, requests, seen = [], threading.Event(), [], []
    threading.Thread(target=flaky_server, args=(port, ready, requests), kwargs=dict(expect_pause=seen), daemon=True).start()
    ready.wait(5)
    ds = stream.RemoteTargets(f"127.0.0.1:{port[0]}", capacity=12, high=0.5, low=0.25, reconnect_s=20)
    for _ in range(100):  # 3 batches x 4 = 12 sequences arrive, the buffer is full and paused when the connection drops
        if ds.buffer.paused and ds.buffer.fill >= 1.0:
            break
        time.sleep(0.05)
    assert ds.buffer.paused
    for _ in range(100):
        if seen:
            break
        time.sleep(0.05)
    assert seen == [True]  # the fresh connection was told to PAUSE straight away
    ds.close()


def test_a_teacher_or_eval_set_that_changed_across_a_reconnect_is_refused():
    for change in ({"teacher": "two"}, {"n_eval": 3}, {"stream_sha256": "b" * 64}):
        port, ready, requests = [], threading.Event(), []
        threading.Thread(target=flaky_server, args=(port, ready, requests),
                         kwargs=dict(session_meta={1: {"teacher": "one"}, 2: {"teacher": "one"} | change}), daemon=True).start()
        ready.wait(5)
        ds = stream.RemoteTargets(f"127.0.0.1:{port[0]}", reconnect_s=20, capacity=4)
        it = iter(ds)
        with pytest.raises(RuntimeError, match="teacher stream failed"):
            for _ in range(100):
                next(it)
        assert "changed across a reconnect" in str(ds.buffer.error)
        ds.close()


def test_reconnect_gives_up_after_its_budget():
    ds = stream.RemoteTargets(f"127.0.0.1:{one_shot_server()}", reconnect_s=2, connect_timeout=1)
    ds.sock.shutdown(socket.SHUT_RDWR)  # the server serves one connection only: it is gone for good after this
    t0 = time.monotonic()
    with pytest.raises(RuntimeError, match="teacher stream failed"):
        next(iter(ds))
    assert time.monotonic() - t0 < 15 and "unreachable" in str(ds.buffer.error)
    ds.close()


def test_remote_resume_needs_the_exact_stream_position(tmp_path):
    from distill import DistillArguments, train

    ckpt = tmp_path / "checkpoint-5"
    ckpt.mkdir()
    args = DistillArguments(output_dir=str(tmp_path), teacher_addr="localhost:1", resume_from_checkpoint=str(ckpt))
    with pytest.raises(ValueError, match="stream_position.json"):
        train(args)  # fails before connecting or loading anything


def test_heartbeats_keep_a_slow_start_from_looking_like_a_stall():
    buf = stream.ReceiveBuffer(10, stall_timeout=0.6)
    stop = threading.Event()

    def server_seeking():  # no data for 1.5 s, but alive: the real server's heartbeat every few seconds
        t0 = time.monotonic()
        while time.monotonic() - t0 < 1.5:
            buf.touch()
            time.sleep(0.2)
        buf.put(["row"])
        stop.set()

    threading.Thread(target=server_seeking, daemon=True).start()
    assert buf.get() == "row"  # waited 1.5 s > the 0.6 s stall limit without raising
    stop.wait(2)
    silent = stream.ReceiveBuffer(10, stall_timeout=0.3)
    with pytest.raises(TimeoutError):
        silent.get()  # no heartbeat, no data: still a stall


def test_server_refuses_a_malformed_start():
    from types import SimpleNamespace

    from teacher_server import serve

    for bad in ({"seed": -1, "start_seq": 0}, {"seed": 0, "start_seq": "x"}, {"start_seq": 0}):
        client, server = socket.socketpair()
        args = SimpleNamespace(handshake_s=5, extract_batch=4)
        stream.send_frame(client, stream.START, json.dumps(bad).encode())
        with pytest.raises((stream.ProtocolError, KeyError)):
            serve(server, None, None, args, [], lambda _: None)
        client.close(), server.close()


# ---------------------------------------------------------------- the server's stream is a pure function of (seed, position)

def test_batch_stream_is_the_same_whatever_position_it_is_entered_at():
    from itertools import islice

    from teacher_server import batch_stream, mask_seed

    data = [{"input_ids": [i, i + 1]} for i in range(40)]
    collate = lambda i: lambda rows: (i, [r["input_ids"][0] for r in rows],
                                      torch.rand(3, generator=torch.Generator().manual_seed(mask_seed(7, i))).tolist())
    full = list(islice(batch_stream(lambda: data, collate, 4), 10))
    assert [i for i, _ in full] == list(range(10)) and full[2][1][1][0] == 8  # batch 2 = sequences 8..11
    for start in (0, 3, 9):
        assert list(islice(batch_stream(lambda: data, collate, 4, start_batch=start), 10 - start)) == full[start:]
    assert len({mask_seed(7, i) for i in range(1000)}) == 1000 and mask_seed(7, 5) != mask_seed(8, 5)
    with pytest.raises(EOFError):  # a finite dataset ends loudly
        list(batch_stream(lambda: data[:10], collate, 4))


def test_training_stream_identity_changes_with_source_and_teacher():
    from types import SimpleNamespace

    from teacher_server import ServeArguments, stream_identity

    args = ServeArguments(output_dir="x", dataset_revision="commit")
    tok = SimpleNamespace(backend_tokenizer=SimpleNamespace(to_str=lambda: "tokenizer"))
    teacher = SimpleNamespace(model_fingerprint="model")
    first = stream_identity(args, tok, teacher)
    args.shuffle_buffer += 1
    assert stream_identity(args, tok, teacher) != first
    args.shuffle_buffer -= 1
    teacher.model_fingerprint = "different"
    assert stream_identity(args, tok, teacher) != first
    args.dataset_revision = ""
    with pytest.raises(ValueError, match="resolved dataset revision"):
        stream_identity(args, tok, teacher)


def test_feed_exhaustion_is_terminal(monkeypatch):
    from teacher_server import Feed, FeedError, ServeArguments

    monkeypatch.setattr("teacher_server.make_train_dataset", lambda *_: [{"input_ids": [1]}])
    monkeypatch.setattr("teacher_server.make_collator", lambda *_: lambda rows: rows)
    args = ServeArguments(output_dir="x", extract_batch=2)
    feed = Feed(None, args, 0, 0, lambda _: None)
    try:
        with pytest.raises(FeedError, match="training stream ended"):
            feed.get(timeout=1)
    finally:
        feed.close()


def test_blocked_feed_is_logged_and_abandoned_not_fatal():
    """A trainer leaving during the dataset's slow startup must not kill the server."""
    from teacher_server import Feed, ServeArguments

    release, entered, logs = threading.Event(), threading.Event(), []

    def slow_dataset(*_):
        entered.set()
        release.wait(10)  # blocked inside the dataset, like the shuffle-buffer fill
        return [{"input_ids": [1]}] * 100

    import teacher_server
    mp = pytest.MonkeyPatch()
    mp.setattr(teacher_server, "make_train_dataset", slow_dataset)
    mp.setattr(teacher_server, "make_collator", lambda *_: lambda rows: rows)
    try:
        feed = Feed(None, ServeArguments(output_dir="x", extract_batch=2, feed_stop_s=0.2), 0, 0, logs.append)
        assert entered.wait(5)
        feed.close()  # does not raise
        assert any("abandoning" in m for m in logs) and feed.thread.is_alive()
        release.set()
        feed.thread.join(5)
        assert not feed.thread.is_alive()  # it exits once the blocked call returns
    finally:
        release.set()
        mp.undo()


def test_terminal_feed_error_reaches_client_without_reconnect():
    ready, port = threading.Event(), []
    def server():
        with socket.create_server(("127.0.0.1", 0)) as srv:
            port.append(srv.getsockname()[1])
            ready.set()
            conn, _ = srv.accept()
            with conn:
                handshake(conn, batch=4)
                stream.send_frame(conn, stream.ERROR, b"dataset exhausted")
                threading.Event().wait(1)
    thread = threading.Thread(target=server, daemon=True)
    thread.start()
    assert ready.wait(2)
    ds = stream.RemoteTargets(f"127.0.0.1:{port[0]}", reconnect_s=3)
    try:
        with pytest.raises(RuntimeError, match="dataset exhausted"):
            next(iter(ds))
        assert ds.reconnects == 0
    finally:
        ds.close()
        thread.join(2)


def test_server_handshake_send_has_a_deadline():
    from types import SimpleNamespace

    from teacher_server import serve

    client, server = socket.socketpair()
    class SlowSend:
        def __getattr__(self, name):
            return getattr(server, name)
        def send(self, data):
            threading.Event().wait(0.03)
            return server.send(data[:1])
    teacher = SimpleNamespace(W=torch.ones(2, 2), hidden=2, fingerprint="h", stream_identity="a" * 64)
    args = SimpleNamespace(handshake_s=0.2, extract_batch=4, seq_len=4, temperature=2, teacher="t")
    errors = []
    def run():
        try:
            serve(SlowSend(), teacher, None, args, [[b"x" * 100]], lambda _: None)
        except OSError as e:
            errors.append(e)
    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    stream.send_frame(client, stream.START, b'{"seed":0}')
    thread.join(2)
    client.close()
    server.close()
    assert not thread.is_alive() and errors and isinstance(errors[0], TimeoutError)


# ---------------------------------------------------------------- server feed retry and stream loop

def _feed_args(**kw):
    from teacher_server import ServeArguments
    return ServeArguments(output_dir="x", extract_batch=2, **kw)


def test_feed_retries_a_failing_dataset_at_the_same_batch(monkeypatch):
    from teacher_server import Feed

    calls, logs = [], []

    def dataset(*_):
        calls.append(1)
        if len(calls) == 1:
            raise ConnectionError("dataset host hiccup")
        return [{"input_ids": [i]} for i in range(100)]

    monkeypatch.setattr("teacher_server.make_train_dataset", dataset)
    monkeypatch.setattr("teacher_server.make_collator", lambda *_: lambda rows: [r["input_ids"][0] for r in rows])
    feed = Feed(None, _feed_args(), 0, 3, logs.append)  # entered at batch 3 -> sequences 6, 7 first
    try:
        assert feed.get(timeout=10) == (3, [6, 7]) and feed.get(timeout=5) == (4, [8, 9])
        assert any("retrying batch 3" in m for m in logs) and len(calls) == 2
    finally:
        feed.close()


def test_feed_gives_up_after_consecutive_failures(monkeypatch):
    from teacher_server import Feed, FeedError

    def broken(*_):
        raise ConnectionError("down")

    monkeypatch.setattr("teacher_server.make_train_dataset", broken)
    feed = Feed(None, _feed_args(max_feed_failures=2), 0, 0, lambda _: None)
    try:
        with pytest.raises(FeedError, match="data feed failed at batch 0"):
            for _ in range(20):
                try:
                    feed.get(timeout=1)
                except queue.Empty:
                    continue
    finally:
        feed.close()


def test_teacher_call_produces_a_consistent_wire_frame():
    """Teacher.__call__ on a stand-in encoder: masked positions only, fp16 hidden states, arrays pass the frame checks."""
    from types import SimpleNamespace

    from teacher_server import Teacher

    H, V, B, L = 6, 20, 3, 10
    ids, labels, _ = make_batch(B=B, L=L)
    head = torch.nn.Linear(H, H)
    model = SimpleNamespace(model=lambda input_ids, attention_mask: SimpleNamespace(last_hidden_state=torch.randn(*input_ids.shape, H)),
                            head=head)
    t = object.__new__(Teacher)
    t.args, t.device, t.model = SimpleNamespace(temperature=2.0), "cpu", model
    t.W, t.b = torch.randn(V, H), torch.zeros(V)
    arrays = t({"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": labels})
    n = int((labels != -100).sum())
    assert arrays["hidden"].shape == (n, H) and arrays["hidden"].dtype == np.float16
    stream.RemoteTargets._batch(stream.BATCH, frame_of(arrays), "train")


# ---------------------------------------------------------------- handshake validation, heartbeats, helpers

def raw_server(script):
    """Accepts one connection and runs script(conn); returns the port."""
    srv = socket.create_server(("127.0.0.1", 0))

    def run():
        conn, _ = srv.accept()
        with conn:
            try:
                script(conn)
            except OSError:
                pass
        srv.close()

    threading.Thread(target=run, daemon=True).start()
    return srv.getsockname()[1]


def test_malformed_hello_is_refused():
    def bad_length(conn):
        stream.recv_frame(conn)
        blob = json.dumps({"protocol": stream.PROTOCOL}).encode()
        stream.send_frame(conn, stream.HELLO, struct.pack("!I", len(blob) + 5), blob)
        stream.recv_frame(conn)

    with pytest.raises(stream.ProtocolError, match="invalid HELLO length"):
        stream.RemoteTargets(f"127.0.0.1:{raw_server(bad_length)}", connect_timeout=2)

    def not_hello(conn):
        stream.recv_frame(conn)
        stream.send_frame(conn, stream.TRAIN)
        stream.recv_frame(conn)

    with pytest.raises(stream.ProtocolError, match="expected HELLO"):
        stream.RemoteTargets(f"127.0.0.1:{raw_server(not_hello)}", connect_timeout=2)


def test_stream_identity_and_train_marker_are_validated():
    with pytest.raises(stream.ProtocolError, match="stream identity"):
        stream.RemoteTargets(f"127.0.0.1:{one_shot_server(stream_sha256=None)}", connect_timeout=2)

    def loud_train_marker(conn):
        stream.recv_frame(conn)
        blob = json.dumps({"protocol": stream.PROTOCOL, "hidden": 6, "batch": 4, "seed": 0, "start_batch": 0, "drop": 0,
                           "eval_sha256": stream.frames_digest([]), "stream_sha256": "a" * 64}).encode()
        stream.send_frame(conn, stream.HELLO, struct.pack("!I", len(blob)), blob)
        stream.send_frame(conn, stream.TRAIN, b"junk")
        stream.recv_frame(conn)

    with pytest.raises(stream.ProtocolError, match="TRAIN marker must be empty"):
        stream.RemoteTargets(f"127.0.0.1:{raw_server(loud_train_marker)}", connect_timeout=2, seed=0)


def test_heartbeats_are_consumed_silently_and_a_server_error_stops_the_run():
    def script(conn):
        handshake(conn, 4)
        for _ in range(3):
            stream.send_frame(conn, stream.HEARTBEAT)
        stream.send_frame(conn, stream.BATCH, *train_frame(0, 4))
        stream.send_frame(conn, stream.ERROR, b"disk on fire")
        stream.recv_frame(conn)

    ds = stream.RemoteTargets(f"127.0.0.1:{raw_server(script)}", capacity=64, seed=0)
    it = iter(ds)
    assert len([next(it) for _ in range(4)]) == 4  # the heartbeats did not produce rows or errors
    with pytest.raises(RuntimeError, match="disk on fire"):
        next(it)
    ds.close()
    ds.close()  # closing twice is harmless (the socket is already shut down)


def test_recv_frame_honours_its_deadline():
    a, b = socket.socketpair()
    with pytest.raises(TimeoutError):
        stream.recv_frame(b, deadline=time.monotonic() - 1)
    a.close(), b.close()


def test_keepalive_options_on_a_platform_without_ioctl(monkeypatch):
    calls = []
    for name, value in (("TCP_KEEPIDLE", 101), ("TCP_KEEPINTVL", 102), ("TCP_KEEPCNT", 103)):
        monkeypatch.setattr(socket, name, value, raising=False)

    class Sock:
        def setsockopt(self, *a):
            calls.append(a)

    stream.tune(Sock(), keepalive_s=9)
    values = {a[1]: a[2] for a in calls if a[0] == socket.IPPROTO_TCP and a[1] in (101, 102, 103)}
    assert values == {101: 9, 102: 3, 103: 5}  # idle 9 s, probe every 3 s, five probes


def test_eval_rows_are_indexable():
    ds = stream.RemoteTargets(f"127.0.0.1:{one_shot_server()}", connect_timeout=2)
    try:
        assert len(ds.eval) == 4 and ds.eval[0]["input_ids"].shape[0] > 0
    finally:
        ds.close()
