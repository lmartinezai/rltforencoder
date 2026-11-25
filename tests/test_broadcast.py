"""One teacher pass feeding several trainers at different speeds (hubs), the heartbeats both ways, a trainer that waits for a lost
teacher, and the supervisor. The teacher and the data feed are deterministic stand-ins (batch i is a pure function of i); the
sockets, the wire format, the server loop, RemoteTargets and the supervisor are the real ones."""
import json
import os
import re
import socket
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import stream
import teacher_supervisor
from store import batch_arrays, rows_from_arrays
from teacher_server import FeedError, Hubs, Recording, ReplayFeed, ServeArguments, accept_loop, record

pytestmark = pytest.mark.timeout(90)
B, L = 4, 10  # sequences per batch, tokens per sequence


def make_arrays(seed, width=5):
    """Batch `seed`'s teacher output (arrays of one wire frame): a pure function of the index, like the real stream."""
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(5, 1000, (B, L), generator=g)
    labels = torch.where(torch.rand(B, L, generator=g) < 0.4, ids, torch.full_like(ids, -100))
    labels[:, 0] = ids[:, 0]
    n = int((labels != -100).sum())
    e = torch.empty(0)
    from targets import Targets
    tg = Targets(e.long().reshape(0, 0), e.float().reshape(0, 0), e.float(), torch.ones(n, dtype=torch.bool),
                 torch.randn(n, width, generator=g).half())
    return batch_arrays(ids, labels, tg)


def rows_of(index, width=5):
    return rows_from_arrays(make_arrays(index, width))


def same(row, ref):
    return row.keys() == ref.keys() and all(np.array_equal(np.asarray(row[k]), np.asarray(ref[k])) for k in row)


class CountingFeed:
    """Scripted data feed: batch i carries i in its ids. `n` batches then FeedError("dataset exhausted"); `delay` seconds per batch."""

    def __init__(self, start=0, n=None, delay=0.0):
        self.i, self.n, self.delay, self.closed = start, n, delay, False

    def get(self, timeout=None):
        if self.closed:
            raise FeedError("closed")
        if self.n is not None and self.i >= self.n:
            raise FeedError("dataset exhausted")
        if self.delay:
            time.sleep(self.delay)
        i, self.i = self.i, self.i + 1
        return i, {"input_ids": torch.full((B, L), i)}

    def close(self):
        self.closed = True


class FakeTeacher:
    W, fingerprint, stream_identity, hidden = torch.ones(2, 6), "head", "a" * 64, 6
    calls = 0
    fail_from = None
    width = 5  # bytes per masked position / 2: a wide teacher makes frames bigger than the socket buffers

    def __call__(self, x):
        self.calls += 1
        if self.fail_from is not None and self.calls > self.fail_from:
            raise RuntimeError("CUDA error: an illegal memory access")
        return make_arrays(int(x["input_ids"][0, 0]), self.width)


def server_args(tmp_path, **over):
    kw = dict(output_dir="x", extract_batch=B, seq_len=L, handshake_s=5, heartbeat_s=0.2, client_silence_s=3, window=2,
              ram_batches=3, share_start_s=0, join_grace_s=0, spill_dir=str(tmp_path / "spill"), spill_gb=10, hub_linger_s=30, feed_stop_s=1, log_every=10_000,
              max_clients=4)
    return ServeArguments(**(kw | over))


def wire(parts):
    return b"".join(bytes(memoryview(p).cast("B")) for p in parts if memoryview(p).nbytes)


def read(item):
    """The bytes of what Subscription.next returned, whichever tier it came from."""
    if item[0] == "disk":
        return Path(item[2]).read_bytes()
    assert item[0] == "ram", item
    return wire(item[2])


def until(cond, seconds=10, what="condition"):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if cond():
            return
        time.sleep(0.02)
    raise AssertionError(f"timed out waiting for {what}")


@pytest.fixture()
def hubs_of(tmp_path):
    made = []

    def make(feed=None, teacher=None, log=lambda _: None, **over):
        args = server_args(tmp_path, **over)
        feeds = []

        def make_feed(seed, start):
            feeds.append(feed(seed, start) if callable(feed) else CountingFeed(start))
            return feeds[-1]

        hubs = Hubs(teacher or FakeTeacher(), args, log, make_feed)
        made.append(hubs)
        hubs.feeds, hubs.args = feeds, args
        return hubs

    yield make
    for h in made:
        h.close()


# ---------------------------------------------------------------- hub internals

def test_the_teacher_works_only_while_somebody_is_within_the_window(hubs_of):
    hubs = hubs_of(window=3)
    sub = hubs.subscribe(0, 0)
    until(lambda: sub.hub.head == 3, what="a full window")
    time.sleep(0.5)
    assert sub.hub.head == 3 and hubs.teacher.calls == 3  # nobody is hungry: the teacher idles
    for _ in range(2):
        assert read(sub.next(1))
        sub.advance()
    until(lambda: sub.hub.head == 5, what="the window refilled")
    sub.leave()


def test_a_fast_and_a_slow_subscriber_get_identical_bytes_and_the_backlog_waits_on_disk(hubs_of, tmp_path):
    hubs = hubs_of(window=2, ram_batches=2)
    slow = hubs.subscribe(0, 0)
    hub = slow.hub
    fast = hubs.subscribe(0, 0)
    assert fast.hub is hub and len(hubs.feeds) == 1  # one teacher pass for both
    got = []
    for _ in range(12):  # the fast one reads 12 batches while the slow one reads nothing
        item = fast.next(5)
        got.append(read(item))
        fast.advance()
    assert hub.head >= 12 and slow.cursor == 0 and hub.disk and hub.disk_bytes > 0  # not held back; the backlog is on disk
    assert os.listdir(hub.spill_dir)
    slow_got = []
    for _ in range(12):
        slow_got.append(read(slow.next(5)))
        slow.advance()
    expected = [wire(stream.encode_batch("train", make_arrays(i), index=i)) for i in range(12)]
    assert got == expected and slow_got == expected
    until(lambda: not hub.disk, what="the backlog deleted once both are past it")
    assert not os.listdir(hub.spill_dir) if os.path.isdir(hub.spill_dir) else True
    fast.leave(), slow.leave()


def test_the_backlog_cap_is_the_only_thing_that_slows_the_fast_trainer(hubs_of):
    frame = len(wire(stream.encode_batch("train", make_arrays(0), index=0)))
    hubs = hubs_of(window=2, ram_batches=2, spill_gb=6 * frame / 1e9)  # room for about six spilled batches
    slow = hubs.subscribe(0, 0)
    fast = hubs.subscribe(0, 0)
    hub = slow.hub
    done = threading.Event()
    count = []

    def reader():
        while not done.is_set():
            item = fast.next(0.2)
            if item and item[0] in ("ram", "disk"):
                count.append(item[1])
                fast.advance()

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    until(lambda: len(count) > 8, what="the fast trainer to run ahead")
    time.sleep(0.7)
    stalled_at = len(count)
    time.sleep(0.5)
    assert len(count) == stalled_at and hub.disk_bytes <= 6 * frame + frame  # capped: it waits for the slow one
    for _ in range(6):  # the slow one catches up a little: the fast one moves again
        slow.next(5)
        slow.advance()
    until(lambda: len(count) > stalled_at, what="the fast trainer to resume")
    done.set()
    thread.join(2)
    slow.leave(), fast.leave()


def test_joining_rules_follow_what_the_stream_still_holds(hubs_of):
    hubs = hubs_of(window=2, ram_batches=2)
    a = hubs.subscribe(0, 3)  # a stream entered at batch 3
    hub = a.hub
    until(lambda: hub.head == 5, what="the window")
    assert hub.join(2) is None  # before the stream began
    for start, joins in ((3, True), (5, True), (7, True), (8, False)):  # up to `window` ahead of the head: the teacher will get there
        sub = hub.join(start)
        assert (sub is not None) == joins, start
        if sub:
            sub.leave()
    b = hubs.subscribe(1, 3)  # another seed never shares
    assert b.hub is not hub and len(hubs.feeds) == 2
    for _ in range(4):  # the stream moves on: what its only trainer has passed is dropped, so an old start can no longer join
        a.next(5), a.advance()
    assert hub.tail == 7 and hub.join(3) is None
    c = hubs.subscribe(0, 3)  # before the tail: its own stream, replayed from batch 3
    assert c.hub is not hub and len(hubs.feeds) == 3 and hubs.feeds[2].i >= 3
    assert read(c.next(5)) == wire(stream.encode_batch("train", make_arrays(3), index=3))  # and the same bytes
    for sub in (a, b, c):
        sub.leave()


def test_a_second_trainer_started_later_still_joins_at_the_start_while_the_stream_is_young(hubs_of):
    hubs = hubs_of(window=2, ram_batches=2, share_start_s=60)
    a = hubs.subscribe(0, 0)
    hub = a.hub
    for _ in range(10):  # the first trainer runs ahead: normally everything it passed would be forgotten
        a.next(5), a.advance()
    b = hubs.subscribe(0, 0)  # the second trainer starts a little later, from the beginning
    assert b.hub is hub and len(hubs.feeds) == 1 and hub.tail == 0 and hub.disk
    assert [read(b.next(5)) for _ in range(1)] == [wire(stream.encode_batch("train", make_arrays(0), index=0))]
    a.leave(), b.leave()


def test_the_start_sharing_window_expires(hubs_of):
    hubs = hubs_of(window=2, ram_batches=2, share_start_s=0.4)
    a = hubs.subscribe(0, 0)
    hub = a.hub
    for _ in range(4):
        a.next(5), a.advance()
    time.sleep(0.5)  # the window is over: from now on what the trainers have passed is forgotten
    for _ in range(4):
        a.next(5), a.advance()
    until(lambda: hub.tail >= 6 and not hub.disk, what="the history trimmed")
    assert hub.join(0) is None  # a latecomer replays on its own
    c = hubs.subscribe(0, 0)
    assert c.hub is not hub and len(hubs.feeds) == 2
    a.leave(), c.leave()


def test_trainers_reconnecting_together_share_one_stream_whichever_comes_first(hubs_of):
    """After a server restart the two trainers are at different positions and reconnect in any order: the new stream waits a moment
    and starts at the earliest, so neither ends up with a private stream (a second teacher pass at half speed each)."""
    hubs = hubs_of(window=2, join_grace_s=0.8)
    ahead = hubs.subscribe(0, 10)  # the trainer that is further on happens to reconnect first
    hub = ahead.hub
    behind = hubs.subscribe(0, 4)
    assert behind.hub is hub and hub.tail == 4 and hub.start == 4 and len(hubs.feeds) == 2  # one stream, restarted at the earlier batch
    assert hubs.teacher.calls == 0  # nothing was produced during the grace
    until(lambda: hubs.feeds[0].closed, what="the abandoned feed stopped")
    assert read(behind.next(5)) == wire(stream.encode_batch("train", make_arrays(4), index=4))
    assert read(ahead.next(5)) == wire(stream.encode_batch("train", make_arrays(10), index=10))  # produced up to where it asks
    assert hubs.teacher.calls >= 7  # one pass: batches 4..10 once each, not a second stream
    ahead.leave(), behind.leave()


def test_a_trainer_ahead_of_the_stream_start_just_joins_it(hubs_of):
    hubs = hubs_of(window=2, join_grace_s=0.5)
    first = hubs.subscribe(0, 4)
    second = hubs.subscribe(0, 9)
    assert second.hub is first.hub and first.hub.start == 4 and len(hubs.feeds) == 1
    first.leave(), second.leave()


def test_a_trainer_that_comes_back_rejoins_the_same_stream_without_a_replay(hubs_of):
    hubs = hubs_of(window=2)
    a = hubs.subscribe(0, 0)
    hub = a.hub
    for _ in range(3):
        a.next(5), a.advance()
    a.leave()  # the connection dropped
    again = hubs.subscribe(0, a.cursor)  # reconnects at its next batch
    assert again.hub is hub and len(hubs.feeds) == 1
    assert read(again.next(5)) == wire(stream.encode_batch("train", make_arrays(3), index=3))
    again.leave()


def test_a_stream_nobody_returns_to_is_closed_and_its_files_are_deleted(hubs_of):
    hubs = hubs_of(window=2, ram_batches=2, hub_linger_s=0.5)
    slow, fast = hubs.subscribe(0, 0), hubs.subscribe(0, 0)
    hub = slow.hub
    for _ in range(8):
        fast.next(5), fast.advance()
    until(lambda: hub.disk, what="a backlog on disk")
    directory = hub.spill_dir
    slow.leave(), fast.leave()
    until(lambda: hub.closed, what="the lingering stream to close")
    until(lambda: not os.path.isdir(directory), what="its backlog deleted")
    assert hubs.status()["hubs"] == [] and hubs.feeds[0].closed


def test_a_stream_that_never_produced_is_not_kept_for_the_next_trainer(hubs_of):
    class Stuck:
        closed = False

        def get(self, timeout=None):
            time.sleep(min(timeout or 0.1, 0.1))
            raise __import__("queue").Empty

        def close(self):
            self.closed = True

    hubs = hubs_of(feed=lambda seed, start: Stuck())
    first = hubs.subscribe(0, 0)
    hub = first.hub
    first.leave()
    until(lambda: hub.closed, what="the stuck stream to close")
    assert hubs.subscribe(0, 0).hub is not hub


def test_a_finished_data_stream_is_delivered_in_full_before_its_error(hubs_of):
    hubs = hubs_of(feed=lambda seed, start: CountingFeed(start, n=6), window=2, ram_batches=2)
    slow, fast = hubs.subscribe(0, 0), hubs.subscribe(0, 0)
    hub = slow.hub
    for _ in range(6):  # the fast trainer drives the stream to its end while the slow one has read nothing
        assert fast.next(5)[0] in ("ram", "disk")
        fast.advance()
    item = fast.next(5)
    assert item[0] == "error" and "dataset exhausted" in item[1] and hub.error
    got = []
    for _ in range(6):  # the slow one still gets every batch, from RAM and disk, before the error
        got.append(read(slow.next(5)))
        slow.advance()
    assert got == [wire(stream.encode_batch("train", make_arrays(i), index=i)) for i in range(6)]
    assert slow.next(5)[0] == "error"
    directory = hub.spill_dir
    fast.leave()
    assert os.path.isdir(directory) or not hub.disk  # the last trainer has not left yet: nothing is deleted under it
    slow.leave()
    until(lambda: not os.path.isdir(directory), what="files removed after the last trainer left")
    assert hub.closed


def test_a_failing_teacher_closes_the_stream_and_is_reported_not_fatal(hubs_of):
    teacher = FakeTeacher()
    teacher.fail_from = 2
    hubs = hubs_of(teacher=teacher, window=4)
    sub = hubs.subscribe(0, 0)
    hub = sub.hub
    until(lambda: hub.closed, what="the hub to close on the teacher error")
    assert hub.head == 2 and sub.next(1) == ("closed",)  # the connection ends; the trainer reconnects and waits
    assert hubs.status()["teacher_failures"] == 1 and hubs.status()["hubs"] == []
    teacher.fail_from = None  # the fault clears
    again = hubs.subscribe(0, 0)
    assert again.hub is not hub and read(again.next(5)) == wire(stream.encode_batch("train", make_arrays(0), index=0))
    until(lambda: hubs.status()["teacher_failures"] == 0, what="the failure count reset by a good call")
    again.leave()


def test_the_throughput_log_reports_the_last_window_and_where_the_time_went(hubs_of):
    logs = []
    hubs = hubs_of(log=logs.append, log_every=3, window=100)
    sub = hubs.subscribe(0, 0)
    until(lambda: sum("tok/s" in m for m in logs) >= 2, what="two throughput lines")
    sub.leave()
    line = next(m for m in logs if "tok/s" in m)
    for field in ("teacher ", "waiting for data ", "nobody needs data ", "storing ", "clients 1", "backlog "):
        assert field in line, field
    shares = [int(x) for x in re.findall(r"(\d+)%", line)]
    assert len(shares) == 4 and all(0 <= x <= 100 for x in shares) and sum(shares) <= 105  # the four buckets partition the window


def test_a_busy_server_refuses_more_trainers_than_max_clients(hubs_of):
    hubs = hubs_of(max_clients=2)
    hubs.subscribe(0, 0), hubs.subscribe(0, 0)
    with pytest.raises(ConnectionRefusedError, match="busy"):
        hubs.subscribe(0, 0)


def test_status_reports_streams_and_how_long_the_teacher_call_in_flight_has_run(hubs_of):
    gate = threading.Event()

    class SlowTeacher(FakeTeacher):
        def __call__(self, x):
            gate.wait(10)
            return super().__call__(x)

    hubs = hubs_of(teacher=SlowTeacher())
    sub = hubs.subscribe(0, 0)
    until(lambda: hubs.status()["teacher_call_s"] > 0.3, what="a teacher call in flight")
    status = hubs.status()
    assert status["protocol"] == stream.PROTOCOL and status["hubs"][0]["clients"] == [0] and status["teacher_failures"] == 0
    gate.set()
    until(lambda: hubs.status()["teacher_call_s"] == 0.0 or sub.hub.head > 0, what="the call to finish")
    sub.leave()


# ---------------------------------------------------------------- real sockets: server loop + RemoteTargets

@pytest.fixture()
def server(tmp_path):
    started = []

    def start(feed=None, teacher=None, port=0, evals=None, **over):
        args = server_args(tmp_path, **over)
        feeds, logs = [], []

        def make_feed(seed, start_batch):
            feeds.append(feed(seed, start_batch) if callable(feed) else CountingFeed(start_batch))
            return feeds[-1]

        evals = evals or [stream.encode_batch("eval", make_arrays(1000 + i)) for i in range(2)]
        srv = socket.create_server(("127.0.0.1", port))
        thread = threading.Thread(target=accept_loop, args=(srv, teacher or FakeTeacher(), None, args, evals, logs.append, make_feed),
                                  daemon=True)
        thread.start()
        s = SimpleNamespace(addr=f"127.0.0.1:{srv.getsockname()[1]}", port=srv.getsockname()[1], srv=srv, feeds=feeds, logs=logs,
                            args=args, thread=thread)
        started.append(s)
        return s

    yield start
    for s in started:
        s.srv.close()
        s.thread.join(15)


def client(addr, **kw):
    kw = {"connect_timeout": 10, "heartbeat_s": 0.2, "silence_s": 2.0, "log": lambda *_: None, "seed": 7} | kw
    return stream.RemoteTargets(addr, **kw)


def reference(n_batches, first=0):
    return [row for i in range(first, first + n_batches) for row in rows_of(i)]


def take(ds, n):
    it = iter(ds)
    return [next(it) for _ in range(n)]


def test_two_real_trainers_get_the_same_stream_each_at_its_own_speed(server):
    s = server(window=2, ram_batches=2, share_start_s=60)
    fast, slow = client(s.addr, capacity=8), client(s.addr, capacity=8)
    try:
        got_fast = take(fast, 40)  # ten batches while the slow trainer reads nothing
        status = stream.query_status(s.addr)
        (hub,) = status["hubs"]
        assert len(s.feeds) == 1 and hub["head"] >= 10 and max(hub["clients"]) - min(hub["clients"]) >= 4  # one pass, not held back
        got_slow = take(slow, 40)
    finally:
        fast.close()
        slow.close()
    expected = reference(10)
    assert all(same(r, e) for r, e in zip(got_fast, expected, strict=True))
    assert all(same(r, e) for r, e in zip(got_slow, expected, strict=True))


def test_frames_much_bigger_than_the_socket_buffers_reach_a_slow_reader_intact(server):
    """The first real run died here: a frame that does not fit in the buffers needs many partial sends, and the sender's socket
    must tolerate the reader's timeout ticks while it waits for room."""
    teacher = FakeTeacher()
    teacher.width = 600_000  # ~20 MB per batch, like the real teacher's: more than the buffers of both ends together
    s = server(teacher=teacher, window=2, ram_batches=2)
    ds = client(s.addr, capacity=8, connect_timeout=30)
    try:
        it = iter(ds)
        got = []
        for _ in range(12):
            got.append(next(it))
            time.sleep(0.5)  # a reader slower than the sender: its receive thread stalls with a frame half-sent
        assert ds.reconnects == 0
    finally:
        ds.close()
    expected = [row for i in range(3) for row in rows_of(i, 600_000)]
    assert all(same(r, e) for r, e in zip(got, expected, strict=True))


def test_a_dropped_trainer_rejoins_its_stream_without_a_second_teacher_pass(server):
    s = server()
    ds = client(s.addr, capacity=16)
    try:
        got = take(ds, 8)
        ds.sock.shutdown(socket.SHUT_RDWR)  # the cable is pulled
        got += take(ds, 24)
        assert ds.reconnects >= 1 and len(s.feeds) == 1
    finally:
        ds.close()
    assert all(same(r, e) for r, e in zip(got, reference(8), strict=True))


def raw_trainer(addr, seed=7, start_seq=0):
    """A bare protocol client: START, read HELLO + eval + TRAIN."""
    host, port = addr.rsplit(":", 1)
    sock = socket.create_connection((host, int(port)), timeout=10)
    stream.send_frame(sock, stream.START, json.dumps({"seed": seed, "start_seq": start_seq}).encode())
    assert stream.recv_frame(sock)[0] == stream.HELLO
    while (kind := stream.recv_frame(sock)[0]) != stream.TRAIN:
        assert kind == stream.BATCH
    return sock


def test_the_server_heartbeats_a_paused_trainer_and_resumes_on_request(server):
    s = server()
    sock = raw_trainer(s.addr)
    sock.settimeout(5)
    stream.send_frame(sock, stream.PAUSE)
    kinds = []
    t0 = time.monotonic()
    while time.monotonic() - t0 < 1.5:
        kinds.append(stream.recv_frame(sock)[0])
    assert kinds.count(stream.HEARTBEAT) >= 3  # alive while idle: this is what keeps the trainer from declaring it dead
    while stream.recv_frame(sock)[0] != stream.HEARTBEAT:
        pass
    stream.send_frame(sock, stream.RESUME)
    until(lambda: stream.recv_frame(sock)[0] == stream.BATCH, what="data after RESUME")
    sock.close()


def test_a_trainer_that_goes_silent_is_dropped_and_frees_its_place(server):
    s = server(client_silence_s=1)
    sock = raw_trainer(s.addr)  # connected, then says nothing at all (no heartbeats)
    until(lambda: stream.query_status(s.addr)["hubs"] and stream.query_status(s.addr)["hubs"][0]["clients"] == [], 8, "the silent trainer dropped")
    sock.settimeout(5)
    with pytest.raises(OSError):
        while True:
            stream.recv_frame(sock)  # drains what was sent, then the connection is closed
    sock.close()
    assert any("silent" in m for m in s.logs)


def test_the_status_query_is_answered_with_trainers_connected_and_the_server_full(server):
    s = server(max_clients=1)
    first = client(s.addr)
    try:
        status = stream.query_status(s.addr)
        assert status["protocol"] == stream.PROTOCOL and status["hubs"][0]["clients"] != []
        with pytest.raises(OSError):  # a second trainer is refused (the connection ends); the status query is not
            host, port = s.addr.rsplit(":", 1)
            extra = socket.create_connection((host, int(port)), timeout=10)
            stream.send_frame(extra, stream.START, b'{"seed": 7}')
            extra.settimeout(5)
            while True:
                stream.recv_frame(extra)
        assert stream.query_status(s.addr)["protocol"] == stream.PROTOCOL
        assert take(first, 4)  # and the first trainer is unaffected
    finally:
        first.close()


def test_a_finished_data_stream_reaches_the_trainer_after_all_its_rows(server):
    s = server(feed=lambda seed, start: CountingFeed(start, n=6))
    ds = client(s.addr, capacity=64)
    try:
        got = take(ds, 24)
        with pytest.raises(RuntimeError, match="dataset exhausted"):
            next(iter(ds))
        assert ds.reconnects == 0
    finally:
        ds.close()
    assert all(same(r, e) for r, e in zip(got, reference(6), strict=True))


class SilentAfterHandshake:
    """A raw teacher that answers the handshake correctly and then hangs: no data, no heartbeats, connection open."""

    def __init__(self, serve_after):
        self.srv = socket.create_server(("127.0.0.1", 0))
        self.addr = f"127.0.0.1:{self.srv.getsockname()[1]}"
        self.connections, self.serve_after = [], serve_after
        threading.Thread(target=self._run, daemon=True).start()

    def _run(self):
        evals = [stream.encode_batch("eval", make_arrays(1000))]
        while True:
            try:
                conn, _ = self.srv.accept()
            except OSError:
                return
            self.connections.append(conn)
            _, raw = stream.recv_frame(conn)
            request = json.loads(bytes(raw))
            batch, start = B, request["start_seq"] // B
            meta = {"protocol": stream.PROTOCOL, "hidden": 6, "batch": batch, "seed": request["seed"], "start_batch": start, "drop": 0,
                    "eval_sha256": stream.frames_digest(evals), "stream_sha256": "a" * 64}
            blob = json.dumps(meta).encode()
            stream.send_frame(conn, stream.HELLO, struct.pack("!I", len(blob)), blob)
            for frame in evals:
                stream.send_frame(conn, stream.BATCH, *frame)
            stream.send_frame(conn, stream.TRAIN)
            if len(self.connections) > self.serve_after:  # from now on a healthy teacher
                for i in range(start, start + 20):
                    stream.send_frame(conn, stream.BATCH, *stream.encode_batch("train", make_arrays(i), index=i))
            # else: the first connection hangs here, silently

    def close(self):
        self.srv.close()
        for c in self.connections:
            c.close()


def test_a_trainer_treats_a_silent_teacher_as_dead_and_reconnects_at_its_next_batch():
    teacher = SilentAfterHandshake(serve_after=1)
    logs = []
    ds = client(teacher.addr, silence_s=1.0, log=logs.append)
    try:
        t0 = time.monotonic()
        got = take(ds, 8)  # the first connection delivered nothing: only the reconnect gets data
        assert ds.reconnects == 1 and 0.8 < time.monotonic() - t0 < 10 and any("silent" in str(m) for m in logs)
    finally:
        ds.close()
        teacher.close()
    assert all(same(r, e) for r, e in zip(got, reference(2), strict=True))


def test_training_waits_for_a_teacher_that_went_away_and_continues_exactly_when_it_returns(server):
    s1 = server()
    port = s1.port
    ds = client(s1.addr, capacity=16, silence_s=1.5)
    rows, errors = [], []

    def consume():
        try:
            it = iter(ds)
            while len(rows) < 400:
                rows.append(next(it))
        except Exception as e:  # noqa: BLE001
            errors.append(e)

    thread = threading.Thread(target=consume, daemon=True)
    thread.start()
    try:
        until(lambda: len(rows) >= 8, what="the first rows")
        s1.srv.close()  # the teacher process is gone: its hubs close and every connection ends
        assert thread.is_alive() and not errors
        time.sleep(2.5)  # long enough to drain its buffer and be stalled with the teacher still down
        stalled = len(rows)
        assert thread.is_alive() and not errors and stalled < 400  # it waits; it does not fail
        server(port=port)  # the supervisor brought the teacher back: a brand new process, nothing in memory
        thread.join(30)
        assert not thread.is_alive() and not errors and ds.reconnects >= 1
    finally:
        ds.close()
    assert all(same(r, e) for r, e in zip(rows, reference(100), strict=True))  # not one row lost, repeated or different


def test_the_first_connection_waits_for_a_teacher_that_is_not_up_yet(server):
    probe = socket.create_server(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    box = []
    thread = threading.Thread(target=lambda: box.append(client(f"127.0.0.1:{port}", connect_timeout=2)), daemon=True)
    thread.start()
    time.sleep(1.5)
    assert thread.is_alive()  # still retrying, not failed
    server(port=port)
    thread.join(20)
    assert box
    try:
        assert all(same(r, e) for r, e in zip(take(box[0], 8), reference(2), strict=True))
    finally:
        box[0].close()


def test_a_trainer_without_patience_still_fails_fast(server):
    probe = socket.create_server(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    with pytest.raises(OSError):
        client(f"127.0.0.1:{port}", connect_timeout=1, reconnect_s=0)
    with pytest.raises(ConnectionError, match="unreachable"):
        client(f"127.0.0.1:{port}", connect_timeout=1, reconnect_s=1.5)


# ---------------------------------------------------------------- the supervisor

CFG = teacher_supervisor.Watch(port=1, poll_s=1.0, grace_s=100.0, unreachable_s=30.0, wedge_s=60.0, teacher_failures=3)


@pytest.mark.parametrize("status,last_ok,now,expected", [
    ({"teacher_call_s": 0.0, "teacher_failures": 0}, 5.0, 10.0, None),
    ({"teacher_call_s": 61.0, "teacher_failures": 0}, 5.0, 10.0, "stuck in one teacher call"),
    ({"teacher_call_s": 59.0, "teacher_failures": 0}, 5.0, 10.0, None),
    ({"teacher_call_s": 0.0, "teacher_failures": 3}, 5.0, 10.0, "3 teacher calls failed"),
    ({"teacher_call_s": 0.0, "teacher_failures": 2}, 5.0, 10.0, None),
    (None, None, 50.0, None),  # still loading the model: inside the start-up grace
    (None, None, 101.0, "since it started"),  # never answered
    (None, 10.0, 39.0, None),  # answered before, quiet for 29 s
    (None, 10.0, 41.0, "no status answer for 31s"),  # quiet for 31 s
])
def test_judge(status, last_ok, now, expected):
    why = teacher_supervisor.judge(CFG, now, 0.0, last_ok, status)
    assert (why is None) if expected is None else (why is not None and expected in why)


class FakeProc:
    count = 0

    def __init__(self):
        FakeProc.count += 1
        self.pid, self.returncode, self.dead = 1000 + FakeProc.count, None, False

    def poll(self):
        return self.returncode


def run_fake(script, spawns_wanted):
    """supervise() over scripted status answers on a fake clock until it has started `spawns_wanted` servers.
    script(proc_number, tick) -> status dict | None | 'exit'."""
    clock = [0.0]
    procs, killed, logs = [], [], []

    def spawn():
        procs.append(FakeProc())
        clock_at_start.append(clock[0])
        return procs[-1]

    def sleep(seconds):
        clock[0] += seconds
        tick[0] += 1

    def query():
        result = script(len(procs), tick[0])
        if result == "exit":
            procs[-1].returncode = 1
            return {"teacher_call_s": 0.0, "teacher_failures": 0}
        if result is None:
            raise ConnectionRefusedError("no answer")
        return result

    clock_at_start, tick = [], [0]
    real_kill = teacher_supervisor.kill_tree
    teacher_supervisor.kill_tree = lambda proc: killed.append(proc.pid)
    try:
        n = teacher_supervisor.supervise(CFG, spawn, query, clock=lambda: clock[0], sleep=sleep, log=logs.append,
                                         stop=lambda: len(procs) >= spawns_wanted)
    finally:
        teacher_supervisor.kill_tree = real_kill
    return n, procs, killed, logs


GOOD = {"teacher_call_s": 0.0, "teacher_failures": 0}


def test_supervisor_restarts_a_server_that_exited():
    n, procs, killed, logs = run_fake(lambda p, t: "exit" if p == 1 else GOOD, spawns_wanted=2)
    assert n == 1 and len(procs) == 2 and killed[0] == procs[0].pid and any("exited with code 1" in m for m in logs)


def test_supervisor_restarts_a_server_that_stops_answering_after_the_start_up_grace_not_before():
    # silent for the whole run: it must survive the 100 s grace, then be restarted
    n, _, _, logs = run_fake(lambda p, t: None if p == 1 else GOOD, spawns_wanted=2)
    reason = next(m for m in logs if "restarting" in m)
    assert n == 1 and "since it started" in reason and "no status answer for 10" in reason  # 100+ s of poll ticks, not fewer


def test_supervisor_restarts_a_wedged_teacher_and_a_server_that_forgot_how_to_answer_later():
    def script(p, t):
        if p == 1:
            return GOOD if t < 5 else {"teacher_call_s": 90.0, "teacher_failures": 0}
        if p == 2:
            return GOOD if t < 40 else None
        return GOOD

    n, _, _, logs = run_fake(script, spawns_wanted=3)
    assert n == 2 and any("stuck in one teacher call" in m for m in logs) and any("no status answer" in m for m in logs)


def test_supervisor_backs_off_between_restarts_of_a_server_that_cannot_start():
    sleeps = []
    clock = [0.0]
    procs, killed = [], []
    real_kill = teacher_supervisor.kill_tree
    teacher_supervisor.kill_tree = lambda proc: killed.append(proc.pid)
    try:
        def spawn():
            procs.append(FakeProc())
            procs[-1].returncode = 1  # dies at once, every time
            return procs[-1]

        teacher_supervisor.supervise(CFG, spawn, lambda: GOOD, clock=lambda: clock[0], sleep=lambda s: sleeps.append(s),
                                     log=lambda _: None, stop=lambda: len(killed) >= 6)
    finally:
        teacher_supervisor.kill_tree = real_kill
    waits = [s for s in sleeps if s != CFG.poll_s]
    assert waits == [2, 4, 8, 16, 32, 60][:len(waits)] and len(waits) >= 5  # exponential, capped at 60 s, and it never gives up


STATUS_SERVER = """
import json, socket, sys
sys.path.insert(0, sys.argv[2])
import stream
srv = socket.create_server(("127.0.0.1", int(sys.argv[1])))
while True:
    conn, _ = srv.accept()
    stream.recv_frame(conn)
    stream.send_frame(conn, stream.QUERY, json.dumps({"protocol": stream.PROTOCOL, "teacher_call_s": 0.0, "teacher_failures": 0, "hubs": []}).encode())
    conn.close()
"""


def test_supervisor_restarts_a_real_process_that_gets_killed():
    probe = socket.create_server(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()
    cfg = teacher_supervisor.Watch(port=port, poll_s=0.3, grace_s=30.0, unreachable_s=5.0, wedge_s=60.0)
    nb = str(Path(__file__).resolve().parents[1] / "src")
    spawned, logs, stop = [], [], threading.Event()

    def spawn():
        spawned.append(subprocess.Popen([sys.executable, "-c", STATUS_SERVER, str(port), nb]))
        return spawned[-1]

    thread = threading.Thread(target=teacher_supervisor.supervise, args=(cfg, spawn), kwargs={"log": logs.append, "stop": stop.is_set},
                              daemon=True)
    thread.start()
    try:
        until(lambda: spawned and stream_ok(port), 30, "the first server to answer")
        first = spawned[0]
        first.kill()  # a crash
        until(lambda: len(spawned) == 2 and stream_ok(port), 30, "the supervisor to bring a new server up")
        assert spawned[1].pid != first.pid and any("exited" in m for m in logs)
    finally:
        stop.set()
        thread.join(30)
        for p in spawned:
            p.kill()
    assert not thread.is_alive()


def stream_ok(port):
    try:
        return stream.query_status(f"127.0.0.1:{port}", timeout=2)["protocol"] == stream.PROTOCOL
    except (OSError, ValueError):
        return False


def test_a_recorded_stream_replays_byte_identical_with_the_same_identity(server, tmp_path):
    """--record_dir then --replay_dir: a trainer receives the same identity, eval set and batches as from the live server."""
    evals = [stream.encode_batch("eval", make_arrays(1000 + i)) for i in range(2)]
    args = server_args(tmp_path, record_dir=str(tmp_path / "rec"), record_batches=14, seed=7)
    record(FakeTeacher(), args, evals, lambda _: None, lambda seed, start: CountingFeed(start))
    recording = Recording(tmp_path / "rec")
    assert recording.stream_identity == FakeTeacher.stream_identity and recording.fingerprint == FakeTeacher.fingerprint
    assert [wire(f) for f in recording.evals()] == [wire(f) for f in evals]
    live = server(teacher=FakeTeacher(), evals=evals)
    replay = server(teacher=recording, evals=recording.evals(), feed=lambda seed, start: ReplayFeed(recording, seed, start, B, lambda _: None))
    a, b = client(live.addr), client(replay.addr)
    try:
        got_live, got_replay = take(a, 40), take(b, 40)  # ten batches
        assert a.meta == b.meta  # protocol, hidden, decoder, eval_sha256, stream_sha256, seed, start
    finally:
        a.close()
        b.close()
    assert all(same(x, y) for x, y in zip(got_live, got_replay, strict=True))
    assert all(same(x, y) for x, y in zip(got_replay, reference(10), strict=True))


def test_a_recording_resumes_after_its_last_batch_and_refuses_another_stream(tmp_path):
    evals = [stream.encode_batch("eval", make_arrays(1000))]
    args = server_args(tmp_path, record_dir=str(tmp_path / "rec"), record_batches=6, seed=7)
    root = tmp_path / "rec"
    with pytest.raises(FeedError, match="exhausted"):  # the data feed dies after batch 2
        record(FakeTeacher(), args, evals, lambda _: None, lambda seed, start: CountingFeed(start, n=3))
    assert sorted(p.name for p in (root / "train").iterdir()) == ["0.bin", "1.bin", "2.bin"] and not (root / "COMPLETE").exists()
    starts = []
    record(FakeTeacher(), args, evals, lambda _: None, lambda seed, start: starts.append(start) or CountingFeed(start))
    assert starts == [3] and (root / "COMPLETE").exists()  # resumed after the last complete batch, not from the start
    recording = Recording(root)
    assert [wire(recording.frame(i)) for i in range(6)] == [wire(stream.encode_batch("train", make_arrays(i), index=i)) for i in range(6)]
    other = type("Other", (FakeTeacher,), {"stream_identity": "b" * 64})()
    with pytest.raises(ValueError, match="different stream"):
        record(other, args, evals, lambda _: None, lambda seed, start: CountingFeed(start))

