"""Teacher-over-the-network: wire protocol, receive buffer and the trainer-side dataset.

The teacher runs on another machine (teacher_server.py) and pushes *batches* of sequences (many at once, never one at a
time): each frame is one teacher batch (corrupted ids + BF16 or FP16 hidden states of its masked positions, the
arrays of store.batch_arrays). The trainer keeps them in a RAM buffer; training never waits on the network as long as
the buffer is non-empty, and the buffer throttles the teacher with hysteresis: PAUSE when it is more than `high` full,
RESUME once it has drained to `low` (defaults 95% / 70%). While paused the teacher machine runs nothing.

Frames: 1 byte kind + 8 byte big-endian length + payload.
  client -> server   S start {seed, start_seq} | P pause | R resume | X bye | K heartbeat | Q status query
  server -> client   H hello (json: stream meta) | B batch (split "eval"|"train", train frames carry their index) | T end of eval
                     K heartbeat | Q status (json) | E error

Liveness runs both ways at the application level: each side sends K whenever it has been idle for `heartbeat_s` (the server's
sender also does while paused or while its data feed seeks), and a side that hears nothing for its silence limit treats the
connection as dead: the trainer reconnects (retrying for as long as `reconnect_s` allows, forever by default, while training
stalls on an empty buffer), the server drops the trainer. TCP keepalive stays underneath as the last layer.

Reproducibility: the train stream is a pure function of (seed, position). Batch i holds the same sequences with the same masks
however the run was timed, buffered, interrupted or reconnected. The trainer owns the position: START asks for sequence
`start_seq` (a checkpoint resume) and a reconnect asks for the next batch it has not received; every train frame carries its
batch index and a gap or a repeat is an error. Buffers (server prefetch, the trainer's receive buffer) only ever delay data.
"""
import hashlib
import json
import socket
import struct
import threading
import time
from collections import deque

import numpy as np
import torch

from store import check_arrays, rows_from_arrays

HEAD = struct.Struct("!cQ")
START, PAUSE, RESUME, BYE, QUERY = b"S", b"P", b"R", b"X", b"Q"
HELLO, BATCH, TRAIN, HEARTBEAT, ERROR = b"H", b"B", b"T", b"K", b"E"
SOCK_BUF = 8 << 20
MAX_FRAME = 1 << 30  # a peer claiming more is broken or hostile: refuse instead of allocating it
CONTROL_MAX = 64 << 10  # trainer -> server frames (START/PAUSE/RESUME/BYE) are tiny: a bigger one is not from our trainer
def mask_seed(seed, index):
    """Seed of batch `index`'s masking: a pure function of the run seed and the batch's place in the stream."""
    return (seed * 1_000_003 + index) % (1 << 62)


PROTOCOL = 6  # 5: BF16 hidden states as uint16 bit patterns; 6: heartbeats both ways, status query, several trainers per stream
IDENTITY = ("protocol", "teacher", "hidden", "decoder_sha256", "batch", "seed", "stream_sha256")
ALIGN = 8  # every array in a frame starts on an 8-byte boundary (frombuffer views are then aligned)


class ProtocolError(ConnectionError):
    """The peer sent something that is not this protocol."""


def tune(sock, keepalive_s=10):
    """Big buffers, no Nagle, and keepalive so a peer that vanished (power loss, cable) raises instead of hanging forever."""
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    for opt in (socket.SO_SNDBUF, socket.SO_RCVBUF):
        sock.setsockopt(socket.SOL_SOCKET, opt, SOCK_BUF)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
    if hasattr(sock, "ioctl"):  # Windows: (on, idle ms, interval ms); gives up after 10 probes
        sock.ioctl(socket.SIO_KEEPALIVE_VALS, (1, keepalive_s * 1000, keepalive_s * 300))
    else:
        for opt, v in (("TCP_KEEPIDLE", keepalive_s), ("TCP_KEEPINTVL", max(1, keepalive_s // 3)), ("TCP_KEEPCNT", 5)):
            sock.setsockopt(socket.IPPROTO_TCP, getattr(socket, opt), v)
    return sock


def _send_all(sock, data, stop):
    """sendall that survives the socket's timeout: a timeout is only a tick to look at `stop` (and not the end of the send)."""
    if stop is None:
        sock.sendall(data)
        return
    view = memoryview(data)
    while view:
        try:
            view = view[sock.send(view):]
        except TimeoutError:
            if stop.is_set():
                raise ConnectionAbortedError("connection closing") from None


def send_frame(sock, kind, *parts, stop=None):
    """One frame. With `stop` (an Event) the socket may carry a timeout (the server's reader ticks on it) without a long send
    failing: the timeout only wakes the loop to check `stop`. Never `dup()` a socket to get a different timeout: blocking mode is
    a property of the kernel socket, shared by the duplicates, and the last settimeout wins."""
    parts = [m.cast("B") for m in map(memoryview, parts) if m.nbytes]  # empty arrays (no sparse rows) send nothing
    _send_all(sock, HEAD.pack(kind, sum(len(p) for p in parts)), stop)
    for p in parts:
        _send_all(sock, p, stop)


def _read(sock, n, deadline=None, idle=None):
    """Read exactly n bytes. With `idle` (and no deadline) the socket's own timeout is a tick: every tick without a byte calls
    idle(seconds since the last byte of this read), which may raise to give up; a partial read is never lost."""
    buf = bytearray(n)
    view, got = memoryview(buf), 0
    last = time.monotonic()
    while got < n:
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("teacher handshake timed out")
            sock.settimeout(remaining)
        try:
            k = sock.recv_into(view[got:], min(n - got, 1 << 20))
        except TimeoutError:
            if deadline is not None or idle is None:
                raise
            idle(time.monotonic() - last)
            continue
        if not k:
            raise ConnectionError("peer closed the connection")
        got += k
        last = time.monotonic()
    return buf


def recv_frame(sock, deadline=None, limit=MAX_FRAME, idle=None):
    kind, n = HEAD.unpack(_read(sock, HEAD.size, deadline, idle))
    if n > limit:
        raise ProtocolError(f"frame of {n} bytes (limit {limit})")
    return kind, _read(sock, n, deadline, idle)


def query_status(addr, timeout=10.0):
    """Ask a teacher server how it is (what its supervisor polls): -> the status dict, or an OSError if it does not answer."""
    host, port = addr.rsplit(":", 1)
    deadline = time.monotonic() + timeout
    with socket.create_connection((host.strip("[]"), int(port)), timeout=timeout) as sock:
        send_frame(sock, QUERY)
        kind, raw = recv_frame(sock, deadline, limit=CONTROL_MAX)
    if kind != QUERY:
        raise ProtocolError(f"expected a status reply, got {kind!r}")
    return json.loads(bytes(raw))


def _pad(n):
    return (-n) % ALIGN


def frames_digest(frames):
    """sha256 over the payload bytes of frames (lists of buffers as encode_batch returns them): the trainer hashes what it received."""
    h = hashlib.sha256()
    for parts in frames:
        for p in parts:
            m = memoryview(p)
            if m.nbytes:
                h.update(m.cast("B"))
    return h.hexdigest()


def encode_batch(split, arrays, index=None):
    """-> buffers of one B frame: u32 header length, json header (split, batch index, name/dtype/shape per array), raw array bytes.
    The header is space-padded and each array zero-padded so every array starts 8-byte aligned."""
    arrays = {k: np.ascontiguousarray(v) for k, v in arrays.items()}
    head = json.dumps({"split": split, "index": index,
                       "arrays": {k: [v.dtype.str, list(v.shape)] for k, v in arrays.items()}}).encode()
    head += b" " * _pad(4 + len(head))
    parts = [struct.pack("!I", len(head)), head]
    for v in arrays.values():
        parts += [v, bytes(_pad(v.nbytes))]
    return parts


def decode_batch(buf):
    """Inverse of encode_batch -> (split, {name: ndarray}); the arrays are views into `buf` (no copy)."""
    split, _, arrays = decode_frame(buf)
    return split, arrays


def decode_frame(buf):
    """-> (split, batch index or None, {name: ndarray})."""
    (n,) = struct.unpack_from("!I", buf, 0)
    head, pos, out = json.loads(bytes(buf[4:4 + n])), 4 + n, {}
    for k, (dtype, shape) in head["arrays"].items():
        count = int(np.prod(shape))
        nbytes = count * np.dtype(dtype).itemsize
        out[k] = np.frombuffer(buf, dtype=np.dtype(dtype), count=count, offset=pos).reshape(shape)
        pos += nbytes + _pad(nbytes)
    return head["split"], head.get("index"), out


class ReceiveBuffer:
    """FIFO of per-sequence rows between the receiver thread and the trainer, with high/low watermarks.

    `put` past `high` x capacity fires on_pause once; `get` draining to `low` x capacity fires on_resume once. The
    The queue is bounded; the receiver blocks when it is full and TCP applies backpressure to in-flight frames.
    A frame being decoded may hold additional rows outside the queue."""

    def __init__(self, capacity, high=0.95, low=0.70, on_pause=lambda: None, on_resume=lambda: None, stall_timeout=600):
        assert 0 < low < high <= 1 and capacity > 0
        self.capacity, self.high, self.low, self.stall_timeout = capacity, high, low, stall_timeout
        self.on_pause, self.on_resume = on_pause, on_resume
        self.rows, self.cv, self.paused, self.error = deque(), threading.Condition(), False, None
        self.stall_s, self.paused_s, self._paused_at, self.received, self.consumed = 0.0, 0.0, 0.0, 0, 0
        self.activity = time.monotonic()  # last sign of life from the server (data or heartbeat)

    @property
    def fill(self):
        return len(self.rows) / self.capacity

    def put(self, rows):
        with self.cv:
            offset = 0
            while offset < len(rows):
                while len(self.rows) >= self.capacity and self.error is None:
                    self.cv.wait()
                if self.error is not None:
                    raise RuntimeError("teacher buffer closed") from self.error
                end = min(len(rows), offset + self.capacity - len(self.rows))
                self.rows.extend(rows[offset:end])
                self.activity = time.monotonic()
                self.received += end - offset
                offset = end
                if not self.paused and self.fill >= self.high:
                    self.paused, self._paused_at = True, time.monotonic()
                    self.on_pause()
                self.cv.notify_all()

    def touch(self):
        """The server is alive (a heartbeat while it seeks to the requested position): not a stall."""
        with self.cv:
            self.activity = time.monotonic()

    def fail(self, exc):
        with self.cv:
            self.error = exc
            self.cv.notify_all()

    def get(self):
        with self.cv:
            if not self.rows:
                t0 = time.monotonic()
                while not self.rows:
                    if self.error:
                        raise RuntimeError(f"teacher stream failed: {self.error!r}") from self.error
                    if self.stall_timeout > 0 and time.monotonic() - max(t0, self.activity) > self.stall_timeout:
                        raise TimeoutError(f"no teacher data for {self.stall_timeout}s (server up but silent, or stalled)")
                    self.cv.wait(timeout=1.0)
                self.stall_s += time.monotonic() - t0
            row = self.rows.popleft()
            self.consumed += 1
            self.cv.notify_all()
            if self.paused and self.fill <= self.low:
                self.paused = False
                self.paused_s += time.monotonic() - self._paused_at
                self.on_resume()
            return row

    def stats(self):
        return {"stream_fill": self.fill, "stream_stall_s": self.stall_s, "stream_paused_s": self.paused_s}


class EvalRows(torch.utils.data.Dataset):
    def __init__(self, rows):
        self.rows = rows

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        return self.rows[i]


class RemoteTargets(torch.utils.data.IterableDataset):
    """Training stream from a teacher server. Connecting also fetches the fixed eval set (`.eval`); a daemon thread receives
    teacher batches into the ReceiveBuffer while the trainer consumes from it. Items are TargetStore-format rows, so
    `collate_store` is the collator. Use with num_workers=0 (one process owns the socket).

    Exact position: the stream is a pure function of (`seed`, position) and the trainer owns the position. `start_seq` is the
    number of sequences already consumed by an earlier run (a checkpoint resume); `rows_consumed` is the absolute count now.
    Every train frame must carry the next batch index: a gap or a repeat raises. A teacher that goes away is waited for: a dropped
    or silent connection (`silence_s` without a byte, heartbeats count) is retried with exponential backoff for `reconnect_s`
    seconds (negative = forever, the default; 0 = never retry) while training keeps consuming the buffer and then stalls on it.
    It resumes at the next batch not yet received, so nothing is lost or repeated, and the teacher identity (IDENTITY) and eval
    set must be unchanged. The first connection is retried the same way. A server that sends something that is not the protocol
    is never retried. `stall_timeout` (<= 0: never) additionally fails the run when a *healthy* connection delivers nothing for
    that long."""

    def __init__(self, addr, capacity=8192, high=0.95, low=0.70, seed=0, connect_timeout=60, stall_timeout=0, reconnect_s=-1,
                 start_seq=0, heartbeat_s=5.0, silence_s=30.0, log=print):
        host, port = addr.rsplit(":", 1)
        self.host, self.port, self.seed, self.start_seq = host.strip("[]"), int(port), seed, start_seq  # "[::1]:5555" -> ::1
        self.connect_timeout, self.reconnect_s, self.reconnects = connect_timeout, reconnect_s, 0
        self.heartbeat_s, self.silence_s, self.log = heartbeat_s, silence_s, log
        self.lock = threading.Lock()  # PAUSE (receiver thread) and RESUME (trainer thread) share the send side
        self.closing = threading.Event()
        self.buffer = ReceiveBuffer(capacity, high, low, lambda: self._ctl(PAUSE), lambda: self._ctl(RESUME), stall_timeout)
        self.sock, self.meta, rows = self._connect_retrying(start_seq, "connect")
        self.batch, self.next_batch, self.drop = self.meta["batch"], self.meta["start_batch"], self.meta["drop"]
        self.eval, self.eval_sha256 = EvalRows(rows), self.meta["eval_sha256"]
        self.thread = threading.Thread(target=self._receive, daemon=True, name="teacher-receiver")
        self.thread.start()
        threading.Thread(target=self._heartbeat, daemon=True, name="teacher-heartbeat").start()

    @property
    def rows_consumed(self):
        """Absolute number of training sequences the trainer has taken from this stream (including earlier runs)."""
        return self.start_seq + self.buffer.consumed

    def _connect(self, start_seq):
        """One connection + handshake asking for the stream from `start_seq` -> (socket, meta, eval rows). Closes the socket on failure."""
        sock = tune(socket.create_connection((self.host, self.port), timeout=self.connect_timeout))
        try:
            deadline = time.monotonic() + self.connect_timeout
            send_frame(sock, START, json.dumps({"seed": self.seed, "start_seq": start_seq}).encode())
            kind, raw = recv_frame(sock, deadline)
            if kind != HELLO:
                raise ProtocolError(f"expected HELLO, got {kind!r}")
            (n,) = struct.unpack_from("!I", raw, 0)
            if n != len(raw) - 4:
                raise ProtocolError("invalid HELLO length")
            meta = json.loads(bytes(raw[4:]))
            if meta.get("protocol") != PROTOCOL:
                raise ProtocolError(f"teacher server speaks protocol {meta.get('protocol')!r}, this trainer {PROTOCOL}: "
                                    "update both machines to the same code")
            if not isinstance(meta.get("stream_sha256"), str) or len(meta["stream_sha256"]) != 64:
                raise ProtocolError("teacher did not provide a training stream identity")
            if (meta.get("seed"), meta["start_batch"] * meta["batch"] + meta["drop"]) != (self.seed, start_seq):
                raise ProtocolError(f"server answered seed/position {meta.get('seed')}/{meta['start_batch'] * meta['batch'] + meta['drop']}"
                                    f" to a request for {self.seed}/{start_seq}")
            rows, digest = [], hashlib.sha256()
            while True:
                kind, buf = recv_frame(sock, deadline)
                if kind == TRAIN:
                    if buf:
                        raise ProtocolError("TRAIN marker must be empty")
                    break
                digest.update(buf)
                rows += self._batch(kind, buf, "eval")
            if digest.hexdigest() != meta["eval_sha256"]:
                raise ProtocolError("the eval set that arrived is not the one the server announced")
            sock.settimeout(self.heartbeat_s)  # from here on the timeout is the receiver's idle tick (_read with `idle`)
            return sock, meta, rows
        except BaseException:
            sock.close()
            raise

    @staticmethod
    def _frame(kind, buf, expected):
        """-> (batch index, rows) of one BATCH frame, checked against the split the caller expects."""
        if kind != BATCH:
            raise ProtocolError(f"expected {expected} BATCH, got {kind!r}")
        split, index, arrays = decode_frame(buf)
        if split != expected:
            raise ProtocolError(f"expected {expected} batch, got {split!r}")
        try:
            check_arrays(arrays)
        except ValueError as e:
            raise ProtocolError(f"inconsistent {expected} batch: {e}") from e
        return index, rows_from_arrays(arrays)

    @staticmethod
    def _batch(kind, buf, expected):
        return RemoteTargets._frame(kind, buf, expected)[1]

    def _ctl(self, kind, payload=b""):
        with self.lock:
            try:
                send_frame(self.sock, kind, payload)
            except OSError:
                pass  # the receiver thread reports (or repairs) the broken connection

    def _connect_retrying(self, start_seq, what, cause=None):
        """`_connect`, retried with backoff for `reconnect_s` seconds (negative: forever) while the teacher is unreachable. Anything
        that is not a network failure (a protocol or identity problem) is raised at once: waiting would not fix it."""
        started, wait, attempts = time.monotonic(), 1.0, 0
        while True:
            try:
                return self._connect(start_seq)
            except ProtocolError:
                raise
            except OSError as e:
                attempts += 1
                waited = time.monotonic() - started
                if self.reconnect_s == 0 or self.closing.is_set():
                    raise  # not retrying: the failure as it was (a handshake TimeoutError stays a TimeoutError)
                if self.reconnect_s > 0 and waited + wait > self.reconnect_s:
                    raise ConnectionError(f"teacher unreachable for {waited:.0f}s ({what})" + (f" after {cause!r}" if cause else "")) from e
                if attempts == 1 or attempts % 10 == 0:
                    self.log(f"[teacher stream] teacher unreachable ({e!r}): training waits for it, retrying (attempt {attempts}, {waited:.0f}s)")
                self.closing.wait(wait)
                wait = min(wait * 2, 10.0)

    def _idle(self, silent_s):
        """The receive socket ticked every `heartbeat_s` without a byte: give up on it after `silence_s` (server hung or cable pulled)."""
        if silent_s > self.silence_s:
            raise ConnectionError(f"teacher silent for {silent_s:.0f}s (no data, no heartbeat)")

    def _heartbeat(self):
        """Tell the server we are alive whatever else is flowing, so it can drop a trainer that vanished without a FIN."""
        while not self.closing.wait(self.heartbeat_s):
            self._ctl(HEARTBEAT)

    def _receive(self):
        try:
            while True:
                try:
                    while True:
                        kind, buf = recv_frame(self.sock, idle=self._idle)
                        if kind == HEARTBEAT:
                            self.buffer.touch()
                            continue
                        if kind == ERROR:
                            raise ProtocolError(f"teacher data feed failed: {bytes(buf).decode('utf-8', errors='replace')}")
                        index, rows = self._frame(kind, buf, "train")
                        if index != self.next_batch:
                            raise ProtocolError(f"stream position broken: expected batch {self.next_batch}, got {index}")
                        self.next_batch += 1
                        rows, self.drop = rows[self.drop:], 0  # a resume offset inside the first batch: skip the consumed rows
                        self.buffer.put(rows)
                except ProtocolError:
                    raise
                except OSError as e:  # connection loss, keepalive timeout, reset
                    if self.closing.is_set():
                        return
                    self._reconnect(e)
        except Exception as e:  # noqa: BLE001 - whatever kills this thread must reach the trainer, not vanish
            self.buffer.fail(e)

    def _reconnect(self, cause):
        if self.reconnect_s == 0 or self.closing.is_set():
            raise cause
        self.log(f"[teacher stream] connection lost ({cause!r}); training continues from its buffer and waits for the teacher")
        sock, meta, _ = self._connect_retrying(self.next_batch * self.batch, "reconnect", cause)  # exactly where the stream broke off
        if {k: meta.get(k) for k in IDENTITY} != {k: self.meta.get(k) for k in IDENTITY} or \
                meta["eval_sha256"] != self.eval_sha256:
            sock.close()
            raise ProtocolError(f"teacher changed across a reconnect: {self.meta} -> {meta}")
        with self.buffer.cv, self.lock:  # same lock order as PAUSE/RESUME: the swap and the PAUSE state are atomic
            old, self.sock = self.sock, sock
            if self.buffer.paused:  # a fresh server connection starts out streaming
                send_frame(sock, PAUSE)
        old.close()
        self.reconnects += 1
        self.log(f"[teacher stream] reconnected at batch {self.next_batch}")

    def __iter__(self):
        while True:
            yield self.buffer.get()

    def stats(self):
        return self.buffer.stats() | {"stream_reconnects": self.reconnects}

    def close(self):
        self.closing.set()
        self._ctl(BYE)
        self.buffer.fail(ConnectionError("teacher stream closed"))
        try:
            self.sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        self.sock.close()
        self.thread.join(timeout=5)
