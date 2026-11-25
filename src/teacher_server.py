"""Teacher server: runs the teacher on this machine and pushes its targets to the training machine over TCP.

A feed thread streams + tokenises + packs + masks FineWeb `prefetch` batches ahead (overlapping the GPU), the GPU stage runs
the teacher on big batches (--extract_batch sequences at once, never one at a time), a sender thread writes frames to the
socket. The trainer (stream.RemoteTargets) owns the throttle: it sends PAUSE when its buffer is more than 95% full and
RESUME at 70%; while all trainers are paused the teacher runs nothing.

Several trainers can be served at once from the same teacher pass (hub): trainers on the same seed get identical bytes, each at
its own speed. A batch is kept (RAM, then disk) until the slowest of them has been sent it. Both sides heartbeat; a status query
(`stream.query_status`) is what src/teacher_supervisor.py polls to restart a wedged or dead server.

The data position belongs to the trainer: the train stream is a pure function of (the trainer's seed, position), so batch i is
always the same sequences with the same masks, however the run was timed, buffered, restarted or reconnected. A trainer
asks for the stream from sequence `start_seq` (0 for a new run, its consumed count after a checkpoint resume or a
reconnect); the server replays the stream to that point (tokenising only, no teacher). The server keeps no position state.

    uv run python src/teacher_server.py --port 5555 [--host <tailnet ip>]
    uv run python src/distill.py --teacher_addr <this machine's ip>:5555 ...        # on the training machine
"""
import hashlib
import importlib.metadata
import inspect
import json
import os
import queue
import shutil
import socket
import struct
import sys
import threading
import time
from dataclasses import dataclass, replace
from itertools import islice

import torch
from huggingface_hub import HfApi
from transformers import AutoTokenizer, HfArgumentParser

import data
import pins
import stream
from data import make_collator, make_eval_dataset, make_train_dataset
from distill import DistillArguments, build_teacher, teacher_decoder_params
from store import batch_arrays, decoder_fingerprint, teacher_fingerprint
from stream import mask_seed
from targets import make_targets, teacher_hidden

MIN_HANDSHAKE_RATE = 2e6  # bytes/s the eval set must move at (16 Mbit/s) or the handshake is abandoned


@dataclass
class ServeArguments(DistillArguments):
    host: str = "0.0.0.0"
    port: int = 5555
    prefetch: int = 4  # prepared (tokenised + masked) batches waiting for the GPU
    window: int = 32  # the teacher works while some trainer has fewer than this many batches queued ahead of it (the surplus waits on disk)
    join_grace_s: float = 20.0  # a brand-new stream waits this long for other trainers before it produces, and starts at the earliest of them
    ram_batches: int = 16  # newest encoded batches kept in RAM; a trainer further behind than that is read from disk
    spill_dir: str = "spill"  # backlog for trainers at different speeds: batches wait here until every trainer has them
    spill_gb: float = 40.0  # backlog size at which the teacher idles (the only thing that slows a fast trainer)
    share_start_s: float = 600.0  # a new stream keeps everything from its start this long, so a second trainer started a few minutes later joins it
    hub_linger_s: int = 120  # a stream nobody is subscribed to is kept this long (a reconnecting trainer rejoins it, no replay)
    max_clients: int = 8
    heartbeat_s: float = 3.0  # an idle sender (paused trainer, seeking feed) sends a heartbeat this often
    client_silence_s: int = 60  # a trainer that sends nothing (its heartbeat is every 5 s) for this long is dropped
    compile_teacher: bool = True  # torch.compile the encoder: measured +35% tok/s on an RTX 3060 Ti (fused elementwise ops)
    log_every: int = 20  # batches
    max_retry_s: int = 60  # longest backoff between retries of a failing data stream
    handshake_s: int = 30  # a client must send START within this long or is dropped (a silent connection must not hold the server)
    max_feed_failures: int = 5  # consecutive failures of a stream that never delivered before reporting a terminal feed error
    feed_stop_s: int = 60  # how long a closing connection waits for its data feed to stop (a dataset call may block ~45 s)
    # One GPU for teacher and trainer, taking turns: --record_dir writes what this server would stream (seed --seed, batches 0 ..
    # --record_batches - 1) to disk and exits; --replay_dir serves that recording with no teacher and no dataset, with the same identity
    # and byte-identical frames. Neither changes the stream: the frames come from the same feed, teacher and encoder as a live server's.
    record_dir: str = ""
    record_batches: int = 0
    replay_dir: str = ""


class Teacher:
    """Teacher: corrupted batch -> arrays of one wire frame (BF16 or FP16 hidden states)."""

    def __init__(self, args):
        self.args, self.device = args, args.device
        self.model = build_teacher(args)
        self.model_fingerprint = teacher_fingerprint(self.model)
        if args.compile_teacher:
            self.model.model = torch.compile(self.model.model)  # the encoder only: the masked-position gather stays eager
        W, b = teacher_decoder_params(args)
        self.fingerprint = decoder_fingerprint(W, b)  # sent in the hello: the trainer's frozen head must be this exact decoder
        self.W, self.b = W.to(self.device), b.to(self.device)

    @property
    def hidden(self):
        return int(self.W.shape[1])

    @torch.no_grad()
    def __call__(self, x):
        x = {k: v.to(self.device, non_blocking=True) for k, v in x.items()}
        tg = make_targets(teacher_hidden(self.model, x), self.W, self.b, self.args.temperature, 0, 1.0)
        return batch_arrays(x["input_ids"], x["labels"], tg)


def stream_identity(args, tok, teacher):
    """Fingerprint every server-side input to sequence order, corruption and teacher targets."""
    if not args.dataset_revision or not hasattr(tok, "backend_tokenizer"):
        raise ValueError("a resolved dataset revision and a fast tokenizer are required for exact remote resume")
    keys = ("dataset", "dataset_config", "dataset_revision", "train_split", "streaming", "holdout_pct", "id_column",
            "shuffle_buffer", "mask_prob", "seq_len", "extract_batch", "teacher", "compile_teacher")
    payload = {k: getattr(args, k) for k in keys}
    payload.update(tokenizer=tok.backend_tokenizer.to_str(),
                   token_ids={k: getattr(tok, k, None) for k in ("cls_token_id", "sep_token_id", "mask_token_id", "pad_token_id", "all_special_ids")},
                   teacher_state=teacher.model_fingerprint,
                   versions={k: importlib.metadata.version(k) for k in ("torch", "datasets", "transformers", "tokenizers")},
                   pipeline="\n".join((inspect.getsource(data), inspect.getsource(mask_seed),
                                        inspect.getsource(batch_stream), inspect.getsource(teacher_hidden),
                                        inspect.getsource(make_targets))))
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()

def eval_frames(teacher, tok, args):
    """The fixed held-out set, built once per server run: [encoded batch] (same masks every time)."""
    collate, rows = make_collator(tok, args, args.seed + 1), list(make_eval_dataset(tok, args))
    return [stream.encode_batch("eval", teacher(collate(rows[i:i + args.extract_batch])))
            for i in range(0, len(rows), args.extract_batch)]


def batch_stream(make_dataset, make_collate, batch, start_batch=0, stop=None):
    """Endless (index, corrupted batch). Batch i is always the same `batch` sequences of the same deterministic dataset with a
    mask drawn from `make_collate(i)`: independent of timing, restarts and where the stream was entered. Entering at
    `start_batch` replays the dataset up to there (tokenising, no teacher)."""
    it = iter(make_dataset())
    for _ in range(start_batch * batch):
        if stop is not None and stop.is_set():
            return
        if next(it, None) is None:
            raise EOFError("the training stream ended before the requested position")
    index = start_batch
    while len(rows := list(islice(it, batch))) == batch:
        yield index, make_collate(index)(rows)
        index += 1
    raise EOFError("the training stream ended")


class Feed:
    """One trainer's corrupted batches [extract_batch, seq_len] from batch `start_batch` on, produced by a thread `prefetch`
    batches ahead so tokenising and masking overlap the GPU stage. Prefetch only ever delays data: the order is fixed by
    `batch_stream`. A failing stream (dataset host hiccup) is reopened at the next batch with exponential backoff."""

    def __init__(self, tok, args, seed, start_batch, log):
        self.args, self.log, self.index = args, log, start_batch
        run_args = replace(args, seed=seed)
        self.make_dataset = lambda: make_train_dataset(tok, run_args)
        self.make_collate = lambda i: make_collator(tok, run_args, mask_seed(seed, i))
        self.out, self.stop = queue.Queue(args.prefetch), threading.Event()
        self.error, self.progressed = None, False  # progressed: a failing dataset host is retried for ever once data has flowed
        self.thread = threading.Thread(target=self._run, daemon=True, name="feed")
        self.thread.start()

    def _run(self):
        wait, failures = 1, 0
        while not self.stop.is_set():
            try:
                for index, batch in batch_stream(self.make_dataset, self.make_collate, self.args.extract_batch, self.index, self.stop):
                    while not self.stop.is_set():
                        try:
                            self.out.put((index, batch), timeout=0.5)
                            break
                        except queue.Full:
                            pass
                    if self.stop.is_set():
                        return
                    self.index, wait, failures, self.progressed = index + 1, 1, 0, True
            except Exception as e:  # noqa: BLE001 - dataset host hiccups come in many exception types; retry them all
                failures += 1
                if isinstance(e, EOFError) or (failures >= self.args.max_feed_failures and not self.progressed):
                    self.error = f"data feed failed at batch {self.index}: {e!r}"
                    return
                self.log(f"data stream failed ({e!r}); retrying batch {self.index} in {wait}s")
                self.stop.wait(wait)
                wait = min(wait * 2, self.args.max_retry_s)

    def get(self, timeout=None):
        try:
            return self.out.get(timeout=timeout)
        except queue.Empty:
            if self.error:
                raise FeedError(self.error) from None
            raise

    def close(self):
        """Stop the feed. A thread blocked inside the dataset (the shuffle-buffer fill alone is ~45 s) exits as soon as that
        call returns, so a slow stop is logged and the server carries on: a trainer that leaves early must not kill it."""
        self.stop.set()
        self.thread.join(timeout=self.args.feed_stop_s)
        if self.thread.is_alive():
            self.log(f"data feed still blocked in the dataset after {self.args.feed_stop_s}s; abandoning it (it exits when the call returns)")


class FeedError(RuntimeError):
    pass


class Subscription:
    """One trainer's place in a Hub: `cursor` is the next batch index it has not been sent."""

    def __init__(self, hub, cursor):
        self.hub, self.cursor = hub, cursor

    def next(self, timeout):
        """-> ("ram", index, parts) | ("disk", index, path) | ("error", message) | ("closed",) | None after `timeout` seconds."""
        return self.hub.next_for(self, timeout)

    def advance(self):
        self.hub.advance(self)

    def leave(self):
        self.hub.leave(self)


class Hub:
    """One data stream (a seed, entered at some batch) that every trainer subscribed to it is fed from.

    The producer thread runs the teacher once per batch and keeps the encoded frame until the slowest subscriber has been sent it:
    the newest `ram_batches` frames in RAM, older ones that somebody still needs on disk (`spill_dir`), and each file is deleted as
    soon as every subscriber is past it. A fast trainer therefore runs at its own speed and a slow one reads the backlog at its
    own, and both get exactly the same bytes. The producer works while some subscriber has fewer than `window` batches queued
    ahead of it, and idles when all do or when the backlog on disk reaches `spill_gb` (the only way a fast trainer is slowed).
    For the first `share_start_s` seconds nothing is forgotten, so a second trainer that starts a little later still joins at batch 0."""

    def __init__(self, feed, encode, args, log, seed, start_batch, spill_dir, restart_feed=None):
        self.feed, self.encode, self.args, self.log, self.seed = feed, encode, args, log, seed
        self.restart_feed = restart_feed  # start -> a Feed from that batch: lets a new stream start earlier for a trainer that comes second
        self.go_at = time.monotonic() + args.join_grace_s  # nothing is produced before this: trainers reconnecting together all get in
        self.ram, self.disk, self.disk_bytes = {}, {}, 0  # index -> parts | index -> (path, nbytes)
        self.start = self.tail = self.head = start_batch  # [tail, head) is available to a new subscriber
        self.created = time.monotonic()
        self.subs, self.cv = [], threading.Condition()
        self.error, self.closed, self.idle_since = None, False, time.monotonic()
        self.spill_dir = spill_dir
        self.thread = threading.Thread(target=self._run, daemon=True, name=f"hub-{seed}")
        self.thread.start()

    # ---- subscribers
    def join(self, start_batch):
        with self.cv:
            if self.closed or self.error:
                return None
            if start_batch < self.tail and self.head == self.start and self.restart_feed:
                self._rewind(start_batch)  # nothing has been produced: begin at the earlier position so this trainer can share the stream
            ahead = self.args.window * (4 if self.head == self.start else 1)  # an unproduced stream will simply produce up to a trainer that is further on
            if not self.tail <= start_batch <= self.head + ahead:
                return None
            sub = Subscription(self, start_batch)
            self.subs.append(sub)
            self.cv.notify_all()
            return sub

    def leave(self, sub):
        with self.cv:
            if sub in self.subs:
                self.subs.remove(sub)
            if not self.subs:
                self.idle_since = time.monotonic()
                if self.error:  # the stream ended or failed and the last trainer has now taken what was left
                    self.closed = True
                    self._cleanup()
                elif self.head == self.start:  # nothing was ever produced (a feed stuck in the dataset): nothing to keep for a returning trainer
                    self.closed = True
            self._trim()
            self.cv.notify_all()

    def advance(self, sub):
        with self.cv:
            sub.cursor += 1
            self._trim()
            self.cv.notify_all()

    def next_for(self, sub, timeout):
        end = time.monotonic() + timeout
        with self.cv:
            while True:
                i = sub.cursor
                if i in self.ram:
                    return "ram", i, self.ram[i]
                if i in self.disk:
                    return "disk", i, self.disk[i][0]
                if self.error:
                    return "error", self.error
                if self.closed:
                    return ("closed",)
                remaining = end - time.monotonic()
                if remaining <= 0:
                    return None
                self.cv.wait(remaining)

    def _rewind(self, start):
        """Move an unproduced stream's start back to `start`: a fresh feed from there (cheaper than the old replay), the old one stopped."""
        old, self.feed = self.feed, self.restart_feed(start)
        self.start = self.tail = self.head = start
        self.log(f"stream of seed {self.seed} now starts at batch {start}: an earlier trainer joined before it produced anything")
        threading.Thread(target=old.close, daemon=True).start()

    # ---- storage (call with the lock held)
    def _trim(self):
        """Forget every frame all subscribers are past. With nobody subscribed everything is kept (a trainer may come back)."""
        if not self.subs or self._holding():
            return
        low = min(s.cursor for s in self.subs)
        for i in [i for i in self.ram if i < low]:
            del self.ram[i]
        for i in [i for i in self.disk if i < low]:
            self._drop_file(i)
        self.tail = max(self.tail, min(low, self.head))

    def _holding(self):
        """True while the stream still keeps its whole history for a trainer that has not joined yet (`share_start_s`)."""
        return time.monotonic() - self.created < self.args.share_start_s

    def _drop_file(self, i):
        path, nbytes = self.disk.pop(i)
        self.disk_bytes -= nbytes
        try:
            os.remove(path)
        except OSError:
            pass

    def _evict(self):
        """Keep `ram_batches` frames in RAM: an older frame goes to disk if anybody can still need it, else it is dropped."""
        while len(self.ram) > self.args.ram_batches:
            i = min(self.ram)
            parts = self.ram.pop(i)
            if self.subs and not self._holding() and min(s.cursor for s in self.subs) > i:
                continue
            os.makedirs(self.spill_dir, exist_ok=True)
            path = os.path.join(self.spill_dir, f"{i}.bin")
            nbytes = 0
            with open(path, "wb") as f:
                for p in parts:
                    if memoryview(p).nbytes:  # empty arrays (no sparse rows) carry no bytes and cannot be cast
                        m = memoryview(p).cast("B")
                        f.write(m)
                        nbytes += m.nbytes
            self.disk[i], self.disk_bytes = (path, nbytes), self.disk_bytes + nbytes

    def _wants(self):
        return bool(self.subs) and time.monotonic() >= self.go_at and self.disk_bytes < self.args.spill_gb * 1e9 and \
            any(self.head - s.cursor < self.args.window for s in self.subs)

    # ---- producer
    def _run(self):
        n, sent, nbytes = 0, 0, 0
        spent = {"busy": 0.0, "data": 0.0, "idle": 0.0, "store": 0.0}  # seconds in the teacher / waiting for data / nobody needs data / storing
        mark = (time.monotonic(), 0, 0, dict(spent))  # the previous log line: rates are per window, not since the stream began
        try:
            while True:
                t_idle = time.monotonic()
                with self.cv:
                    while not self.closed and not self._wants():
                        if not self.subs and time.monotonic() - self.idle_since > self.args.hub_linger_s:
                            self.closed = True
                        else:
                            self.cv.wait(0.5)
                    if self.closed:
                        return
                spent["idle"] += time.monotonic() - t_idle
                t_data = time.monotonic()
                feed = self.feed
                try:
                    index, x = feed.get(timeout=0.5)
                except queue.Empty:
                    spent["data"] += time.monotonic() - t_data
                    continue
                t1 = time.monotonic()
                spent["data"] += t1 - t_data
                if feed is not self.feed:  # the stream was moved back while we waited: this batch is from the abandoned feed
                    continue
                frame = self.encode(x, index)
                t_store = time.monotonic()
                spent["busy"] += t_store - t1
                with self.cv:
                    if self.closed:
                        return
                    if feed is not self.feed:
                        continue
                    if index != self.head:
                        raise FeedError(f"feed delivered batch {index}, expected {self.head}")
                    self.ram[index], self.head = frame, index + 1
                    self._evict()
                    self._trim()
                    self.cv.notify_all()
                spent["store"] += time.monotonic() - t_store
                n, sent = n + 1, sent + len(x["input_ids"])
                nbytes += sum(memoryview(p).nbytes for p in frame)
                if n % self.args.log_every == 0:
                    now = time.monotonic()
                    dt = max(now - mark[0], 1e-6)  # the clock can tick 0 between two early batches
                    seqs, window = sent - mark[1], {k: spent[k] - mark[3][k] for k in spent}
                    self.log(f"seed {self.seed}: {sent} sequences  last {seqs} at {seqs * self.args.seq_len / dt:,.0f} tok/s  "
                             f"{(nbytes - mark[2]) / dt / 1e6:.1f} MB/s ({nbytes / sent / 1e3:.0f} KB/seq)  "
                             f"teacher {window['busy'] / dt:.0%}  waiting for data {window['data'] / dt:.0%}  "
                             f"nobody needs data {window['idle'] / dt:.0%}  storing {window['store'] / dt:.0%}  "
                             f"clients {len(self.subs)}  backlog {self.head - min((s.cursor for s in self.subs), default=self.head)} batches "
                             f"({self.disk_bytes / 1e9:.1f} GB on disk)")
                    mark = (now, sent, nbytes, dict(spent))
        except FeedError as e:
            with self.cv:
                self.error = str(e)
        except Exception as e:  # noqa: BLE001 - a failing teacher drops the connections (trainers reconnect and wait), it never kills the server
            self.log(f"hub for seed {self.seed} failed: {e!r}")
            with self.cv:
                self.closed = True
        finally:
            with self.cv:
                self.cv.notify_all()
            self.feed.close()
            if not self.error:  # on a feed error the trainers still drain the backlog; the last one to leave cleans up
                self._cleanup()

    def _cleanup(self):
        with self.cv:
            for i in list(self.disk):
                self._drop_file(i)
            self.ram.clear()
            self.closed = True
        try:
            os.rmdir(self.spill_dir)
        except OSError:
            pass

    def close(self):
        with self.cv:
            self.closed = True
            self.cv.notify_all()
        self.thread.join(timeout=self.args.feed_stop_s + 5)

    def describe(self):
        with self.cv:
            return {"seed": self.seed, "tail": self.tail, "head": self.head, "ram": len(self.ram), "disk_gb": self.disk_bytes / 1e9,
                    "clients": [s.cursor for s in self.subs], "error": self.error, "closed": self.closed}


def wipe_spill(spill_dir):
    """Delete the backlog directories this server creates (`hub<N>`) and nothing else that may live in `spill_dir`."""
    if os.path.isdir(spill_dir):
        for name in os.listdir(spill_dir):
            if name.startswith("hub") and name[3:].isdigit():
                shutil.rmtree(os.path.join(spill_dir, name), ignore_errors=True)
    try:
        os.rmdir(spill_dir)
    except OSError:
        pass


class Hubs:
    """Every live stream of a server. Trainers on the same seed share a Hub (one teacher pass, identical bytes) when the stream still
    holds their start position; anything else gets its own. All Hubs share the one GPU."""

    def __init__(self, teacher, args, log, make_feed):
        self.teacher, self.args, self.log, self.make_feed = teacher, args, log, make_feed
        self.lock, self.gpu, self.hubs, self.t0 = threading.Lock(), threading.Lock(), [], time.monotonic()
        self.call_started, self.failures, self.next_id = None, 0, 0
        wipe_spill(args.spill_dir)  # a previous run's backlog is not this run's

    def encode(self, x, index):
        """The one place the teacher runs: serialised across hubs, timed for the status report (a hung call is what a supervisor looks for)."""
        if isinstance(self.teacher, Recording):  # --replay_dir: the frame the live teacher produced for this batch
            return self.teacher.frame(index)
        with self.gpu:
            self.call_started = time.monotonic()
            try:
                frame = stream.encode_batch("train", self.teacher(x), index=index)
                self.failures = 0
                return frame
            except Exception:
                self.failures += 1
                raise
            finally:
                self.call_started = None

    def subscribe(self, seed, start_batch):
        with self.lock:
            self.hubs = [h for h in self.hubs if not h.closed and not h.error]
            if sum(len(h.subs) for h in self.hubs) >= self.args.max_clients:
                raise ConnectionRefusedError(f"server busy: {self.args.max_clients} trainers already connected")
            for hub in self.hubs:
                if hub.seed == seed and (sub := hub.join(start_batch)):
                    self.log(f"trainer joins the stream of seed {seed} at batch {start_batch} (stream is at {hub.head}, "
                             f"{len(hub.subs)} trainers)")
                    return sub
            self.next_id += 1
            hub = Hub(self.make_feed(seed, start_batch), self.encode, self.args, self.log, seed, start_batch,
                      os.path.join(self.args.spill_dir, f"hub{self.next_id}"), lambda start: self.make_feed(seed, start))
            self.hubs.append(hub)
            return hub.join(start_batch)

    def status(self):
        call = self.call_started
        return {"protocol": stream.PROTOCOL, "uptime_s": time.monotonic() - self.t0,
                "teacher_call_s": 0.0 if call is None else time.monotonic() - call, "teacher_failures": self.failures,
                "hubs": [d for d in (h.describe() for h in list(self.hubs)) if not d["closed"]]}

    def close(self):
        with self.lock:
            hubs, self.hubs = self.hubs, []
        for h in hubs:
            h.close()
        wipe_spill(self.args.spill_dir)


def serve(conn, teacher, hubs, args, evals, log):
    kind, raw = stream.recv_frame(conn, time.monotonic() + args.handshake_s, stream.CONTROL_MAX)
    if kind == stream.QUERY:  # a supervisor's health check: answer and hang up
        stream.send_frame(conn, stream.QUERY, json.dumps(hubs.status()).encode())
        return
    if kind != stream.START:
        raise stream.ProtocolError(f"expected START, got {kind!r}")
    request = json.loads(bytes(raw))
    seed, start_seq = request["seed"], request.get("start_seq", 0)
    if not (isinstance(seed, int) and isinstance(start_seq, int) and seed >= 0 and start_seq >= 0):
        raise stream.ProtocolError(f"bad START request {request}")
    start_batch, drop = divmod(start_seq, args.extract_batch)  # the trainer discards the first `drop` rows of batch `start_batch`
    meta = json.dumps({"protocol": stream.PROTOCOL, "hidden": teacher.hidden, "decoder_sha256": teacher.fingerprint,
                       "seq_len": args.seq_len, "T": args.temperature, "teacher": args.teacher, "batch": args.extract_batch,
                       "seed": seed, "start_batch": start_batch, "drop": drop, "eval_sha256": stream.frames_digest(evals),
                       "stream_sha256": teacher.stream_identity}).encode()
    eval_bytes = sum(memoryview(m).nbytes for frame in evals for m in frame)
    deadline = time.monotonic() + args.handshake_s + eval_bytes / MIN_HANDSHAKE_RATE  # a big eval set needs time, a stalled peer still times out
    def send_handshake(kind, *parts):
        parts = [m.cast("B") for m in map(memoryview, parts) if m.nbytes]
        for part in [memoryview(stream.HEAD.pack(kind, sum(map(len, parts)))), *parts]:
            while part:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("teacher handshake send timed out")
                conn.settimeout(remaining)
                sent = conn.send(part)
                if not sent:
                    raise ConnectionError("peer closed during handshake")
                part = part[sent:]
    send_handshake(stream.HELLO, struct.pack("!I", len(meta)), meta)
    for frame in evals:
        send_handshake(stream.BATCH, *frame)
    send_handshake(stream.TRAIN)
    log(f"trainer asked for seed {seed} from sequence {start_seq} (batch {start_batch}, drop {drop})")

    sub = hubs.subscribe(seed, start_batch)  # may refuse (server full): nothing to undo yet
    conn.settimeout(args.heartbeat_s)  # one socket for both threads: the reader ticks on this, the sender's send_frame(stop=) rides it
    run, stop = threading.Event(), threading.Event()  # run: the trainer wants more (cleared by PAUSE)
    run.set()

    def control():  # trainer -> server: PAUSE / RESUME / BYE / heartbeats; a trainer silent for client_silence_s is dropped
        def idle(silent_s):
            if silent_s > args.client_silence_s:
                raise TimeoutError(f"trainer silent for {silent_s:.0f}s")
        try:
            while True:
                kind, _ = stream.recv_frame(conn, limit=stream.CONTROL_MAX, idle=idle)
                if kind == stream.PAUSE:
                    run.clear()
                    log("PAUSE: trainer buffer is full")
                elif kind == stream.RESUME:
                    run.set()
                    log("RESUME")
                elif kind == stream.BYE:
                    break
        except OSError as e:  # includes the silence limit, keepalive timeouts and ProtocolError
            if not isinstance(e, ConnectionError):
                log(f"dropping trainer: {e!r}")
        stop.set(), run.set()

    def sender():
        last = time.monotonic()
        try:
            while not stop.is_set():
                item = sub.next(args.heartbeat_s) if run.is_set() else None
                if item is None:  # idle, paused, or the stream is seeking: say we are alive
                    if not run.is_set():
                        run.wait(args.heartbeat_s)
                    if time.monotonic() - last >= args.heartbeat_s and not stop.is_set():
                        stream.send_frame(conn, stream.HEARTBEAT, stop=stop)
                        last = time.monotonic()
                    continue
                if item[0] == "error":
                    stream.send_frame(conn, stream.ERROR, item[1].encode(), stop=stop)
                    break
                if item[0] == "closed":
                    break
                if item[0] == "disk":
                    with open(item[2], "rb") as f:
                        stream.send_frame(conn, stream.BATCH, f.read(), stop=stop)
                else:
                    stream.send_frame(conn, stream.BATCH, *item[2], stop=stop)
                sub.advance()
                last = time.monotonic()
        except Exception as e:  # noqa: BLE001 - whatever kills the sender must end the connection
            if not isinstance(e, OSError):
                log(f"sender failed: {e!r}")
        stop.set()
        run.set()

    threads = [threading.Thread(target=f, daemon=True) for f in (control, sender)]
    [t.start() for t in threads]
    log(f"streaming (batches of {args.extract_batch} sequences)")
    try:
        stop.wait()
    finally:
        stop.set()
        sub.leave()
        [t.join(timeout=5) for t in threads]


def accept_loop(srv, teacher, tok, args, evals, log, make_feed=None):
    """Serve trainers (any number at once, up to --max_clients) until the listening socket is closed. One bad or dying client never
    takes the server down."""
    make_feed = make_feed or (lambda seed, start: Feed(tok, args, seed, start, log))
    hubs, handlers = Hubs(teacher, args, log, make_feed), threading.BoundedSemaphore(4 * args.max_clients)

    def handle(conn, peer):
        try:
            serve(stream.tune(conn), teacher, hubs, args, evals, log)
        except Exception as e:  # noqa: BLE001
            log(f"connection from {peer[0]} ended: {e!r}")
        finally:
            conn.close()
            handlers.release()

    try:
        while True:
            try:
                conn, peer = srv.accept()
            except OSError:  # the listening socket was closed
                return
            if not handlers.acquire(blocking=False):
                conn.close()
                continue
            threading.Thread(target=handle, args=(conn, peer), daemon=True, name=f"serve-{peer[0]}").start()
    finally:
        hubs.close()


def write_frame(path, parts):
    """One frame's buffers, concatenated (the spill format: exactly the payload the sender sends), written atomically."""
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as f:
        for part in parts:
            if memoryview(part).nbytes:  # empty arrays (no sparse rows) carry no bytes and cannot be cast
                f.write(memoryview(part).cast("B"))
    os.replace(tmp, path)


def identity_of(teacher, args, evals):
    return {"protocol": stream.PROTOCOL, "hidden": teacher.hidden, "decoder_sha256": teacher.fingerprint, "seq_len": args.seq_len,
            "T": args.temperature, "teacher": args.teacher, "batch": args.extract_batch, "seed": args.seed,
            "eval_sha256": stream.frames_digest(evals), "stream_sha256": teacher.stream_identity}


def record(teacher, args, evals, log, make_feed):
    """--record_dir: write the frames this server would send for seed `args.seed`, batches 0 .. record_batches - 1, its eval frames and its
    identity, then return. An interrupted recording resumes after its last complete batch (the identity must be unchanged)."""
    if args.record_batches <= 0:
        raise ValueError("--record_dir needs --record_batches (the training batches of 64 sequences to record)")
    root = os.path.abspath(args.record_dir)
    os.makedirs(os.path.join(root, "train"), exist_ok=True)
    os.makedirs(os.path.join(root, "eval"), exist_ok=True)
    identity = {**identity_of(teacher, args, evals), "batches": args.record_batches}
    path = os.path.join(root, "identity.json")
    if os.path.exists(path) and json.load(open(path)) != identity:  # noqa: SIM115
        raise ValueError(f"{root} holds a recording of a different stream; choose another --record_dir")
    with open(path, "w") as f:
        json.dump(identity, f, indent=1)
    for j, frame in enumerate(evals):
        write_frame(os.path.join(root, "eval", f"{j}.bin"), frame)
    start = 0
    while os.path.exists(os.path.join(root, "train", f"{start}.bin")):
        start += 1
    feed = make_feed(args.seed, start)
    try:
        for i in range(start, args.record_batches):
            while True:
                try:
                    index, x = feed.get(timeout=5)
                    break
                except queue.Empty:
                    continue
            if index != i:
                raise FeedError(f"feed delivered batch {index}, expected {i}")
            write_frame(os.path.join(root, "train", f"{i}.bin"), stream.encode_batch("train", teacher(x), index=i))
            if (i + 1) % args.log_every == 0:
                log(f"recorded {i + 1}/{args.record_batches} batches")
    finally:
        feed.close()
    with open(os.path.join(root, "COMPLETE"), "w") as f:
        f.write(f"{args.record_batches}\n")
    log(f"recording complete: {args.record_batches} batches in {root}")


class Recording:
    """--replay_dir: stands in for the Teacher with a recording's identity and frames (no model, no GPU, no dataset)."""

    def __init__(self, root):
        self.root = os.path.abspath(root)
        if not os.path.exists(os.path.join(self.root, "COMPLETE")):
            raise ValueError(f"{self.root} is not a complete recording (run the server with --record_dir first)")
        self.meta = json.load(open(os.path.join(self.root, "identity.json")))  # noqa: SIM115
        self.hidden, self.fingerprint, self.stream_identity = self.meta["hidden"], self.meta["decoder_sha256"], self.meta["stream_sha256"]

    def frame(self, index):
        with open(os.path.join(self.root, "train", f"{index}.bin"), "rb") as f:
            return [f.read()]

    def evals(self):
        names = sorted(os.listdir(os.path.join(self.root, "eval")), key=lambda n: int(n.split(".")[0]))
        out = []
        for name in names:
            with open(os.path.join(self.root, "eval", name), "rb") as f:
                out.append([f.read()])
        if stream.frames_digest(out) != self.meta["eval_sha256"]:
            raise ValueError("the recording's eval frames do not match its identity")
        return out


class ReplayFeed:
    """Batch indices of a recording for its one seed; past its end it idles (a trainer stops before needing more, given the margin)."""

    def __init__(self, recording, seed, start_batch, batch, log):
        if seed != recording.meta["seed"]:
            raise ConnectionRefusedError(f"this recording holds seed {recording.meta['seed']}, not {seed}")
        self.recording, self.index, self.batch, self.log, self.warned = recording, start_batch, batch, log, False

    def get(self, timeout=None):
        if self.index >= self.recording.meta["batches"]:
            if not self.warned:
                self.log(f"a trainer asked for batch {self.index}, past the {self.recording.meta['batches']} recorded: record more")
                self.warned = True
            time.sleep(timeout or 0.5)
            raise queue.Empty
        index, self.index = self.index, self.index + 1
        return index, {"input_ids": range(self.batch)}

    def close(self):
        pass


def main():
    sys.stdout.reconfigure(line_buffering=True)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")  # one Rust thread pool per process is plenty; more only contend
    (args,) = HfArgumentParser(ServeArguments).parse_args_into_dataclasses()
    t0 = time.monotonic()
    log = lambda s: print(f"[{time.monotonic() - t0:7.0f}s] {s}", flush=True)
    if args.replay_dir:
        teacher, tok = Recording(args.replay_dir), None
        evals = teacher.evals()
        make_feed = lambda seed, start: ReplayFeed(teacher, seed, start, args.extract_batch, log)
        log(f"replaying {teacher.meta['batches']} recorded batches of seed {teacher.meta['seed']} (stream {teacher.stream_identity[:12]})")
    else:
        args.dataset_revision = HfApi().dataset_info(args.dataset, revision=args.dataset_revision or None).sha
        tok = AutoTokenizer.from_pretrained(args.teacher, **pins.kw(args.teacher))
        teacher = Teacher(args)
        teacher.stream_identity = stream_identity(args, tok, teacher)
        evals = eval_frames(teacher, tok, args)
        make_feed = None
        if args.record_dir:
            return record(teacher, args, evals, log, lambda seed, start: Feed(tok, args, seed, start, log))
    log(f"teacher ready, {len(evals)} eval batches cached; listening on {args.host}:{args.port}")
    with socket.create_server((args.host, args.port)) as srv:
        accept_loop(srv, teacher, tok, args, evals, log, make_feed)


if __name__ == "__main__":
    main()
