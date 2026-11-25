"""ModernBERT-large -> Bort masked-LM distillation.

Generic machinery (accumulation, micro-batching, BF16, grad checkpointing, fused AdamW, schedule, eval,
checkpoint/resume, CLI) is `transformers.Trainer` + `HfArgumentParser`; masking is HF's MLM collator and the
student's sparse prediction is HF's own `sparse_prediction` path. Specific to this experiment: the loss
(targets.py), the recurrence-depth schedule, per-depth eval, stream-position resume, and the pipelines (exactly one source):

  remote    --teacher_addr host:port   targets stream in from teacher_server.py; no teacher, no data pipeline on this machine
  discrete  --stage extract            teacher pass over the stream -> targets on disk -> teacher unloaded
            --targets_dir DIR          student trains from that store
  joint     (neither)                  teacher + student on the GPU together (teacher targets computed live)

    uv run python src/distill.py --teacher_addr <ip>:5555 --compile_layers --per_device_train_batch_size 4 --gradient_accumulation_steps 8
    uv run python src/distill.py --stage extract --targets_dir runs/store --extract_tokens 50000000
    uv run python src/distill.py --targets_dir runs/store --max_steps 1000
    uv run python src/distill.py --max_steps 1000                        # joint, full-vocab KL, streaming FineWeb

In remote mode the server owns the data: --dataset/--holdout_pct/--shuffle_buffer/--mask_prob/--seq_len/... are the server's
flags and are rejected here when changed from their defaults. CUDA settings (allocator cap, TF32) apply in every entry point.
"""
import dataclasses
import functools
import json
import math
import random
import sys
from collections import defaultdict, deque
from contextlib import ExitStack, contextmanager, nullcontext
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path

import torch
from datasets import IterableDataset
from huggingface_hub import hf_hub_download
from safetensors import safe_open
from torch import nn
from transformers import (
    AutoConfig,
    AutoModelForMaskedLM,
    AutoTokenizer,
    HfArgumentParser,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)
from transformers.optimization import (
    _get_cosine_schedule_with_warmup_lr_lambda as _cosine_lr_lambda,  # pinned by a test
)

import pins
from arch import BortForMaskedLM, MakeBortConfig, StealEmbeddingsFromModernBert
from data import make_collator, make_eval_dataset, make_train_dataset
from store import TargetStore, collate_store, decoder_fingerprint, extract, split_targets
from stream import RemoteTargets, mask_seed
from targets import full_kl, make_targets, target_kl, teacher_hidden

TEACHER = "answerdotai/ModernBERT-large"
EVAL_MASK_POSITION = 1 << 40  # positional-mask index evaluation pins itself to (far from any training position)
MEMORY_FRACTION = 0.9  # allocator cap for training and for the benchmarks: overflow is a loud OOM, not a silent WDDM spill


@dataclass
class DistillArguments(TrainingArguments):
    """TrainingArguments + distillation settings; saved with every checkpoint. Loss weights are untuned defaults."""

    stage: str = "train"  # "train" | "extract" (discrete pipeline: write teacher targets to --targets_dir)
    teacher: str = TEACHER
    temperature: float = 2.0
    ce_weight: float = 0.6
    kd_weight: float = 0.4
    mask_prob: float = 0.3
    max_depth: int = 8  # recurrence depth is sampled from 1..max_depth
    prelude_layers: int = 2  # Bort only: blocks before the loop, in the loop and after it (2 + 4 + 2 = the 8-block Bort); the flat baseline gets the same total
    looped_layers: int = 4
    coda_layers: int = 2
    flat_layer_types: str = "bort"  # compare_arch only: "bort" = the looped model's attention layout (full at both ends of each part), "modernbert" = ModernBERT's own (full every 3rd block); deep flat models need the latter to be initialisable from the teacher
    gated_merge: bool = True  # Bort only: False builds the looped core without the gated merge (arch.py use_merge)
    merge_base: str = "state"  # Bort only: "state" adds the gated feedback to the loop state (arms A, A'); "anchor" adds it to the prelude output, RLT's form (input injection)
    init: str = "random"  # "random" | "teacher_layers": start every layer from the nearest compatible ModernBERT-large layer (see init_from_teacher)
    # teacher signal
    kd_mode: str = "full"  # "full": live full-vocab KL | "compact": through the compact-target codec (targets.py)
    topk: int = 0  # compact: top-K + tail bucket where it covers >= topp of the mass, else fp16 hidden; 0 = hidden only
    topp: float = 0.99
    targets_dir: str = ""  # train from this store (discrete pipeline) instead of running a live teacher
    teacher_addr: str = ""  # train from a teacher server (teacher_server.py) at host:port: no teacher on this machine
    stream_buffer: int = 8192  # sequences buffered from the teacher server (RAM)
    stream_high: float = 0.95  # buffer fullness at which the server's teacher is paused ...
    stream_low: float = 0.70  # ... and resumed again
    stream_timeout: int = 0  # fail the run when a healthy connection delivers nothing for this many seconds (0 = never: wait)
    stream_reconnect_s: int = -1  # seconds a lost or silent teacher is retried before the run fails (-1 = for ever, 0 = never)
    stream_heartbeat_s: float = 5.0  # the trainer pings the server this often, and listens for the server's heartbeats on this tick
    stream_silence_s: float = 30.0  # no byte (data or heartbeat) from the server for this long = the connection is dead: reconnect
    keep_fraction: float = 1.0  # extraction: keep the teacher targets of this fraction of each sequence's masked positions (the rest leave the loss)
    extract_tokens: int = 50_000_000
    extract_batch: int = 64
    compile_layers: bool = False  # torch.compile each encoder layer (~+10% update speed, ~20 s compile)
    # data (FineWeb has no validation split: hold out ~holdout_pct% of documents by id hash)
    dataset: str = "HuggingFaceFW/fineweb"
    dataset_config: str = "sample-10BT"
    dataset_revision: str = ""  # pin a dataset revision for reproducible extraction and resume
    train_split: str = "train"
    eval_split: str = "validation"  # only used when holdout_pct == 0
    streaming: bool = True
    holdout_pct: int = 1
    id_column: str = "id"
    shuffle_buffer: int = 1000
    seq_len: int = 512
    eval_samples: int = 256  # packed eval sequences
    eval_depths: list[int] = field(default_factory=lambda: [1, 4, 8])

    # Trainer defaults for this experiment
    output_dir: str = "runs/distill"
    bf16: bool = True
    optim: str = "adamw_torch_fused"
    lr_scheduler_type: str = "cosine"
    learning_rate: float = 3e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_steps: float = 100
    # Warm restart: >0 decays the cosine to zero at this step, then restarts it (no second warmup) and decays over max_steps.
    # lr_restart_step=1000 --max_steps 6200 reproduces the bort100m run, which was a 1000-step run resumed under 6200 steps.
    lr_restart_step: int = 0
    max_steps: int = 1000
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 2  # = the training micro-batch: bigger eval batches push a 12 GB card into WDDM spill
    gradient_accumulation_steps: int = 8
    gradient_checkpointing_kwargs: dict = field(default_factory=lambda: {"use_reentrant": False})
    # An IterableDataset (the teacher stream) defaults to accelerate's dispatch_batches=True, which slices every tensor of a
    # batch to the batch size: it cut the per-position kd_* targets to one row per sequence. Never dispatch.
    accelerator_config: dict = field(default_factory=lambda: {"dispatch_batches": False})
    remove_unused_columns: bool = False
    prediction_loss_only: bool = True
    logging_steps: int = 10
    logging_nan_inf_filter: bool = False  # the filter syncs the GPU after every micro-batch
    save_steps: int = 500
    eval_strategy: str = "steps"
    eval_steps: int = 500
    report_to: str = "none"


# ---------------------------------------------------------------- models

def build_teacher(args):
    """Frozen bf16 teacher on the GPU."""
    teacher = AutoModelForMaskedLM.from_pretrained(args.teacher, dtype=torch.bfloat16, **pins.kw(args.teacher))
    return teacher.to(args.device).eval().requires_grad_(False)


def teacher_checkpoint(name):
    """The teacher's weight file: inside a local directory, or fetched from the Hub."""
    path = Path(name)
    return path / "model.safetensors" if path.is_dir() else hf_hub_download(name, "model.safetensors", **pins.kw(name))


def teacher_decoder_params(args):
    """(embedding/decoder weight [V,H], decoder bias [V]) exactly as the bf16 teacher sees them, as fp32 CPU
    tensors. Read straight from the checkpoint, so the student never needs the teacher model instantiated."""
    with safe_open(teacher_checkpoint(args.teacher), "pt") as f:
        return tuple(f.get_tensor(k).bfloat16().float() for k in ("model.embeddings.tok_embeddings.weight",
                                                                  "decoder.bias"))


class _EmbeddingSource:
    """Duck-types what StealEmbeddingsFromModernBert reads from a teacher."""

    def __init__(self, weight):
        self.embedding = nn.Embedding.from_pretrained(weight)

    def get_input_embeddings(self):
        return self.embedding


def teacher_layer_map(student_types, teacher_types):
    """Which teacher layer each student layer starts from: the same attention type (full/sliding), distinct, increasing, and as close
    to evenly spaced through the teacher as that allows (least total distance from the evenly spaced positions, found exactly).
    Student layer 0 is teacher layer 0 and no other student layer may take it: it is the one ModernBERT layer without an attention
    norm, so only it is shape-compatible with the student's first layer."""
    S, T = len(student_types), len(teacher_types)
    if student_types[0] != teacher_types[0]:
        raise ValueError("the student's first layer must have the teacher's first layer's attention type")
    best = {0: (0.0, [0])}  # teacher layer taken by the layer placed so far -> (distance so far, the assignment)
    for j in range(1, S):
        ideal = j * (T - 1) / (S - 1)
        best = {t: min((cost + abs(t - ideal), path + [t]) for before, (cost, path) in best.items() if before < t)
                for t in range(1, T) if teacher_types[t] == student_types[j] and any(before < t for before in best)}
        if not best:
            raise ValueError(f"no teacher layer of type {student_types[j]} left for student layer {j}")
    return min(best.values())[1]


HEAD_KEYS = ("model.embeddings.norm.weight", "model.final_norm.weight", "head.dense.weight", "head.norm.weight", "decoder.bias")


def teacher_init_tensors(path, mapping):
    """The teacher's tensors under the student's names: layer `mapping[j]` as layer j, plus the embedding norm, final norm, head and
    decoder bias. Read straight from the checkpoint (fp32), like teacher_decoder_params."""
    wanted = {t: j for j, t in enumerate(mapping)}
    out = {}
    with safe_open(path, "pt") as f:
        for key in f.keys():  # noqa: SIM118 - safetensors' handle is not a dict
            if key.startswith("model.layers."):
                _, _, layer, rest = key.split(".", 3)
                if int(layer) in wanted:
                    out[f"model.layers.{wanted[int(layer)]}.{rest}"] = f.get_tensor(key).float()
            elif key in HEAD_KEYS:
                out[key] = f.get_tensor(key).float()
    return out


def init_from_teacher(student, args, path=None, teacher_types=None):
    """Start every layer of the student from the nearest compatible ModernBERT-large layer (embeddings already come from it), so the
    student begins as a shallow cut of the teacher instead of noise: layers by `teacher_layer_map`, and the embedding norm, final norm,
    head and decoder bias as they are. Works for Bort and for the flat baseline alike (same module names). Bort's gated merge, which
    ModernBERT has no counterpart for, starts as the identity (its output projection is zero). -> the layer mapping."""
    if teacher_types is None:
        teacher_types = AutoConfig.from_pretrained(args.teacher, **pins.kw(args.teacher)).layer_types
    mapping = teacher_layer_map(student.config.layer_types, teacher_types)
    result = student.load_state_dict(teacher_init_tensors(path or teacher_checkpoint(args.teacher), mapping), strict=False)  # shape mismatches raise
    if result.unexpected_keys:
        raise ValueError(f"teacher tensors the student has no place for: {result.unexpected_keys}")
    merge = getattr(student.model, "merge", None)
    if merge is not None:
        nn.init.zeros_(merge.proj.weight)
    return mapping


def build_student(args, embedding_weight=None):
    """Bort. The token embeddings come from the teacher (copied, frozen, tied); everything else is random, or with
    `--init teacher_layers` the nearest compatible teacher layers (init_from_teacher). `--gated_merge false` builds the looped core
    without the gated merge."""
    weight = teacher_decoder_params(args)[0] if embedding_weight is None else embedding_weight
    config = MakeBortConfig(args.teacher, args.prelude_layers, args.looped_layers, args.coda_layers)
    config.sparse_prediction = True  # HF's own path: logits only at labelled positions, [N_masked, V]
    config.use_merge = args.gated_merge
    config.merge_base = getattr(args, "merge_base", "state")
    student = BortForMaskedLM(config)
    StealEmbeddingsFromModernBert(student, _EmbeddingSource(weight), freeze=True)
    if args.init == "teacher_layers":
        print(f"[init] student layers start from teacher layers {init_from_teacher(student, args)}", flush=True)
    return student


# ---------------------------------------------------------------- loss / step

def configure_cuda():
    """Process-wide CUDA settings of every training run, applied by every entry point (train, run_extract, autotune workers)."""
    if not torch.cuda.is_available():
        return
    # WDDM silently pages VRAM overflow into system RAM (a 3-5x slowdown with no error). Cap the allocator below the
    # card's size so overflow is a loud OOM instead; the headroom is the desktop's share.
    torch.cuda.set_per_process_memory_fraction(MEMORY_FRACTION)
    # fp32 matmuls (the canonical teacher-logit decode) on TF32 tensor cores: the operands are fp16/bf16-valued, which TF32
    # holds exactly, so the result is bit-identical to the SIMT fp32 GEMM (measured KL 6e-9) at 1.8x the speed.
    torch.set_float32_matmul_precision("high")


# Flags that describe the data pipeline: in remote mode the server owns them, so a changed value here would be silently ignored.
SERVER_OWNED = ("dataset", "dataset_config", "dataset_revision", "train_split", "eval_split", "streaming", "holdout_pct", "id_column",
                "shuffle_buffer", "mask_prob", "seq_len", "eval_samples", "extract_batch")


def validate_args(args):
    """Refuse contradictory or silently-ignored configurations instead of picking one."""
    if args.init not in ("random", "teacher_layers"):
        raise ValueError(f"--init must be random or teacher_layers, got {args.init!r}")
    if args.merge_base not in ("state", "anchor"):
        raise ValueError(f"--merge_base must be state or anchor, got {args.merge_base!r}")
    if args.merge_base == "anchor" and not args.gated_merge:
        raise ValueError("--merge_base anchor needs the gated merge (--gated_merge true)")
    sources = [name for name, on in (("--teacher_addr", args.teacher_addr), ("--targets_dir", args.targets_dir.strip())) if on]
    if args.stage == "extract":
        if not args.targets_dir.strip():
            raise ValueError("--targets_dir is required for extraction")
        if args.teacher_addr:
            raise ValueError("--stage extract runs the teacher locally: --teacher_addr does not apply")
    elif len(sources) > 1:
        raise ValueError(f"choose one teacher source, got {' and '.join(sources)}")
    if args.teacher_addr and args.stage != "extract":
        defaults = {f.name: f.default for f in dataclasses.fields(DistillArguments) if f.default is not dataclasses.MISSING}
        changed = [f"--{k}" for k in SERVER_OWNED if getattr(args, k) != defaults[k]]
        if changed:
            raise ValueError(f"{', '.join(changed)} configure the data pipeline, which the teacher server owns: set them there")
    if args.lr_restart_step:
        if args.lr_scheduler_type != "cosine":
            raise ValueError("--lr_restart_step restarts a cosine schedule: set --lr_scheduler_type cosine")
        if not args.get_warmup_steps(args.max_steps) < args.lr_restart_step < args.max_steps:
            raise ValueError(f"--lr_restart_step must lie after warmup and before --max_steps, got {args.lr_restart_step}")


def restart_cosine(step, warmup, restart, total):
    """LR factor of a cosine schedule with warmup that is cut at `restart` (decayed to zero there) and then restarted without
    a second warmup, decaying again to zero at `total`. restart=0 is the plain schedule."""
    end = restart if restart and step < restart else total
    return _cosine_lr_lambda(step, num_warmup_steps=warmup, num_training_steps=end, num_cycles=0.5)


def check_server_meta(meta, args, weight, bias):
    """The student's frozen head must be exactly the teacher's decoder, or W z + b is silently wrong: same checkpoint
    name, hidden size AND decoder fingerprint (as for stored targets)."""
    want = {"teacher": args.teacher, "hidden": int(weight.shape[1]), "decoder_sha256": decoder_fingerprint(weight, bias)}
    bad = [k for k, v in want.items() if meta.get(k) != v]
    if bad:
        raise SystemExit(f"teacher server does not match this trainer's teacher on {bad}: "
                         f"server {[meta.get(k) for k in bad]} vs trainer {[want[k] for k in bad]}")


def depth_for_step(step, max_depth=8, seed=0):
    """Balanced shuffled buckets: each depth once per `max_depth` optimizer steps. A pure function of the
    step, so it is shared by all accumulation micro-batches and survives resume without sampler state."""
    perm = list(range(1, max_depth + 1))
    random.Random(seed * 1_000_003 + step // max_depth).shuffle(perm)
    return perm[step % max_depth]


def combine(ce, kl, args):
    """ce_weight*CE + kd_weight*T^2*KL (KL is taken at temperature T, hence the T^2 to restore gradient scale)."""
    return args.ce_weight * ce + args.kd_weight * args.temperature**2 * kl


def distill_step(student, teacher, bias, batch, depth, args):
    """One micro-batch -> (loss, parts). Both models get the identical corrupted batch. Call under autocast.

    batch: input_ids/attention_mask/labels, plus kd_* stored targets when training from a store (teacher=None).
    bias: the teacher's decoder bias [V]; the decoder weight is the student's tied (and frozen) embedding."""
    batch, stored = split_targets(dict(batch)) if "kd_ids" in batch else (dict(batch), None)
    T, W = args.temperature, getattr(student, "module", student).decoder.weight
    if stored is None:  # teacher first: its activations are freed before the student's graph is built
        z = teacher_hidden(teacher, batch)
        if args.kd_mode == "compact":
            stored = make_targets(z, W, bias, T, args.topk, args.topp)
    out = student(**batch, recurrence_depth=depth)  # out.loss: CE vs original ids; out.logits: [N_masked, V]
    kl = target_kl(out.logits, stored, W, bias, T).mean() if stored is not None else full_kl(out.logits, z, W, bias, T)
    return combine(out.loss, kl, args), {"ce": out.loss.detach(), "kl": kl.detach()}


# ---------------------------------------------------------------- trainer

class DistillTrainer(Trainer):
    def __init__(self, *args, teacher, teacher_bias, **kwargs):
        super().__init__(*args, **kwargs)
        # Bort's forward takes **kwargs, so Trainer would assume the loss is already normalised by
        # num_items_in_batch and skip the divide-by-accumulation. Ours is a per-micro-batch mean: keep the divide.
        self.model_accepts_loss_kwargs = False
        if isinstance(self.data_collator, PositionalCollator):
            self.add_callback(ResumeMaskPosition(self.data_collator))
        self.teacher, self.teacher_bias = teacher, teacher_bias.to(self.args.device)
        self.depth, self._stats = None, defaultdict(list)

    def create_scheduler(self, num_training_steps, optimizer=None):
        if not self.args.lr_restart_step:
            return super().create_scheduler(num_training_steps, optimizer)
        if self.lr_scheduler is None:
            warmup = self.args.get_warmup_steps(num_training_steps)
            factor = functools.partial(restart_cosine, warmup=warmup, restart=self.args.lr_restart_step, total=num_training_steps)
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer or self.optimizer, factor)
            self._created_lr_scheduler = True
        return self.lr_scheduler

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if model.training:  # one depth per optimizer step, shared by every micro-batch
            self.depth = depth_for_step(self.state.global_step, self.args.max_depth, self.args.seed)
        loss, parts = distill_step(model, self.teacher, self.teacher_bias, inputs, self.depth, self.args)
        if model.training:
            for k, v in parts.items():
                self._stats[k].append(v)
        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        with torch.no_grad(), self.compute_loss_context_manager():
            return self.compute_loss(model, inputs).detach(), None, None

    def log(self, logs, *args, **kwargs):
        if self._stats and "loss" in logs:  # training log: add component means and the latest depth
            logs |= {k: torch.stack(v).mean().item() for k, v in self._stats.items()} | {"depth": self.depth}
            if hasattr(self.train_dataset, "stats"):  # remote teacher: buffer fill and cumulative seconds the trainer starved
                logs |= self.train_dataset.stats()
            self._stats.clear()
        super().log(logs, *args, **kwargs)

    def evaluate(self, *args, **kwargs):
        out = {}
        pin = getattr(self.data_collator, "at", None)  # positional masks: every evaluation masks identically
        with torch.random.fork_rng(devices=[]):  # eval reseeds the CPU RNG; leave the training stream untouched
            for d in self.args.eval_depths:
                self.depth = d
                torch.random.default_generator.manual_seed(self.args.seed)  # identical CPU masking, without reseeding CUDA
                with pin(EVAL_MASK_POSITION) if pin else nullcontext():
                    out |= super().evaluate(*args, metric_key_prefix=f"eval_R{d}", **kwargs)
        torch.cuda.empty_cache()  # eval batches differ from training micro-batches; near 12 GB Windows spills VRAM to RAM
        return out


class PositionalCollator:
    """MLM collator whose masks are a pure function of (seed, micro-batch position), like the server's: a batch is masked the
    same however far the dataloader ran ahead before a checkpoint (it looks one micro-batch ahead), so a resumed run masks
    exactly as the uninterrupted one would. `at` pins the position (evaluation uses a fixed one and leaves training's alone)."""

    def __init__(self, tokenizer, args, seed):
        self.tokenizer, self.args, self.seed, self.position = tokenizer, args, seed, 0

    def __call__(self, rows):
        batch = make_collator(self.tokenizer, self.args, mask_seed(self.seed, self.position))(rows)
        self.position += 1
        return batch

    @contextmanager
    def at(self, position):
        saved, self.position = self.position, position
        try:
            yield
        finally:
            self.position = saved


class ResumeMaskPosition(TrainerCallback):
    """Training begins (or resumes) at micro-batch `global_step x accumulation`: that is where the masks continue."""

    def __init__(self, collator):
        self.collator = collator

    def on_train_begin(self, args, state, control, **kwargs):
        self.collator.position = state.global_step * args.gradient_accumulation_steps


class SkipFirst(torch.utils.data.IterableDataset):
    """A deterministic stream without its first `n` items, by replaying it: exactly `iterate, then drop n`. (HF's own
    IterableDataset.skip is NOT that for a shuffled stream: measured, it yields a different order.)"""

    def __init__(self, dataset, n):
        self.dataset, self.n = dataset, n

    def __iter__(self):
        it = iter(self.dataset)
        deque(islice(it, self.n), maxlen=0)
        yield from it


class StreamPosition(TrainerCallback):
    """Exact data position of a streamed run, saved next to every checkpoint (stream_position.json).

    The stream is a pure function of (seed, position): the same sequences in the same order with the same masks, however the
    run was buffered or timed. So the position is just how many sequences the trainer has consumed, which is exactly
    `start_rows + (steps since this run began) x batch x accumulation`: resuming replays the stream to that point (tokenising,
    no teacher) and continues bit-for-bit where the first run would have gone. A remote stream also reports its own count: if the
    two disagree beyond the dataloader's look-ahead (it fetches the next micro-batch early; those rows are re-delivered on resume,
    since the checkpoint records what was trained on) the data position has drifted and the run stops."""

    FILE = "stream_position.json"

    def __init__(self, dataset, seed, start_rows=0, eval_sha256=None, stream_sha256=None):
        self.dataset, self.seed, self.start_rows, self.eval_sha256 = dataset, seed, start_rows, eval_sha256
        self.stream_sha256 = stream_sha256
        self.step0 = 0

    @classmethod
    def load(cls, checkpoint):
        path = Path(checkpoint) / cls.FILE
        if not path.exists():
            raise ValueError(f"{checkpoint} has no {cls.FILE}: it was not written by a streamed run of this version, so its data "
                             "position cannot be restored exactly (restart the run, or resume from a checkpoint that has one)")
        return json.loads(path.read_text())

    def on_train_begin(self, args, state, control, **kwargs):
        self.step0 = state.global_step

    def on_save(self, args, state, control, **kwargs):
        window = args.train_batch_size * args.gradient_accumulation_steps
        rows = self.start_rows + (state.global_step - self.step0) * window  # what the optimizer steps have trained on
        actual = getattr(self.dataset, "rows_consumed", rows)  # a remote stream counts what it handed out
        if not rows <= actual <= rows + window:  # the Trainer's dataloader looks one micro-batch ahead; more or less is a bug
            raise RuntimeError(f"data position drifted at step {state.global_step}: the stream delivered {actual} sequences, "
                               f"the schedule says {rows} (+ at most {window} of look-ahead)")
        (Path(args.output_dir) / f"checkpoint-{state.global_step}" / self.FILE).write_text(json.dumps(
            {"rows_consumed": rows, "seed": self.seed, "global_step": state.global_step, "eval_sha256": self.eval_sha256,
             "stream_sha256": self.stream_sha256}))


class StopOnNonFinite(TrainerCallback):
    """A NaN/inf loss or gradient norm poisons the weights for good; fail loudly at the next log instead of training (and
    checkpointing) garbage for hours. The last finite checkpoint stays on disk."""

    def on_log(self, args, state, control, logs=None, **kwargs):
        bad = {k: v for k, v in (logs or {}).items() if k in ("loss", "grad_norm") and not math.isfinite(v)}
        if bad:
            raise FloatingPointError(f"non-finite {bad} at step {state.global_step}: resume from the last checkpoint, lower the learning rate")


def run_extract(args):
    """Discrete stage 1: stream -> corrupt -> teacher -> targets on disk. The teacher is freed on return."""
    if not args.targets_dir.strip():  # run_extract is also called directly, whatever --stage says
        raise ValueError("--targets_dir is required for extraction")
    validate_args(args)
    configure_cuda()
    root = Path(args.targets_dir)
    if not args.resume_from_checkpoint and any((root / s).exists() for s in ("train", "eval")):
        raise FileExistsError("target store already exists; choose a new --targets_dir or resume extraction")
    tok = AutoTokenizer.from_pretrained(args.teacher, **pins.kw(args.teacher))
    teacher = build_teacher(args)
    W, b = (t.to(args.device) for t in teacher_decoder_params(args))
    n = args.extract_tokens // args.seq_len
    meta = extract(teacher, W, b, make_train_dataset(tok, args), make_collator(tok, args, args.seed), root / "train",
                   args, n, args.extract_batch, log=lambda done: print(f"\rextracted {done}/{n} sequences", end=""),
                   resume=bool(args.resume_from_checkpoint))  # --resume_from_checkpoint 1: continue an interrupted run
    extract(teacher, W, b, make_eval_dataset(tok, args), make_collator(tok, args, args.seed + 1), root / "eval", args,
            args.eval_samples, args.extract_batch, resume=bool(args.resume_from_checkpoint))
    del teacher, W, b
    torch.cuda.empty_cache()
    print("\n" + json.dumps(meta))
    return meta


def train(args, callbacks=(), trainer_class=None, build=None):
    """Train a student. `build(args, embedding_weight)` makes it (default: build_student, Bort); the flat baseline and the halting
    head pass their own."""
    if args.teacher_addr and args.resume_from_checkpoint:
        StreamPosition.load(args.resume_from_checkpoint)  # fail before loading anything if the position cannot be restored
    with ExitStack() as stack:
        return _train(args, callbacks, trainer_class, build or build_student, stack)


def resume_position(args, seed):
    """The stream position of the checkpoint being resumed (None for a fresh run); a checkpoint of another data seed is refused."""
    if not args.resume_from_checkpoint:
        return None
    position = StreamPosition.load(args.resume_from_checkpoint)
    if position["seed"] != seed:
        raise ValueError(f"checkpoint was trained on data seed {position['seed']}, this run asks for {seed}")
    return position


def _train(args, callbacks, trainer_class, build, stack):
    validate_args(args)
    configure_cuda()
    if args.dataloader_num_workers and (args.teacher_addr or (not args.targets_dir and args.streaming)):
        raise ValueError("streaming training requires --dataloader_num_workers 0 for correct stream state")
    tok = AutoTokenizer.from_pretrained(args.teacher, **pins.kw(args.teacher))
    weight, bias = teacher_decoder_params(args)
    stream_position = None
    if args.teacher_addr:  # remote teacher: batches of targets stream in over the network, nothing else to load
        teacher, collator = None, collate_store
        data_seed = args.seed if args.data_seed is None else args.data_seed
        position = resume_position(args, data_seed)
        train_ds = RemoteTargets(args.teacher_addr, args.stream_buffer, args.stream_high, args.stream_low, seed=data_seed,
                                 stall_timeout=args.stream_timeout, reconnect_s=args.stream_reconnect_s,
                                 heartbeat_s=args.stream_heartbeat_s, silence_s=args.stream_silence_s,
                                 start_seq=position["rows_consumed"] if position else 0)
        stack.callback(train_ds.close)
        check_server_meta(train_ds.meta, args, weight, bias)
        if position and position["eval_sha256"] != train_ds.eval_sha256:
            raise ValueError("the server's eval set differs from the one the checkpointed run used (different data, seed or revision)")
        if position and (not position.get("stream_sha256") or position["stream_sha256"] != train_ds.meta["stream_sha256"]):
            raise ValueError("checkpoint training stream identity is missing or differs from the server")
        eval_ds = train_ds.eval
        stream_position = StreamPosition(train_ds, data_seed, train_ds.start_seq, train_ds.eval_sha256,
                                         train_ds.meta["stream_sha256"])
    elif args.targets_dir:  # discrete: stored targets, no teacher on the GPU
        teacher = None
        train_ds, eval_ds = (TargetStore(Path(args.targets_dir) / s) for s in ("train", "eval"))
        expected_head = decoder_fingerprint(weight, bias)
        for name, ds in (("train", train_ds), ("eval", eval_ds)):
            meta = ds.meta
            if meta.get("teacher") != args.teacher or meta.get("decoder_sha256") != expected_head:
                raise ValueError(f"{name} target store teacher or decoder does not match the configured teacher")
            if meta.get("sparse", 0) and meta.get("T") != args.temperature:
                raise ValueError(f"{name} target store sparse targets require extraction temperature {meta.get('T')}")
        for key in ("teacher", "decoder_sha256", "seq_len", "mask_prob", "topk", "topp", "T"):
            if train_ds.meta.get(key) != eval_ds.meta.get(key):
                raise ValueError(f"train and eval target store metadata differ for {key}")
        collator = collate_store
    else:
        teacher, collator = build_teacher(args), PositionalCollator(tok, args, args.seed)
        train_ds, eval_ds = make_train_dataset(tok, args), make_eval_dataset(tok, args)
        if isinstance(train_ds, IterableDataset):  # exact resume: replay the (deterministic) stream past what was consumed
            position = resume_position(args, args.seed)
            if position:
                train_ds = SkipFirst(train_ds, position["rows_consumed"])
            stream_position = StreamPosition(train_ds, args.seed, position["rows_consumed"] if position else 0)
    set_seed(args.seed)
    student = build(args, weight)
    if args.compile_layers:
        for layer in student.model.layers:
            layer.compile()
    if stream_position:  # the position is restored exactly above (replay), not by the Trainer's batch skipping
        args.ignore_data_skip = True  # assignment: dataclasses.replace would re-run TrainingArguments.__post_init__
    trainer = (trainer_class or DistillTrainer)(
        model=student, args=args, teacher=teacher, teacher_bias=bias, data_collator=collator,
        processing_class=tok, train_dataset=train_ds, eval_dataset=eval_ds,
        callbacks=[StopOnNonFinite(), *([stream_position] if stream_position else []), *callbacks])
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    return trainer


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)  # progress must be visible when stdout is a file or pipe
    (cli_args,) = HfArgumentParser(DistillArguments).parse_args_into_dataclasses()
    run_extract(cli_args) if cli_args.stage == "extract" else train(cli_args)
