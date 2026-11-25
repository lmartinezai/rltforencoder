"""Export what each reported training run recorded about itself, as plain JSON, so it can be audited without the checkpoints.

    uv run python src/export_run_records.py --out results/run_records

For each checkpoint: every attribute of the pickled training arguments (`training_args.bin`, a DistillArguments, i.e. HF TrainingArguments plus
this project's own fields), the model's `config.json`, the trainer's final step and log history length, and the SHA-256 of
`model.safetensors`; and the metadata of the stored teacher targets (`STORES`). The pickle can only be loaded while the module that defined its class is importable under the name it was saved with
(`distill` / `compare_arch` on this path), which is why the records are exported once and kept in git.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

RUNS = {  # name used in the paper's tables -> checkpoint directory
    "A": "runs/arm_a_merge/checkpoint-12208", "B": "runs/arm_b_nomerge/checkpoint-12208", "C": "runs/arm_c_noloop/checkpoint-12208",
    **{f"A'_s{s}": f"runs/small/a_s{s}/checkpoint-3072" for s in (1, 2, 3)},
    **{f"B'_s{s}": f"runs/small/b_s{s}/checkpoint-3072" for s in (1, 2, 3)},
    **{f"C'_s{s}": f"runs/small_c/c_s{s}/checkpoint-3072" for s in (1, 2, 3)},
    **{f"D'_s{s}": f"runs/iso_flop/d_s{s}/checkpoint-3072" for s in (1, 2, 3)},
    **{f"E'_s{s}": f"runs/small/e_s{s}/checkpoint-3072" for s in (1, 2, 3)},
}


STORES = ("stores/fineweb_s42_50m", "stores/halting")


def plain(value):
    """JSON-safe form of an argument value (enums by value, paths and other objects as strings)."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [plain(v) for v in value]
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    return getattr(value, "value", str(value))


def redact(args):
    """The teacher server's network address is private: keep only that one was used and on which port."""
    if args.get("teacher_addr"):
        args["teacher_addr"] = "<teacher-host>:" + args["teacher_addr"].rsplit(":", 1)[-1]
    return args


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def record(checkpoint):
    import distill
    # distill.py runs pickled the class as __main__.DistillArguments (the script was run directly); compare_arch.py runs as
    # distill.DistillArguments. Expose it under both names.
    sys.modules["__main__"].DistillArguments = distill.DistillArguments
    args_file, state_file = checkpoint / "training_args.bin", checkpoint / "trainer_state.json"
    # Checkpoints copied back from the second training machine kept only the weights (and sometimes the trainer state): their records say so.
    args = torch.load(args_file, weights_only=False) if args_file.exists() else None
    state = json.loads(state_file.read_text()) if state_file.exists() else None
    return {
        "checkpoint": checkpoint.relative_to(ROOT).as_posix(),
        "args_class": f"{type(args).__module__}.{type(args).__name__}" if args is not None else None,
        # the instance's own attributes as pickled, not the current class's fields: a field added to the class after the run is absent here
        "args": redact({k: plain(v) for k, v in vars(args).items()}) if args is not None else None,
        "config": json.loads((checkpoint / "config.json").read_text()),
        "trainer_state": {"global_step": state["global_step"], "max_steps": state.get("max_steps"), "log_entries": len(state.get("log_history", []))}
        if state is not None else None,
        "model_sha256": sha256(checkpoint / "model.safetensors"),
        # runs that trained from the teacher server record the identity of the stream and of the eval set they saw
        "stream_position": json.loads((checkpoint / "stream_position.json").read_text()) if (checkpoint / "stream_position.json").exists() else None,
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="results/run_records")
    a = ap.parse_args(argv)
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    for store in STORES:  # the stored teacher targets: what each extraction recorded about itself
        rec = {part: json.loads((ROOT / store / part / "meta.json").read_text()) for part in ("train", "eval") if (ROOT / store / part / "meta.json").exists()}
        rec |= {f"{part}/{sub}": json.loads((ROOT / store / part / sub / "meta.json").read_text())
                for part in ("fineweb", "arxiv") for sub in ("train", "eval") if (ROOT / store / part / sub / "meta.json").exists()}
        (out / f"store_{Path(store).name}.json").write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
        print(store, sorted(rec))
    for name, path in RUNS.items():
        rec = record(ROOT / path)
        (out / f"{name.replace(chr(39), 'p')}.json").write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
        print(name, rec["args_class"] or "NO training_args.bin", (rec["trainer_state"] or {}).get("global_step", "NO trainer_state.json"), rec["model_sha256"][:12])


if __name__ == "__main__":
    main()
