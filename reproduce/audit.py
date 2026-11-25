"""Audit the reproduction plan (reproduce/reproduce.py) against what the reported runs recorded about themselves (results/run_records/).

    uv run python reproduce/audit.py            # writes reproduce/AUDIT.md; exits non-zero on any unexplained difference

For every training step: the planned command is parsed into the full argument object the training script would build (every field, defaults
included), and each field is compared with the arguments the run pickled into its checkpoint; the model configuration the planned arguments
build is compared with the checkpoint's config.json; the trainer's final step with the planned number of updates. For the stored teacher targets:
the planned extraction against the store's own metadata. Differences are allowed only where a reason is written down below (EXPECTED); a
field the class gained or lost after a run is listed with its value. Runs whose arguments were not copied back from the second machine are
checked against their configuration and trainer state only, and say so.
"""
import json
import shlex
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "reproduce"))

import reproduce

RECORDS = ROOT / "results" / "run_records"
# Runs whose arguments were not copied back from the second machine: the launcher they were started with (reproduce/as_run/), and the seed
# that replaces the launcher's loop variable.
LAUNCHERS = {"train-8:C": ("reproduce/as_run/arms_200m/launch_200m_c.bat", None),
             **{f"train-4:C'_s{s}": ("reproduce/as_run/small_study/launch_c.bat", s) for s in (1, 2, 3)},
             "train-4:D'_s3": ("reproduce/as_run/iso_flop/launch_220.bat", 3)}
LAUNCHER_EXPECTED = {"output_dir": "C' was launched into runs/small/ on the second machine and copied back to runs/small_c/ here"}
REDACTED = "<teacher-host>"
# Differences that are not differences in what was trained, each with its reason.
EXPECTED_FOR = {"output_dir": ({f"train-4:C'_s{s}" for s in (1, 2, 3)},
                              "C' was launched into runs/small/ on the second machine and copied back to runs/small_c/ here")}
EXPECTED = {
    "_n_gpu": "set by the Trainer from the machine at run time",
    "_setup_devices": "a cached device object, set at run time",
    "distributed_state": "set at run time",
}
# Fields the argument class gained after some runs: the value the code had in effect before the field existed, and the commit that added it.
ADDED_LATER_ASSUMED = {
    "prelude_layers": (2, "fb8bd19 made the 2 + 4 + 2 split configurable; before it the split was fixed at 2 + 4 + 2"),
    "looped_layers": (4, "fb8bd19 made the 2 + 4 + 2 split configurable; before it the split was fixed at 2 + 4 + 2"),
    "coda_layers": (2, "fb8bd19 made the 2 + 4 + 2 split configurable; before it the split was fixed at 2 + 4 + 2"),
    "keep_fraction": (1.0, "6912048 added the option; before it every masked position was kept"),
    "flat_layer_types": ("bort", "2c9ab8c added the option, used by compare_arch.py only; its default reproduces the earlier layout"),
    "merge_base": ("state", "d057ae9 added the anchor-residual form; before it the correction was always added to the loop state"),
}
# Recorded at run time by the training code, not set on the command line: (condition on the planned arguments, recorded value, reason)
RUNTIME_SET = {"ignore_data_skip": (lambda plan: bool(plan["teacher_addr"]), True,
                                    "distill.train sets it for runs that stream from the teacher server (the stream position replaces batch skipping)")}
CONFIG_IGNORED = {"architectures": "written when the model is saved", "transformers_version": "written when the model is saved",
                  "dtype": "written when the model is saved", "_name_or_path": "where the configuration was loaded from, not a model setting"}
# Config keys allowed to differ, with the reason: key -> (steps it applies to, reason)
CONFIG_EXPECTED = {"use_merge": (("train-8:C",), ("the flat_config of C's time did not yet drop this looped-model key; the flat model's class "
                                                  "(transformers' ModernBERT) never reads it"))}


def planned_args(step):
    """The argument object the planned command builds, exactly as the training script builds it."""
    from transformers import HfArgumentParser

    import distill

    argv = [x.replace(reproduce.TEACHER, f"{REDACTED}:5555") for x in step.argv]
    script = next(x for x in argv if x.startswith("src/"))
    (args,) = HfArgumentParser(distill.DistillArguments).parse_args_into_dataclasses(argv[argv.index(script) + 1:])
    if script.endswith("compare_arch.py"):  # compare_arch.main's own adjustment after parsing
        args.max_depth, args.eval_depths = 1, [1]
    return script, args


def config_class(script):
    from transformers import ModernBertConfig

    from arch import BortConfig

    return ModernBertConfig if script.endswith("compare_arch.py") else BortConfig


def launcher_args(path, seed):
    """The argument object the launcher's command line builds (the same parser and post-processing as planned_args)."""
    from transformers import HfArgumentParser

    import distill

    line = next(x for x in (ROOT / path).read_text().splitlines() if "compare_arch.py" in x)
    flags = line.split("compare_arch.py", 1)[1].split(" >", 1)[0]
    if seed is not None:
        flags = flags.replace("%%S", str(seed))
    (args,) = HfArgumentParser(distill.DistillArguments).parse_args_into_dataclasses(shlex.split(flags.replace("\\", "/")))
    args.max_depth, args.eval_depths = 1, [1]  # compare_arch.main's own adjustment after parsing
    return args


def compare_launcher(plan, launched):
    """Every field of the planned arguments against the launcher's; differences only where LAUNCHER_EXPECTED gives a reason."""
    from export_run_records import plain

    rows, problems = [], []
    p, q = {k: plain(v) for k, v in vars(plan).items()}, {k: plain(v) for k, v in vars(launched).items()}
    for key in sorted(p):
        a, b = norm(key, p[key]), norm(key, q[key])
        if a != b:
            if key in LAUNCHER_EXPECTED:
                rows.append((key, a, b, f"expected: {LAUNCHER_EXPECTED[key]}"))
            else:
                rows.append((key, a, b, "MISMATCH"))
                problems.append(key)
    return rows, problems, len(p)


def planned_config(script, args):
    import compare_arch
    from arch import MakeBortConfig

    if script.endswith("compare_arch.py"):
        config = compare_arch.flat_config(args.teacher, layers=(args.prelude_layers, args.looped_layers, args.coda_layers), layer_types=args.flat_layer_types)
    else:  # distill.build_student, without building the model
        config = MakeBortConfig(args.teacher, args.prelude_layers, args.looped_layers, args.coda_layers)
        config.sparse_prediction = True
        config.use_merge = args.gated_merge
        config.merge_base = getattr(args, "merge_base", "state")
    return json.loads(config.to_json_string(use_diff=False))


def norm(key, value):
    if isinstance(value, str) and key in ("output_dir", "targets_dir", "logging_dir"):
        return value.replace("\\", "/")
    return value


def compare_args(plan, rec, step_name=""):
    from export_run_records import plain

    planned = {k: plain(v) for k, v in vars(plan).items()}
    rows, problems = [], []
    for key in sorted(set(planned) | set(rec)):
        if key in planned and key in rec:
            p, r = norm(key, planned[key]), norm(key, rec[key])
            if p != r:
                if key in EXPECTED_FOR and step_name in EXPECTED_FOR[key][0]:
                    rows.append((key, p, r, f"expected: {EXPECTED_FOR[key][1]}"))
                elif key in EXPECTED:
                    rows.append((key, p, r, f"expected: {EXPECTED[key]}"))
                elif key in RUNTIME_SET and RUNTIME_SET[key][0](planned) and r == RUNTIME_SET[key][1]:
                    rows.append((key, p, r, f"expected: {RUNTIME_SET[key][2]}"))
                else:
                    rows.append((key, p, r, "MISMATCH"))
                    problems.append(key)
        elif key in planned:
            assumed, why = ADDED_LATER_ASSUMED.get(key, ("<no assumed value>", "no record of what was in effect"))
            ok = planned[key] == assumed
            rows.append((key, planned[key], "(absent: added to the class after this run)",
                         f"in effect then: {assumed!r} ({why})" + ("" if ok else ": MISMATCH")))
            if not ok:
                problems.append(key)
        else:
            rows.append((key, "(absent: the field was removed from the class after this run)", rec[key], "listed"))
    return rows, problems, len(set(planned) & set(rec))


def compare_config(planned, rec, step_name):
    rows, problems = [], []
    for key in sorted(set(planned) | set(rec)):
        if key in CONFIG_IGNORED:
            continue
        p, r = planned.get(key, "(absent)"), rec.get(key, "(absent)")
        if p != r:
            steps, why = CONFIG_EXPECTED.get(key, ((), ""))
            if step_name in steps:
                rows.append((key, p, r, f"expected: {why}"))
            else:
                rows.append((key, p, r, "MISMATCH"))
                problems.append(key)
    return rows, problems


def audit_store(step, rec):
    _, args = planned_args(step)
    want = {"seed": args.seed, "mask_prob": args.mask_prob, "T": args.temperature, "seq_len": args.seq_len, "teacher": args.teacher,
            "keep_fraction": args.keep_fraction, "topk": args.topk, "batch_size": args.extract_batch}
    rows, problems = [], []
    for part, n in (("train", args.extract_tokens // args.seq_len), ("eval", args.eval_samples)):
        meta = rec[part]
        for key, value in {**want, "n_seq": n}.items():
            ok = meta.get(key) == value
            rows.append((f"{part}.{key}", value, meta.get(key), "" if ok else "MISMATCH"))
            if not ok:
                problems.append(f"{part}.{key}")
        # data settings: the planned arguments' data fields against the store's data block
        for key, value in meta["data"].items():
            ok = getattr(args, key) == value
            pinned = key == "dataset_revision" and value == ""  # the extraction read the newest commit, which is the one pinned (src/pins.py)
            rows.append((f"{part}.data.{key}", getattr(args, key), value,
                         "expected: the run left the revision unpinned and read the newest commit, which is this one" if pinned else "" if ok else "MISMATCH"))
            if not ok and not pinned:
                problems.append(f"{part}.data.{key}")
    return rows, problems


def cell(v):
    s = json.dumps(v) if not isinstance(v, str) else v
    return (s if len(s) <= 70 else s[:67] + "...").replace("|", "\\|")


def main():
    out = ["# Audit of the reproduction plan against the recorded runs", "",
           "Generated by `uv run python reproduce/audit.py` from `reproduce/reproduce.py` and `results/run_records/`. Each training step's planned",
           "command is parsed into the full argument object (every field, defaults included) and compared field by field with the arguments the run",
           "stored in its checkpoint; the model configuration the plan builds is compared with the checkpoint's `config.json`; the final step with the",
           "planned number of updates. Allowed differences and their reasons are listed in `EXPECTED` in `reproduce/audit.py`.", "",
           "| step | arguments | configuration | final step | verdict |", "|---|---|---|---|---|"]
    detail, failures = [], []
    for step in reproduce.PLAN:
        if not step.record:
            continue
        rec = json.loads((RECORDS / f"{step.record}.json").read_text())
        if step.stage == "store" or step.record.startswith("store_"):
            if step.name != "store":
                out.append(f"| {step.name} | store metadata recorded; extraction arguments not compared (halting_data.py has its own) | - | - | listed |")
                continue
            rows, problems = audit_store(step, rec)
            out.append(f"| {step.name} | {len(rows)} store fields compared, {len(problems)} differ | - | - | {'FAIL' if problems else 'pass'} |")
            detail += [f"## {step.name}", "", "| field | planned | recorded | note |", "|---|---|---|---|",
                       *[f"| {k} | {cell(p)} | {cell(r)} | {n} |" for k, p, r, n in rows], ""]
            failures += [f"{step.name}: {p}" for p in problems]
            continue
        script, args = planned_args(step)
        # save_pretrained leaves out values equal to the class default: read the record through the same class to restore them
        recorded = json.loads(config_class(script).from_dict(rec["config"]).to_json_string(use_diff=False))
        cfg_rows, cfg_problems = compare_config(planned_config(script, args), recorded, step.name)
        state = rec.get("trainer_state")
        step_ok = state is None or state["global_step"] == args.max_steps
        step_cell = "not recorded" if state is None else f"{state['global_step']} of {args.max_steps}"
        if rec.get("args") is None:
            path, seed = LAUNCHERS[step.name]
            arg_rows, arg_problems, n = compare_launcher(args, launcher_args(path, seed))
            arg_rows = [(k, p_, r_, n_.replace("MISMATCH", "MISMATCH with the launcher")) for k, p_, r_, n_ in arg_rows]
            arg_cell = (f"**not recorded** (not copied back from the second machine); against the launcher `{path}`: {n} fields compared, "
                        f"{len(arg_problems)} differ")
        else:
            arg_rows, arg_problems, n = compare_args(args, rec["args"], step.name)
            noted = sum(1 for r in arg_rows if not r[3].endswith("MISMATCH"))
            arg_cell = f"{n} fields compared, {len(arg_problems)} differ" + (f"; {noted} listed differences" if noted else "")
        problems = arg_problems + [f"config.{k}" for k in cfg_problems] + ([] if step_ok else ["global_step"])
        failures += [f"{step.name}: {p}" for p in problems]
        cfg_cell = (f"{len(cfg_problems)} keys differ" if cfg_problems else "identical") + (f"; {len(cfg_rows) - len(cfg_problems)} listed difference"
                                                                                         if len(cfg_rows) > len(cfg_problems) else "")
        out.append(f"| {step.name} | {arg_cell} | {cfg_cell} | {step_cell} | {'FAIL' if problems else 'pass'} |")
        if arg_rows or cfg_rows or step.quirks:
            detail += [f"## {step.name}", ""]
            detail += [f"- {q}" for q in step.quirks] + ([""] if step.quirks else [])
            if arg_rows:
                detail += ["| argument | planned | recorded | note |", "|---|---|---|---|", *[f"| {k} | {cell(p)} | {cell(r)} | {n} |" for k, p, r, n in arg_rows], ""]
            if cfg_rows:
                detail += ["| config key | planned | recorded | note |", "|---|---|---|---|", *[f"| {k} | {cell(p)} | {cell(r)} | {n} |" for k, p, r, n in cfg_rows], ""]
    out += ["", f"**{'Unexplained differences: ' + str(len(failures)) if failures else 'No unexplained differences.'}**", ""]
    out += [f"- {f}" for f in failures] + ([""] if failures else [])
    (ROOT / "reproduce" / "AUDIT.md").write_text("\n".join(out + detail) + "\n", encoding="utf-8")
    print("\n".join(out))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
