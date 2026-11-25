"""Reproduce every number in the paper: data, training, evaluation, reports, figures, and the number check, in order.

    uv run python reproduce/reproduce.py --list                       # the plan: every step, its stage, and its recorded quirks
    uv run python reproduce/reproduce.py --dry-run --stage eval       # the exact commands of some stages, nothing run
    uv run python reproduce/reproduce.py --stage store train-4        # run stages (a finished step is skipped on a re-run)
    uv run python reproduce/reproduce.py --stage train-8 --teacher-host 192.0.2.10
    uv run python reproduce/audit.py                                  # check this plan against what the reported runs recorded

The training commands are the ones the reported runs were started with (reproduce/as_run/ holds those launchers as they were); reproduce/audit.py
checks every argument of every training step against the arguments each trained model stored with itself (results/run_records/), field by
field, and writes reproduce/AUDIT.md. The runs whose arguments were not copied back from the second machine are checked there against the
launchers in reproduce/as_run/ instead.

Outputs: trained models go to runs/ and stored targets to stores/, as in the reported runs (a training step is finished when its final
checkpoint exists; a run directory without one is refused rather than overwritten). Everything evaluated, every report, and the figures go to
reproduced/ (git-ignored), never to the committed results/: the evaluation scripts skip models already present in their output files, so
writing next to the committed results would reproduce nothing. The last step runs paper/check_numbers.py on reproduced/: it passes only if the
reproduction gives the numbers the paper states.

Hardware: the eight-block runs need a teacher server (src/teacher_server.py, about 16 GB of GPU memory for ModernBERT-large) reachable from the
training machine; the four-block runs need only the stored teacher targets of the `store` stage. Every training step fits a 12 GB GPU at its
recorded micro-batch split; the 8 GB card used for some runs needed 4 x 8 instead of 8 x 4 (same 32-sequence update, same data and depth order).
The hardware stage measures the machine it runs on: its latencies are comparable to Table 4 only on an RTX 5070 and an Intel Core Ultra 7 265K.

Wall-clock times of the reported runs on the RTX 5070, from their log timestamps: store extraction 20 min; one four-block run 26 to 28 min
(15 runs); one eight-block run 3.2 h with the merge (3 runs); the held-out loss of the eight-block models 8 min; the main latency sweep 29 min. The
other steps were not timed end to end.
"""
import argparse
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = ["uv", "run", "python"]
TEACHER = "{teacher}:5555"  # filled from --teacher-host for the steps that stream from the teacher server
R = "reproduced"  # every evaluation output, report, and figure of a reproduction


@dataclass
class Step:
    name: str
    stage: str
    argv: list[str]
    done: str  # a step is finished when this path exists (training: the final checkpoint's weights; others: their main output in reproduced/)
    record: str = ""  # results/run_records/<record>.json: what the reported run stored about itself (training steps)
    quirks: list[str] = field(default_factory=list)
    background: list[str] = field(default_factory=list)  # a process kept running for the duration of the step (the power logger)
    env: dict[str, str] = field(default_factory=dict)  # extra environment variables (the report steps: where the reproduced results are)


def a(s):
    return shlex.split(s)


# ---------------------------------------------------------------------------------------------------------------- the plan
MODELS = ["modernbert-large=hf:answerdotai/ModernBERT-large", "roberta-base=hf:FacebookAI/roberta-base", "deberta-base=hf:microsoft/deberta-base",
          "distilbert-cased=hf:distilbert/distilbert-base-cased",
          *[f"{x}_s{s}={kind}:{d}/{x}_s{s}/checkpoint-3072" for s in (1, 2, 3)
            for x, kind, d in (("a", "bort", "runs/small"), ("b", "bort", "runs/small"), ("c", "flat", "runs/small_c"), ("d", "flat", "runs/iso_flop"),
                               ("e", "bort", "runs/small"))],
          "A8=bort:runs/arm_a_merge/checkpoint-12208", "B8=bort:runs/arm_b_nomerge/checkpoint-12208", "C8=flat:runs/arm_c_noloop/checkpoint-12208"]
POWER = [*PY, "src/hw_power.py", "--out", f"{R}/hardware"]
COMMON_8 = (f"--teacher_addr {TEACHER} --compile_layers --stream_buffer 2048 --init teacher_layers --max_steps 12208 --warmup_steps 100 "
            "--save_steps 1000 --eval_steps 500 --save_total_limit 2 --disable_tqdm true --report_to none")
COMMON_4 = ("--targets_dir stores/fineweb_s42_50m --compile_layers --init teacher_layers --prelude_layers 1 --looped_layers 2 --coda_layers 1 "
            "--max_steps 3072 --warmup_steps 100 --save_steps 1024 --save_total_limit 1 --disable_tqdm true --report_to none")
SPLIT = {"5070": "--per_device_train_batch_size 8 --gradient_accumulation_steps 4", "3060Ti": "--per_device_train_batch_size 4 --gradient_accumulation_steps 8"}
# One machine, one GPU: the teacher records the eight-block stream to disk first (6,104 batches of 64 are trained on; 64 more are margin), then
# a replay server with no teacher serves it to the trainers. Same identity (stream_sha256) and the same bytes as a live server on this machine.
STREAM = "stores/teacher_stream_s42"
RECORD = Step("teacher:record", "train-8", [*PY, "src/teacher_server.py", "--record_dir", STREAM, "--record_batches", "6168",
                                            "--dataset_revision", "9bb295ddab0e05d785b879661af7260fed5140fc"], f"{STREAM}/COMPLETE")
REPLAY = [*PY, "src/teacher_server.py", "--replay_dir", STREAM, "--host", "127.0.0.1", "--port", "5555"]
SECOND_MACHINE = "Trained on the second machine (RTX 3060 Ti, 8 GB) at micro-batch 4 x 8 accumulation; the update is the same 32 sequences."


def train(name, script, out, args, record, quirks=()):
    steps = args.split("--max_steps ")[1].split()[0]
    return Step(name, name.split(":")[0], [*PY, f"src/{script}", "--output_dir", out, *a(args)], f"{out}/checkpoint-{steps}/model.safetensors", record,
                list(quirks))


PLAN = [
    # -- data: the stored teacher targets of the four-block study (98,304 training + 256 eval sequences, data seed 42)
    Step("store", "store", [*PY, "src/distill.py", "--stage", "extract", "--targets_dir", "stores/fineweb_s42_50m", "--extract_tokens", "50331648",
                             "--dataset_revision", "9bb295ddab0e05d785b879661af7260fed5140fc"],
         "stores/fineweb_s42_50m/eval/meta.json", "store_fineweb_s42_50m",
         ["--extract_tokens 50331648 = 98,304 x 512: exactly 3,072 updates of 32 sequences (the default, 50,000,000, would give 97,656).",
          "Copied to the second machine for its runs (reproduce/as_run/small_study/push_store.py); every four-block run reads identical targets."]),
    # -- eight-block runs: teacher targets streamed from src/teacher_server.py (data seed 42), 12,208 updates of 32 x 512 tokens = 200M tokens
    train("train-8:A", "distill.py", "runs/arm_a_merge", f"--gated_merge true {COMMON_8}", "A",
          [("Ran concurrently with C, the two sharing one teacher pass of the server; B ran afterwards on a second pass. A, B and C record the same "
           "stream_sha256 and eval_sha256 (A and B in results/run_records; C's record stayed on the second machine).")]),
    train("train-8:B", "distill.py", "runs/arm_b_nomerge", f"--gated_merge false {COMMON_8}", "B"),
    train("train-8:C", "compare_arch.py", "runs/arm_c_noloop", COMMON_8, "C",
          [SECOND_MACHINE.replace("at micro-batch 4 x 8 accumulation", "at the default micro-batch 4 x 8 accumulation (as A and B)"),
           "Only config.json and model.safetensors were copied back: no training_args.bin, no trainer_state.json. Checked against the launcher only."]),
    # -- four-block runs: three seeds each; the seed sets the merge gate's initialization, the data order and the depth sequence
    *[train(f"train-4:A'_s{s}", "distill.py", f"runs/small/a_s{s}", f"--gated_merge true --seed {s} {COMMON_4} --eval_steps 256 --eval_depths 1 2 4 8 {SPLIT['5070']}",
            f"Ap_s{s}") for s in (1, 2, 3)],
    *[train(f"train-4:B'_s{s}", "distill.py", f"runs/small/b_s{s}", f"--gated_merge false --seed {s} {COMMON_4} --eval_steps 256 --eval_depths 1 2 4 8 {SPLIT['5070']}",
            f"Bp_s{s}") for s in (1, 2, 3)],
    *[train(f"train-4:E'_s{s}", "distill.py", f"runs/small/e_s{s}",
            f"--gated_merge true --merge_base anchor --seed {s} {COMMON_4} --eval_steps 256 --eval_depths 1 2 4 8 {SPLIT['5070' if s < 3 else '3060Ti']}",
            f"Ep_s{s}", [SECOND_MACHINE] if s == 3 else []) for s in (1, 2, 3)],
    *[train(f"train-4:C'_s{s}", "compare_arch.py", f"runs/small_c/c_s{s}", f"--seed {s} {COMMON_4} --eval_steps 256 {SPLIT['3060Ti']}", f"Cp_s{s}",
            [SECOND_MACHINE, "Launched with --output_dir runs/small/c_s<seed> there and copied back to runs/small_c/ here.",
             ("Only config.json, model.safetensors and trainer_state.json were copied back: no training_args.bin. Checked against the launcher and "
             "the trainer state.")]) for s in (1, 2, 3)],
    *[train(f"train-4:D'_s{s}", "compare_arch.py", f"runs/iso_flop/d_s{s}",
            f"--seed {s} {COMMON_4.replace('--looped_layers 2', '--looped_layers 8')} --flat_layer_types modernbert --eval_steps 512 "
            f"{SPLIT['5070' if s < 3 else '3060Ti']}", f"Dp_s{s}",
            ["--prelude_layers 1 --looped_layers 8 --coda_layers 1 is how compare_arch.py is told to build ten flat blocks; there is no loop.",
             *([SECOND_MACHINE, "No training_args.bin was copied back. Checked against the launcher and the trainer state."] if s == 3 else [])])
      for s in (1, 2, 3)],
    # -- evaluation
    # -- evaluation (outputs in reproduced/)
    Step("eval:pm-bpb", "eval", [*PY, "src/crossmodel_eval.py", "--out", f"{R}/mlm_crossmodel", "--models", *MODELS], f"{R}/mlm_crossmodel/raw.json",
         quirks=[("As run, the models were scored in four invocations as they finished (results accumulate in raw.json); one invocation with "
                  "every model gives the same scores. DeBERTa-base was scored in the NLI run only.")]),
    Step("eval:heldout-loss-8", "eval", [*PY, "src/bench_heldout.py", "--out", f"{R}/bench_heldout.json", "--models", "A=runs/arm_a_merge/checkpoint-12208",
                                         "B=runs/arm_b_nomerge/checkpoint-12208", "C=runs/arm_c_noloop/checkpoint-12208:flat"], f"{R}/bench_heldout.json"),
    Step("eval:nli", "eval", [*PY, "src/zeroshot_nli.py", "--out", f"{R}/zeroshot_nli", "--models", *MODELS], f"{R}/zeroshot_nli/raw.json",
         quirks=["As run, in two invocations (D' and E' later); results accumulate in raw.json."]),
    Step("eval:static-cost", "eval", [*PY, "src/model_cost.py", "--out", f"{R}/hardware"], f"{R}/hardware/static.csv"),
    Step("eval:quantization", "eval", [*PY, "src/quant_eval.py", "--out", f"{R}/hardware", "--texts_file", f"{R}/mlm_crossmodel/texts.json", "--models",
                                       "c_s1", "d_s1", "a_s1", "b_s1", "e_s1", "C8", "A8", "B8", "roberta-base", "teacher"], f"{R}/hardware/quant.json",
         quirks=["As run, e_s1 in a second invocation; results accumulate in quant.json. Needs eval:pm-bpb first (its cached texts)."]),
    Step("eval:compute", "eval", [*PY, "src/compute_ledger.py", "--out", f"{R}/compute_ledger"], f"{R}/compute_ledger.json",
         quirks=["Counts operations with torch.utils.flop_counter on one step at depths 1 and 2 and extends linearly (the paper, Section 5.4)."]),
    # -- halting
    Step("halting:store", "halting", [*PY, "src/halting_data.py", "--out", "stores/halting", "--train", "4096", "--eval", "512"],
         "stores/halting/arxiv/eval/meta.json", "store_halting", ["Data seed 4242: a different stream from the training runs' (seed 42)."]),
    Step("halting:heads", "halting", [*PY, "src/halting_train.py", "--out", f"{R}/halting_v2", "--models",
         *[f"{x}_s{s}=runs/small/{x}_s{s}/checkpoint-3072" for x in "ab" for s in (1, 2, 3)],
         "A8=runs/arm_a_merge/checkpoint-12208", "B8=runs/arm_b_nomerge/checkpoint-12208"], f"{R}/halting_v2/summary.md",
         quirks=["Also writes each model's head into <checkpoint>-halting next to its checkpoint."]),
    Step("halting:explore-features-train", "halting", [*PY, "src/halting_features.py", "--student", "runs/arm_a_merge/checkpoint-12208", "--out", "feats/a_train",
         "--teacher_addr", TEACHER, "--data_seed", "4242", "--first_shard", "0", "--shards", "40"], "feats/a_train/shard-000039.pt",
         quirks=[("As run, shards 4 and 5 were produced on the second machine into feats/a_train_220; a shard is a pure function of "
                  "(student, data seed, shard index), so one directory with all 40 is equivalent.")]),
    Step("halting:explore-features-eval", "halting", [*PY, "src/halting_features.py", "--student", "runs/arm_a_merge/checkpoint-12208", "--out",
         "feats/a_eval_fineweb", "--source", "fineweb", "--samples", "512"], "feats/a_eval_fineweb/shard-000001.pt"),
    Step("halting:explore-features-arxiv", "halting", [*PY, "src/halting_features.py", "--student", "runs/arm_a_merge/checkpoint-12208", "--out",
         "feats/a_eval_arxiv", "--source", "arxiv", "--samples", "512"], "feats/a_eval_arxiv/shard-000001.pt"),
    Step("halting:explore-fit-linear", "halting", [*PY, "src/halting_fit.py", "--train", "feats/a_train", "--eval", "arxiv=feats/a_eval_arxiv",
         "fineweb=feats/a_eval_fineweb", "--out", f"{R}/halting_v2/target_variants/A8_linear.json"], f"{R}/halting_v2/target_variants/A8_linear.json"),
    Step("halting:explore-fit-hidden", "halting", [*PY, "src/halting_fit.py", "--train", "feats/a_train", "--eval", "arxiv=feats/a_eval_arxiv",
         "fineweb=feats/a_eval_fineweb", "--labels", "v1", "future", "gain", "--features", "SD", "SCD", "--epochs", "10", "--hidden", "256", "--out",
         f"{R}/halting_v2/target_variants/A8_mlp.json"], f"{R}/halting_v2/target_variants/A8_mlp.json"),
    # -- hardware: alone on the machine, the power logger running in the background as it did during the reported sweep
    Step("hardware:latency", "hardware", [*PY, "src/hw_bench.py", "--out", f"{R}/hardware", "--backend", "openvino", "--devices", "NPU", "GPU.0", "CPU",
                                          "--retry_errors"], f"{R}/hardware/latency.openvino.done", background=POWER,
         quirks=[("The reported sweep ran src/hw_power.py (a package- and GPU-power logger) in the background; its output is not used, but it was "
                  "part of the machine's load while timing, so it runs here too. E' (e_s1) was timed later without it."),
                 "DeBERTa-base at 512 tokens hangs the NPU and is excluded there.",
                 ("As run, a second pass timed every model at R = 4 on every backend (results/hardware/depth4_pass/); the paper's Table 4 uses the "
                  "main sweep only.")]),
    Step("hardware:latency-cuda", "hardware", [*PY, "src/hw_bench.py", "--out", f"{R}/hardware", "--backend", "torch", "--devices", "cuda", "--retry_errors"],
         f"{R}/hardware/latency.cuda.done", background=POWER),
    Step("hardware:latency-cpu", "hardware", [*PY, "src/hw_bench.py", "--out", f"{R}/hardware", "--backend", "torch", "--devices", "cpu", "--batches", "1",
                                              "--retry_errors"], f"{R}/hardware/latency.cpu.done", background=POWER),
    # -- Table 4: every model at R = 4 on every backend, one pass per backend (as run: reproduce/as_run/iso_flop/sweep_final.sh, then
    #    input_injection/hw_e.sh for E'), the power logger in the background as then
    *[Step(f"hardware:r4-{name}", "hardware", [*PY, "src/hw_bench.py", "--out", f"{R}/hardware/depth4_pass", "--seqs", "512", "--batches", "1",
                                               "--depths", "4", "--backend", backend, "--devices", *devices, *extra],
           f"{R}/hardware/depth4_pass/{name}.done", background=POWER,
           quirks=["DeBERTa-base is left out on the NPU: at 512 tokens it hangs the device."] if name == "npu" else [])
      for name, backend, devices, extra in (("openvino", "openvino", ["GPU.0", "CPU"], []), ("cuda", "torch", ["cuda"], []), ("cpu", "torch", ["cpu"], []),
                                             ("npu", "openvino", ["NPU"], ["--models", "c_s1", "d_s1", "a_s1", "b_s1", "e_s1", "C8", "A8", "B8",
                                                                          "distilbert-cased", "teacher", "roberta-base"]))],
    # -- reports, figures and the number check on reproduced/ (CPU, seconds)
    Step("report:tables", "report", [*PY, "src/mlm_report.py", "--raw", f"{R}/mlm_crossmodel/raw.json", "--out", f"{R}/mlm_crossmodel"], ""),
    Step("report:four-block", "report", [*PY, "src/small_report.py", "--runs", "runs/small", "runs/small_c", "runs/iso_flop", "--out", f"{R}/small_study"], ""),
    Step("report:eight-block", "report", [*PY, "src/arms200m_report.py", "--bench", f"{R}/bench_heldout.json", "--out", f"{R}/arms_200m"], ""),
    Step("report:nli", "report", [*PY, "src/nli_report.py", "--raw", f"{R}/zeroshot_nli/raw.json", "--out", f"{R}/zeroshot_nli"], ""),
    Step("report:hardware", "report", [*PY, "src/hw_report.py", "--dir", f"{R}/hardware"], ""),
    Step("report:figures", "report", [*PY, "paper/figures.py"], "", env={"RESULTS_DIR": R, "FIGURES_DIR": f"{R}/figures"}),
    Step("report:check", "report", [*PY, "paper/check_numbers.py"], "", env={"RESULTS_DIR": R}),
]
STAGES = ("store", "train-8", "train-4", "eval", "halting", "hardware", "report")


# ---------------------------------------------------------------------------------------------------------------- running it
def resolve(argv, teacher):
    if any(TEACHER in x for x in argv) and not teacher:
        raise SystemExit("this step streams from the teacher server: pass --teacher-host (start it with: uv run python src/teacher_server.py --port 5555)")
    return [x.replace(TEACHER, f"{teacher}:5555") for x in argv]


def run(step, teacher):
    if step.done and (ROOT / step.done).exists():
        print(f"[skip] {step.name}: {step.done} exists")
        return
    if "--output_dir" in step.argv:  # never train into a directory that holds a run: an existing run is either finished (skipped above) or partial
        out = ROOT / step.argv[step.argv.index("--output_dir") + 1]
        if out.exists():
            raise SystemExit(f"{step.name}: {out} exists without its final checkpoint ({step.done}); move it away to retrain")
    argv = resolve(step.argv, teacher)
    log = ROOT / "runs" / "logs" / f"{step.name.replace(':', '_').replace(chr(39), 'p')}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    print(f"[run] {step.name}: {shlex.join(argv)}  > {log.relative_to(ROOT)}", flush=True)
    logger = None
    if step.background:
        stop = ROOT / R / "hardware" / "stop"
        stop.parent.mkdir(parents=True, exist_ok=True)
        stop.unlink(missing_ok=True)
        logger = subprocess.Popen(step.background, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(25)  # the idle baseline the logger records before the sweep, as run
    with open(log, "w") as f:
        code = subprocess.run(argv, check=False, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, env={**os.environ, "REPRO_PINNED": "1", **step.env}).returncode
    if logger:
        (ROOT / R / "hardware" / "stop").touch()
        logger.wait()
    if code:
        raise SystemExit(f"{step.name} failed (exit {code}); see {log}")
    if step.done and step.done.endswith(".done"):  # a marker for steps whose outputs accumulate in a shared file (the latency sweeps)
        (ROOT / step.done).write_text("done\n")
    elif step.done and not (ROOT / step.done).exists():
        raise SystemExit(f"{step.name} exited cleanly but did not write {step.done}")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--list", action="store_true", help="print the plan with its recorded quirks")
    p.add_argument("--dry-run", action="store_true", help="print the commands of the selected stages without running them")
    p.add_argument("--stage", nargs="+", choices=STAGES, help="the stages to run or print (required unless --list)")
    p.add_argument("--only", nargs="+", default=[], help="step names, e.g. \"train-4:A'_s1\"")
    p.add_argument("--teacher-host", default="", help="host of src/teacher_server.py (port 5555) for the streaming steps")
    p.add_argument("--single-machine", action="store_true", help="train-8 without a second machine: record the teacher stream, then replay it")
    a_ = p.parse_args(argv)
    if a_.list:
        for s in PLAN:
            print(f"{s.stage:9s} {s.name}" + (f"   [record: {s.record}]" if s.record else ""))
            for q in s.quirks:
                print(f"{'':12s}- {q}")
        return
    if not a_.stage:
        p.error("--stage is required: name the stages to run (or print with --dry-run); --list shows the plan")
    steps = [s for s in PLAN if s.stage in a_.stage and (not a_.only or s.name in a_.only)]
    teacher, server = a_.teacher_host, None
    if a_.single_machine and any(s.stage == "train-8" for s in steps):
        if a_.dry_run:
            print(shlex.join(RECORD.argv))
            print(shlex.join(REPLAY) + "  &")
        else:
            run(RECORD, "")
            server = subprocess.Popen(REPLAY, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            time.sleep(20)  # the replay server reads the recording's identity and eval frames, then listens
        teacher = "127.0.0.1"
    try:
        _run_all(steps, a_, teacher)
    finally:
        if server:
            server.terminate()


def _run_all(steps, a_, teacher):
    for s in steps:
        if a_.dry_run:
            print(shlex.join(s.argv))
        else:
            run(s, teacher)


if __name__ == "__main__":
    sys.exit(main())
