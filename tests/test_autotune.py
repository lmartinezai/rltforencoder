from types import SimpleNamespace
from typing import ClassVar

import pytest
import torch

import autotune

EFF, SEQ = 32, 512


def run_row(mb=4, ckpt=0, compile=1, step_s=1.0, gb=6.0, fail=None):
    r = {"mb": mb, "ckpt": ckpt, "compile": compile}
    return r | ({"oom": True} if fail == "oom" else {"step_s": step_s, "peak_reserved_gb": gb, "steps": 16})


def table(*rows):
    return autotune.summarize(list(rows), EFF, SEQ)


def test_summarize_takes_median_step_time_and_any_failure_disqualifies():
    t = table(run_row(step_s=1.0), run_row(step_s=1.5), run_row(step_s=1.2),
              run_row(mb=8, step_s=0.5), run_row(mb=8, fail="oom"))
    four, eight = (t[t["mb"] == m].iloc[0] for m in (4, 8))
    assert four["ok"] and four["step_s"] == 1.2 and abs(four["tok_s"] - EFF * SEQ / 1.2) < 1e-6
    assert abs(four["spread"] - 0.5) < 1e-9
    assert not eight["ok"] and eight["tok_s"] != eight["tok_s"]  # NaN: one OOM in the repeats disqualifies the candidate


def test_pick_is_the_fastest_feasible_candidate():
    best = autotune.pick(table(run_row(mb=2, step_s=2.0), run_row(mb=4, step_s=1.0), run_row(mb=8, fail="oom")))
    assert best["mb"] == 4
    assert autotune.pick(table(run_row(fail="oom"))) is None  # nothing feasible


def test_ties_inside_the_candidates_noise_go_to_less_vram():
    # mb 4 is 0.5% faster than mb 2 (inside the 1% band): the one reserving less VRAM wins
    assert autotune.pick(table(run_row(mb=4, step_s=1.0, gb=8.0), run_row(mb=2, step_s=1.005, gb=5.0)))["mb"] == 2
    # genuinely faster (>1%, no measured noise) still wins
    assert autotune.pick(table(run_row(mb=4, step_s=1.0, gb=8.0), run_row(mb=2, step_s=1.1, gb=5.0)))["mb"] == 4


def test_large_spread_does_not_make_a_slow_candidate_a_tie():
    t = table(run_row(mb=4, step_s=1.00, gb=8), run_row(mb=4, step_s=1.002, gb=8),
              run_row(mb=2, step_s=1.04, gb=5), run_row(mb=2, step_s=1.14, gb=5),
              run_row(mb=1, step_s=1.40, gb=3), run_row(mb=1, step_s=1.55, gb=3))
    assert autotune.pick(t)["mb"] == 4


def test_cli_line_carries_every_choice():
    best = autotune.pick(table(run_row(mb=4, ckpt=1, compile=1)))
    assert autotune.cli(best, EFF, "adamw_torch_fused") == (
        "--per_device_train_batch_size 4 --gradient_accumulation_steps 8 --gradient_checkpointing --compile_layers "
        "--optim adamw_torch_fused --max_depth 8 --seed 0")
    best = autotune.pick(table(run_row(mb=2, ckpt=0, compile=0)))
    assert "--gradient_checkpointing" not in autotune.cli(best, EFF, "adamw_torch") and "--compile_layers" not in autotune.cli(best, EFF, "adamw_torch")


def test_ladder_must_divide_the_effective_batch():
    autotune.check_ladder([1, 2, 4, 8, 16, 32], 32)
    with pytest.raises(ValueError, match=r"\[6, 12\].*divisors"):
        autotune.check_ladder([1, 2, 4, 6, 8, 12, 16], 32)  # the silent skip that once hid mb 6 and 12


def test_step_timer_times_whole_steps_after_warmup(monkeypatch):
    clock = iter([10.0, 11.0, 13.0, 16.0, 20.0])  # ends of steps 1..5: durations 1, 2, 3, 4
    monkeypatch.setattr(autotune.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(autotune.torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(autotune.torch.cuda, "max_memory_reserved", lambda: 3 * 2**30)
    monkeypatch.setattr(autotune.time, "perf_counter", lambda: next(clock))
    timer = autotune.StepTimer(warmup=2)
    for step in range(1, 6):
        timer.on_step_end(None, SimpleNamespace(global_step=step), None)
    assert timer.result() == {"step_s": (20.0 - 11.0) / 3, "steps": 3, "peak_reserved_gb": 3.0}  # steps 3..5, the warm-up excluded


def test_nothing_feasible_exits_nonzero(monkeypatch, tmp_path):
    monkeypatch.setattr(autotune, "launch", lambda kind, *a, **k: {"oom": True})
    (tmp_path / "pool" / "train").mkdir(parents=True)
    class Store:
        meta: ClassVar[dict] = {"seq_len": 512, "n_seq": 1600}
        def __len__(self):
            return 1600
    monkeypatch.setattr(autotune, "TargetStore", lambda _: Store())
    args = autotune.parser().parse_args(["--pool", str(tmp_path / "pool"), "--out", str(tmp_path / "out"), "--repeats", "1", "--mbs", "4"])
    with pytest.raises(SystemExit, match="no feasible"):
        autotune.run(args, log=lambda *_: None)
    import json
    report = json.loads((tmp_path / "out" / "recommended.json").read_text())
    assert report["status"] == "failed" and report["recommendation"] is None and report["runs"]


def test_rejects_wrong_length_and_incomplete_pool_before_workers(monkeypatch, tmp_path):
    (tmp_path / "pool").mkdir()
    (tmp_path / "pool" / "train").mkdir()
    class Store:
        meta: ClassVar[dict] = {"seq_len": 512, "n_seq": 1200}
        def __len__(self):
            return 1200
    monkeypatch.setattr(autotune, "TargetStore", lambda _: Store())
    monkeypatch.setattr(autotune, "launch", lambda *a, **k: pytest.fail("worker launched"))
    args = autotune.parser().parse_args(["--pool", str(tmp_path / "pool"), "--seq-len", "1024"])
    with pytest.raises(ValueError, match="sequence length"):
        autotune.run(args, log=lambda *_: None)
    args.seq_len = 512
    Store.meta = {"seq_len": 512, "n_seq": 1000}
    with pytest.raises(ValueError, match="count"):
        autotune.run(args, log=lambda *_: None)


def test_pool_must_cover_all_full_updates(monkeypatch, tmp_path):
    args = autotune.parser().parse_args(["--pool", str(tmp_path / "pool"), "--pool-seqs", "1024"])
    monkeypatch.setattr(autotune, "launch", lambda *a, **k: pytest.fail("worker launched"))
    with pytest.raises(ValueError, match="1440"):
        autotune.run(args, log=lambda *_: None)


def test_driver_validates_separately_and_reports_aligned_depths(monkeypatch, tmp_path):
    import json

    from distill import depth_for_step

    pool = tmp_path / "pool"
    pool.mkdir()
    (pool / "train").mkdir()
    class Store:
        meta: ClassVar[dict] = {"seq_len": 512, "n_seq": 2000}
        def __len__(self):
            return 2000
    monkeypatch.setattr(autotune, "TargetStore", lambda _: Store())
    calls = []
    def launch(kind, args, **spec):
        calls.append((kind, args["max_steps"], spec.get("warmup")))
        if kind == "validate":
            return {"valid": True}
        depths = [depth_for_step(s, args["max_depth"], args["seed"])
                  for s in range(spec["warmup"], args["max_steps"])]
        return {"step_s": 1.0, "steps": len(depths), "depths": depths, "peak_reserved_gb": 5.0}
    monkeypatch.setattr(autotune, "launch", launch)
    args = autotune.parser().parse_args(["--pool", str(pool), "--out", str(tmp_path / "out"),
                                        "--mbs", "4", "--repeats", "2"])
    autotune.run(args, log=lambda *_: None)
    assert calls == [("validate", 24, None), ("train", 24, 8), ("train", 24, 8)]
    report = json.loads((tmp_path / "out" / "recommended.json").read_text())
    assert report["status"] == "ok" and report["recommendation"]["comparison"] == "inconclusive"
    assert "--seq_len" not in report["recommendation"]["cli"]


@pytest.mark.parametrize("depth,warmup,cycles,seed", [(8, 2, 2, 0), (8, 8, 1, 31), (5, 6, 3, 17), (1, 1, 2, 4)])
def test_timed_window_contains_complete_actual_shuffled_depth_buckets(depth, warmup, cycles, seed):
    from distill import depth_for_step

    start = autotune.aligned_warmup(warmup, depth)
    assert start >= warmup and start % depth == 0
    for offset in range(0, cycles * depth, depth):
        observed = [depth_for_step(s, depth, seed) for s in range(start + offset, start + offset + depth)]
        assert sorted(observed) == list(range(1, depth + 1))


def test_nonfinite_loss_gradient_and_final_parameters_rejected():
    model = torch.nn.Linear(2, 1)
    for kind in ("loss", "gradient", "parameter"):
        model = torch.nn.Linear(2, 1)
        check = autotune.Finiteness()
        check.check(torch.tensor(float("nan") if kind == "loss" else 1.0))
        model.weight.grad = torch.full_like(model.weight, float("inf") if kind == "gradient" else 1.0)
        check.on_pre_optimizer_step(None, None, None, model=model)
        if kind == "parameter":
            model.bias.data.fill_(float("nan"))
        with pytest.raises(ValueError, match="non-finite"):
            check.validate(model)


def test_real_trainer_invokes_finiteness_checks(monkeypatch, tmp_path):
    import distill
    from distill import DistillArguments

    def step(model, teacher, bias, inputs, depth, args):
        loss = model(inputs["x"]).square().mean()
        return loss, {"ce": loss.detach(), "kl": loss.detach()}

    monkeypatch.setattr(distill, "distill_step", step)
    check = autotune.Finiteness()
    class Monitored(autotune.CheckedTrainer):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.validity = check
    args = DistillArguments(output_dir=str(tmp_path), use_cpu=True, bf16=False, optim="adamw_torch",
                            max_steps=2, per_device_train_batch_size=2, gradient_accumulation_steps=1,
                            eval_strategy="no", save_strategy="no", report_to="none", disable_tqdm=True)
    model = torch.nn.Linear(2, 1)
    trainer = Monitored(model=model, args=args, teacher=None, teacher_bias=torch.zeros(1),
                        train_dataset=[{"x": torch.ones(2)} for _ in range(4)], callbacks=[check])
    trainer.train()
    check.validate(model)
    assert len(check.flags) >= 2 * 3  # loss and gradients for each update, plus final parameters


def test_timer_window_matches_depths_executed_by_trainer(monkeypatch, tmp_path):
    import distill
    from distill import DistillArguments, DistillTrainer

    depths = []
    def step(model, teacher, bias, inputs, depth, args):
        depths.append(depth)
        loss = model(inputs["x"]).square().mean()
        return loss, {"ce": loss.detach(), "kl": loss.detach()}

    monkeypatch.setattr(distill, "distill_step", step)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda: None)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda: 0)
    warmup, depth, cycles = autotune.aligned_warmup(2, 3), 3, 2
    args = DistillArguments(output_dir=str(tmp_path), use_cpu=True, bf16=False, optim="adamw_torch", seed=7,
                            max_depth=depth, max_steps=warmup + depth * cycles, per_device_train_batch_size=1,
                            gradient_accumulation_steps=1, eval_strategy="no", save_strategy="no",
                            report_to="none", disable_tqdm=True)
    timer = autotune.StepTimer(warmup)
    trainer = DistillTrainer(model=torch.nn.Linear(2, 1), args=args, teacher=None, teacher_bias=torch.zeros(1),
                             train_dataset=[{"x": torch.ones(2)} for _ in range(args.max_steps)], callbacks=[timer])
    trainer.train()
    assert timer.result()["steps"] == depth * cycles
    for i in range(warmup, len(depths), depth):
        assert sorted(depths[i:i + depth]) == list(range(1, depth + 1))


# ---------------------------------------------------------------- the real process boundary

def test_launch_reports_a_failed_worker_with_its_stderr_tail():
    res = autotune.launch("train", {"no_such_training_argument": 1}, warmup=1, timeout=300)
    assert "error" in res and "no_such_training_argument" in res["error"] and "oom" not in res


def test_launch_reports_a_timeout():
    assert "timed out" in autotune.launch("train", {}, warmup=1, timeout=0.01)["error"]


@pytest.mark.gpu
def test_driver_end_to_end_in_real_processes(tmp_path):
    """Pool build, validation and timing each in a fresh subprocess; argument parsing, RESULT lines, and the report files."""
    import json

    a = autotune.parser().parse_args(["--mbs", "2", "--eff-batch", "4", "--compile", "0", "--max-depth", "2", "--warmup", "1",
                                      "--cycles", "1", "--repeats", "1", "--seq-len", "64", "--pool-seqs", "32",
                                      "--pool", str(tmp_path / "pool"), "--out", str(tmp_path / "out")])
    lines = []
    table, best = autotune.run(a, log=lines.append)
    assert best["mb"] == 2 and table.iloc[0]["ok"]
    report = json.loads((tmp_path / "out" / "recommended.json").read_text())
    assert report["status"] == "ok" and report["recommendation"]["cli"].startswith("--per_device_train_batch_size 2")
    assert report["validation"][0]["result"] == {"valid": True}
    run = report["runs"][0]
    assert run["steps"] == 2 and sorted(run["depths"]) == [1, 2] and run["step_s"] > 0 and run["peak_reserved_gb"] > 0
    assert (tmp_path / "out" / "autotune.csv").exists()
    assert any("recommended:" in ln for ln in lines)


# ---------------------------------------------------------------- driver validation and report branches

def _fake_pool(monkeypatch, tmp_path, n=2000):
    pool = tmp_path / "pool"
    (pool / "train").mkdir(parents=True)

    class Store:
        meta: ClassVar[dict] = {"seq_len": 512, "n_seq": n}

        def __len__(self):
            return n

    monkeypatch.setattr(autotune, "TargetStore", lambda _: Store())
    return pool


@pytest.mark.parametrize("change,message", [
    ({"eff_batch": 0}, "must be positive"), ({"repeats": 0}, "must be positive"), ({"seq_len": 2}, "must be positive"),
    ({"ckpt": []}, "nonempty"), ({"compile": [2]}, "0 or 1"), ({"ckpt": [0, 3]}, "0 or 1"),
])
def test_driver_rejects_bad_settings_before_any_worker(monkeypatch, tmp_path, change, message):
    monkeypatch.setattr(autotune, "launch", lambda *a, **k: pytest.fail("worker launched"))
    args = autotune.parser().parse_args(["--mbs", "4", "--out", str(tmp_path / "out")])
    for k, v in change.items():
        setattr(args, k, v)
    with pytest.raises(ValueError, match=message):
        autotune.run(args, log=lambda *_: None)


def test_a_failed_pool_build_stops_the_driver(monkeypatch, tmp_path):
    monkeypatch.setattr(autotune, "launch", lambda *a, **k: {"error": "teacher missing"})
    args = autotune.parser().parse_args(["--pool", str(tmp_path / "pool"), "--out", str(tmp_path / "out"), "--pool-seqs", "5000"])
    with pytest.raises(SystemExit, match="pool build failed"):
        autotune.run(args, log=lambda *_: None)


def test_a_train_pool_that_is_too_small_is_refused(monkeypatch, tmp_path):
    pool = _fake_pool(monkeypatch, tmp_path, n=1400)
    monkeypatch.setattr(autotune, "launch", lambda *a, **k: pytest.fail("worker launched"))
    args = autotune.parser().parse_args(["--pool", str(pool), "--out", str(tmp_path / "out")])
    with pytest.raises(ValueError, match="needs at least"):
        autotune.run(args, log=lambda *_: None)


def test_unbalanced_depth_cycles_invalidate_a_measurement(monkeypatch, tmp_path):
    pool = _fake_pool(monkeypatch, tmp_path)

    def launch(kind, args, **spec):
        if kind == "validate":
            return {"valid": True}
        n = args["max_steps"] - spec["warmup"]
        return {"step_s": 1.0, "steps": n, "depths": [1] * n, "peak_reserved_gb": 5.0}  # never visits depth 2..8

    monkeypatch.setattr(autotune, "launch", launch)
    args = autotune.parser().parse_args(["--pool", str(pool), "--out", str(tmp_path / "out"), "--mbs", "4", "--repeats", "1"])
    with pytest.raises(SystemExit, match="no feasible"):
        autotune.run(args, log=lambda *_: None)


def test_candidates_whose_timings_overlap_are_reported_as_inconclusive(monkeypatch, tmp_path):
    import json

    from distill import depth_for_step

    pool = _fake_pool(monkeypatch, tmp_path)
    speed = {2: [1.00, 1.04, 1.02], 4: [1.01, 1.03, 1.00]}  # same range: indistinguishable
    seen = {2: iter(speed[2]), 4: iter(speed[4])}

    def launch(kind, args, **spec):
        if kind == "validate":
            return {"valid": True}
        depths = [depth_for_step(s, args["max_depth"], args["seed"]) for s in range(spec["warmup"], args["max_steps"])]
        mb = args["per_device_train_batch_size"]
        return {"step_s": next(seen[mb]), "steps": len(depths), "depths": depths, "peak_reserved_gb": 5.0}

    monkeypatch.setattr(autotune, "launch", launch)
    args = autotune.parser().parse_args(["--pool", str(pool), "--out", str(tmp_path / "out"), "--mbs", "2,4", "--repeats", "3"])
    lines = []
    autotune.run(args, log=lines.append)
    report = json.loads((tmp_path / "out" / "recommended.json").read_text())
    assert report["recommendation"]["comparison"] == "inconclusive" and report["recommendation"]["overlapping_candidates"]
    assert any("inconclusive" in line for line in lines)
