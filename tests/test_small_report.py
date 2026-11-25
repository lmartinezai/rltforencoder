"""small_report: reads finished runs' trainer states, aggregates over seeds, writes the CSV and summary."""
import csv
import json

import pytest

import small_report as sr


def fake_run(root, arm, seed, offset, depths=(1, 4)):
    run = root / f"{arm}_s{seed}"
    (run / "checkpoint-512").mkdir(parents=True)
    (run / "DONE").write_text("")
    history = []
    for step in (256, 512):
        history.append({"loss": 3.0 - step / 1000 + offset, "step": step})
        for d in depths:  # as the Trainer logs it: one entry per depth
            history.append({f"eval_R{d}_loss": 2.5 - step / 1000 + offset + d / 100, "eval_runtime": 1.0, "step": step})
    (run / "checkpoint-512" / "trainer_state.json").write_text(json.dumps({"log_history": history}))
    return run


def test_collect_summarise_and_write(tmp_path):
    for seed, off in ((1, 0.0), (2, 0.1), (3, 0.2)):
        fake_run(tmp_path, "a", seed, off)
        fake_run(tmp_path, "b", seed, off + 0.05)
        fake_run(tmp_path, "c", seed, off - 0.1, depths=(1,))
    (tmp_path / "a_s9").mkdir()  # unfinished (no DONE, no checkpoint): ignored
    runs = sr.collect([tmp_path])
    assert sorted(runs) == [(arm, s) for arm in "abc" for s in (1, 2, 3)]
    summary = sr.summarise(runs)
    a1 = summary["a"]["eval_R1_loss"]
    assert a1["n"] == 3 and a1["mean"] == pytest.approx(2.5 - 0.512 + 0.01 + 0.1) and a1["std"] == pytest.approx(0.1)
    assert a1["per_seed"] == {1: pytest.approx(1.998), 2: pytest.approx(2.098), 3: pytest.approx(2.198)}
    assert set(summary["c"]) == {"eval_R1_loss"} and set(summary["a"]) == {"eval_R1_loss", "eval_R4_loss"}
    out = tmp_path / "out"
    out.mkdir()
    sr.write_csv(runs, out / "curves.csv")
    with open(out / "curves.csv") as f:
        rows = list(csv.DictReader(f))
    assert {r["arm"] for r in rows} == {"a", "b", "c"} and {r["metric"] for r in rows} == {"loss", "eval_R1_loss", "eval_R4_loss"}
    sr.main(["--runs", str(tmp_path), "--out", str(tmp_path / "main")])
    assert json.loads((tmp_path / "main" / "summary.json").read_text())["b"]["eval_R4_loss"]["n"] == 3


def test_no_finished_runs_is_an_error(tmp_path):
    with pytest.raises(SystemExit, match="no finished runs"):
        sr.main(["--runs", str(tmp_path), "--out", str(tmp_path / "x")])
