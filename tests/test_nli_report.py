"""nli_report: grouping, pooling of MNLI's two dev sets, template averaging, depth selection and the tables, on synthetic raw scores."""
import json

import pytest

import nli_report as nr


def cell(raw, cal, n, majority=0.4, share=0.6):
    return {"raw": raw, "calibrated": cal, "raw_top_label_share": share, "calibrated_top_label_share": share - 0.1, "majority": majority, "n": n}


def per_split(offset):
    templates = lambda base, n, maj: {"T1": cell(base + offset, base + offset + 0.1, n, maj), "T2": cell(base + offset + 0.02, base + offset + 0.12, n, maj)}
    return {"mnli/validation_matched": templates(0.40, 100, 0.35), "mnli/validation_mismatched": templates(0.50, 300, 0.36),
            "qnli/validation": templates(0.50, 50, 0.5), "rte/validation": templates(0.52, 20, 0.53)}


def raw():
    out = {f"a_s{s}": {str(d): per_split(0.01 * d + 0.001 * s) for d in range(1, 9)} for s in (1, 2, 3)}
    out["roberta-base"] = {"1": per_split(0.0)}
    out["unknown"] = {"1": per_split(0.0)}
    return out


def test_mnli_pools_the_two_dev_sets_by_size_and_averages_the_two_templates():
    p = per_split(0.0)
    expected = (100 * (0.40 + 0.42) / 2 + 300 * (0.50 + 0.52) / 2) / 400
    assert nr.task_value(p, "MNLI", "raw") == pytest.approx(expected)
    assert nr.majority(p, "MNLI") == pytest.approx((100 * 0.35 + 300 * 0.36) / 400)


def test_collect_groups_seeds_and_ignores_unknown_models():
    data = nr.collect(raw())
    assert set(data) == {"A' looped + gated merge (4 blocks)", "RoBERTa-base"}
    assert len(data["A' looped + gated merge (4 blocks)"][4]["RTE"]["calibrated"]) == 3


def test_tables_show_only_the_predeclared_depths_with_seed_spread_and_the_floor(tmp_path):
    text = nr.tables(raw())
    rows = [ln for ln in text.splitlines() if ln.startswith("| A'")]
    assert [r.split("|")[2].strip() for r in rows][:3] == ["1", "4", "8"] and "+-" in rows[0] and "majority class (the floor)" in text
    (tmp_path / "raw.json").write_text(json.dumps(raw()))
    nr.main(["--raw", str(tmp_path / "raw.json"), "--out", str(tmp_path / "out")])
    assert (tmp_path / "out" / "accuracy.csv").exists()
