"""mlm_report: grouping of seeds, bits-per-byte, best depth, summary from synthetic raw scores."""
import json
import math

import pytest

import mlm_report as mr


def cell(bits_per_byte, bytes_=1000):
    return {"nats": bits_per_byte * bytes_ * math.log(2), "tokens": 300, "bytes": bytes_}


def raw():
    out = {}
    for seed, off in ((1, 0.0), (2, 0.02), (3, 0.04)):
        out[f"a_s{seed}"] = {str(d): {ds: cell(1.0 - 0.05 * min(d, 4) + 0.01 * max(0, d - 4) + off) for ds in mr.DATASETS} for d in range(1, 9)}
        out[f"c_s{seed}"] = {"1": {ds: cell(0.95 + off) for ds in mr.DATASETS}}
    out["roberta-base"] = {"1": {ds: cell(0.9) for ds in mr.DATASETS}}
    out["unknown-model"] = {"1": {ds: cell(0.5) for ds in mr.DATASETS}}
    return out


def test_bits_per_byte_and_grouping_ignore_unknown_models():
    assert mr.bpb(cell(0.75)) == pytest.approx(0.75)
    rows = mr.table_rows(raw())
    assert set(rows) == {"small a", "small c", "roberta-base"}
    assert len(rows["small a"][4]["arxiv"]) == 3 and rows["small a"][4]["arxiv"][1] == pytest.approx(0.82)


def test_summary_reports_depth_1_the_best_depth_and_depth_8_with_seed_spread(tmp_path):
    text = mr.summary(mr.table_rows(raw()))
    lines = [ln for ln in text.splitlines() if ln.startswith("| A'")]
    assert [ln.split("|")[3].strip() for ln in lines] == ["1", "4 (best)", "8"] and {ln.split("|")[2].strip() for ln in lines} == {"3"}
    assert "+-" in lines[1] and "RoBERTa-base" in text and "C' flat" in text
    (tmp_path / "raw.json").write_text(json.dumps(raw()))
    mr.main(["--raw", str(tmp_path / "raw.json"), "--out", str(tmp_path / "out")])
    assert (tmp_path / "out" / "bpb.csv").exists()
