from math import comb

import pandas as pd
import pytest

from gts.coverage import author_table, coverage_at_k, coverage_curve, hit_counts, partners


def test_coverage_estimator_edges():
    assert coverage_at_k(50, 0, 10) == 0.0
    assert coverage_at_k(50, 50, 1) == 1.0
    assert coverage_at_k(50, 1, 50) == 1.0
    assert coverage_at_k(10, 3, 1) == pytest.approx(0.3)
    assert coverage_at_k(10, 3, 2) == pytest.approx(1 - comb(7, 2) / comb(10, 2))
    with pytest.raises(ValueError):
        coverage_at_k(10, 11, 1)


def test_coverage_is_monotone_in_k():
    values = [coverage_at_k(50, 4, k) for k in range(1, 51)]
    assert all(a <= b for a, b in zip(values, values[1:]))


def test_hit_counts_use_l4_matching():
    gt = ["Jane Q. Smith", "R. Yehuda"]
    texts = ["J. Smith; A. Other", "Jane Smith; Rachel Yehuda", "K. Smith", ""]
    # "K. Smith" is surname-only (L5), which scoring rejects
    assert hit_counts(gt, texts) == [2, 1]


def _papers():
    rows = []
    for i in range(4):
        rows.append({"paper_id": f"p{i}", "field": "physics", "citation_bin": "0-2",
                     "gt_str": ["A. Alpha, B. Beta", "C. Gamma", "D. Delta", "E. Eps"][i]})
    return pd.DataFrame(rows)


def test_partners_stay_in_stratum_and_exclude_self():
    part = partners(_papers(), n_partners=2)
    assert set(part) == {"p0", "p1", "p2", "p3"}
    for pid, others in part.items():
        assert pid not in others and len(others) == 2


def test_author_table_and_curve():
    samples = {"p0": ["A. Alpha; X. Y", "C. Gamma", "Z. Z"], "p1": ["C. Gamma", "Q. Q", "C. Gamma"]}
    greedy = {"p0": "A. Alpha", "p1": "Q. Q"}
    table = author_table(_papers(), samples, greedy, n_partners=3)
    true = table[table["kind"] == "true"].set_index(["paper_id", "author_index"])
    assert true.loc[("p0", 0), "hits"] == 1 and bool(true.loc[("p0", 0), "in_default"])
    assert true.loc[("p0", 1), "hits"] == 0
    assert true.loc[("p1", 0), "hits"] == 2 and not bool(true.loc[("p1", 0), "in_default"])
    chance = table[table["kind"] == "chance"]
    # p0's samples name C. Gamma, p1's author, once
    assert chance[(chance["paper_id"] == "p0") & (chance["partner_id"] == "p1")]["hits"].tolist() == [1]
    curve = coverage_curve(table, ks=(1, 3))
    assert set(curve.columns) >= {"citation_bin", "k", "true", "chance", "net"}
