"""Paired comparisons on the test split: per-paper HR2, true and false names.

    python tools/paired_tests.py data/outputs/g4_bf16 data/outputs/s2

Methods: the default answer, thinking-mode CoT (refusals as no answer), and
frequency selection over the first 20 samples (about CoT's tokens) and over all
50, with thresholds chosen on the tune split. Differences are paired by paper,
with 95% bootstrap intervals (5,000 resamples of test papers).
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyze_session2 import is_refusal, load_pools, tune  # noqa: E402
from gts.data import load_papers  # noqa: E402
from gts.generate import read_chunks  # noqa: E402
from gts.matching import score_answer  # noqa: E402
from gts.prompts import split_thinking  # noqa: E402
from gts.splits import load_split  # noqa: E402


def per_paper_answers(answers: dict[str, str], ids, gt_str) -> pd.DataFrame:
    rows = []
    for pid in ids:
        s = score_answer(gt_str[pid], answers.get(pid, ""))
        rows.append({"paper_id": pid, "hr2": s["hr2"], "true": s["n_match"], "false": s["n_hal"]})
    return pd.DataFrame(rows).set_index("paper_id")


def per_paper_rule(pools, ids, select, gt_str) -> pd.DataFrame:
    rows = []
    for pid in ids:
        pool = pools[pid]
        names = [pool.form[k] for k, s in zip(pool.keys, select(pool)) if s]
        s = score_answer(gt_str[pid], "; ".join(names))
        rows.append({"paper_id": pid, "hr2": s["hr2"], "true": s["n_match"], "false": s["n_hal"]})
    return pd.DataFrame(rows).set_index("paper_id")


def paired(a: pd.DataFrame, b: pd.DataFrame, col: str, scale: float = 1.0, n_boot: int = 5000) -> str:
    d = (a[col] - b[col]).to_numpy() * scale
    gen = np.random.default_rng(0)
    boots = [d[gen.integers(0, len(d), len(d))].mean() for _ in range(n_boot)]
    return f"{d.mean():+.4f} [{np.percentile(boots, 2.5):+.4f}, {np.percentile(boots, 97.5):+.4f}]"


def main(run1: str, run2: str):
    run1, run2 = Path(run1), Path(run2)
    papers = load_papers()
    gt_str = dict(zip(papers.paper_id, papers.gt_str))
    split = load_split()
    tune_ids = sorted(p for p, s in split.items() if s == "tune")
    test_ids = sorted(p for p, s in split.items() if s == "test")
    samples = {r["paper_id"]: r for r in read_chunks(run1 / "samples")}
    pools = load_pools(run1, samples, gt_str)
    greedy = {r["paper_id"]: r["texts"][0] for r in read_chunks(run1 / "greedy")}
    cot = {}
    for r in read_chunks(run2 / "cot_test"):
        answer = split_thinking(r["texts"][0])[1]
        cot[r["paper_id"]] = "" if is_refusal(answer) else answer

    methods = {"default": per_paper_answers(greedy, test_ids, gt_str),
               "cot": per_paper_answers(cot, test_ids, gt_str)}
    for k in (20, 50):
        cand = {f"count>={t}": (lambda t, k: lambda p: p.counts(k) >= t)(t, k) for t in range(1, k + 1)}
        rule, _ = tune(pools, tune_ids, cand)
        methods[f"freq@{k} ({rule})"] = per_paper_rule(pools, test_ids, cand[rule], gt_str)

    summary = pd.DataFrame({name: {"mean_hr2": df.hr2.mean(), "true_per100": 100 * df.true.mean(),
                                   "false_per100": 100 * df.false.mean()} for name, df in methods.items()}).T
    lines = ["# Paired comparisons on the test split", "", f"{len(test_ids)} test papers.", "",
             summary.round(4).to_markdown(), "",
             "Differences (row minus column), mean with 95% bootstrap interval:", "",
             "| comparison | HR2 | true per 100 | false per 100 |", "|---|---|---|---|"]
    names = list(methods)
    for a in names:
        for b in names:
            if a < b or a == b:
                continue
            A, B = methods[a], methods[b]
            lines.append(f"| {a} vs {b} | {paired(A, B, 'hr2')} | {paired(A, B, 'true', 100)} | "
                         f"{paired(A, B, 'false', 100)} |")
    out = run1 / "analysis" / "paired_tests.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main(*sys.argv[1:])
