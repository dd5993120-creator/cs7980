"""Deep-dive on the 1,023-paper control subset: 200 samples, and CoT self-consistency.

    python tools/analyze_deep.py data/outputs/g4_bf16 data/outputs/s2

Needs <s2>/samples200 and/or <s2>/cot5 (plus <s2>/cot_test, cot_tune for single CoT
answers). Writes <run1>/analysis/deep_summary.md and CSV tables.

    coverage@k  share of true authors named by at least one of k answers (unbiased
                estimate from all n answers), against random-partner and
                topic-matched chance baselines
    tokens      mean generated tokens per paper for k answers, so thinking and
                non-thinking answers can be compared at equal compute
Selection thresholds are chosen on the subset's tune papers and reported on its
test papers.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyze_session2 import Pool, evaluate, fmt, tune  # noqa: E402
from gts.coverage import author_table, coverage_at_k  # noqa: E402
from gts.data import load_papers  # noqa: E402
from gts.generate import read_chunks  # noqa: E402
from gts.matching import gt_names  # noqa: E402
from gts.mc import control_subset  # noqa: E402
from gts.prompts import split_thinking  # noqa: E402
from gts.splits import load_split  # noqa: E402


def cot_answers(rec: dict) -> list[str]:
    return [split_thinking(t)[1] for t in rec["texts"]]


def coverage_rows(name, table, n, ks, tokens_per_answer):
    rows = []
    for k in ks:
        if k > n:
            continue
        row = {"answers": name, "k": k, "tokens_per_paper": round(k * tokens_per_answer, 1)}
        for kind in ("true", "chance", "chance_topic"):
            h = table[table.kind == kind].hits
            row[kind] = float(np.mean([coverage_at_k(n, int(c), k) for c in h]))
        row["net_topic"] = row["true"] - row["chance_topic"]
        rows.append(row)
    return rows


def main(run1: str, run2: str):
    run1, run2 = Path(run1), Path(run2)
    out = run1 / "analysis"
    papers = load_papers()
    ids = sorted(control_subset(papers, per_stratum=10))
    sub = papers[papers.paper_id.isin(ids)]
    gt_str = dict(zip(papers.paper_id, papers.gt_str))
    greedy = {r["paper_id"]: r["texts"][0] for r in read_chunks(run1 / "greedy") if r["paper_id"] in set(ids)}
    split = load_split()
    sub_tune = [p for p in ids if split[p] == "tune"]
    sub_test = [p for p in ids if split[p] == "test"]
    lines = ["# Deep-dive on the control subset", "",
             f"{len(ids)} papers (10 per field x citation bin where available); {len(sub_tune)} tune, {len(sub_test)} test.", ""]

    cov, pools = [], {}
    s50 = {r["paper_id"]: r for r in read_chunks(run1 / "samples") if r["paper_id"] in set(ids)}
    tok50 = np.mean([np.mean(r["n_tokens"]) for r in s50.values()])
    tab50 = author_table(sub, {p: r["texts"] for p, r in s50.items()}, greedy, n_partners=3, topic=True)
    cov += coverage_rows("non-thinking, 50 samples (session 1)", tab50, 50, (1, 5, 10, 20, 50), tok50)

    if (run2 / "samples200").exists():
        s200 = {r["paper_id"]: r for r in read_chunks(run2 / "samples200")}
        tok200 = np.mean([np.mean(r["n_tokens"]) for r in s200.values()])
        tab200 = author_table(sub, {p: r["texts"] for p, r in s200.items()}, greedy, n_partners=3, topic=True)
        cov += coverage_rows("non-thinking, 200 samples", tab200, 200, (1, 5, 10, 20, 50, 100, 200), tok200)
        pools["non-thinking"] = {p: Pool(r["texts"], r["n_tokens"], gt_names(gt_str[p])) for p, r in s200.items()}

    if (run2 / "cot5").exists():
        c5 = {r["paper_id"]: r for r in read_chunks(run2 / "cot5")}
        answers = {p: cot_answers(r) for p, r in c5.items()}
        tok_c = np.mean([np.mean(r["n_tokens"]) for r in c5.values()])
        unfinished = np.mean([not split_thinking(t)[2] for r in c5.values() for t in r["texts"]])
        tab_c = author_table(sub, answers, greedy, n_partners=3, topic=True)
        cov += coverage_rows("thinking (CoT), 5 answers", tab_c, 5, (1, 2, 3, 5), tok_c)
        pools["thinking"] = {p: Pool(a, r["n_tokens"], gt_names(gt_str[p])) for (p, a), r in zip(answers.items(), c5.values())}
        lines += [f"CoT answers: {tok_c:.0f} tokens on average; {unfinished:.1%} hit the token limit (no answer).", ""]

    cov = pd.DataFrame([fmt(r) for r in cov])
    cov.to_csv(out / "deep_coverage.csv", index=False)
    lines += ["## Coverage per true author vs. compute", "", cov.to_markdown(index=False), ""]

    # selection at matched compute: k answers, frequency threshold tuned on the subset's tune papers
    rows = []
    for name, ks in (("non-thinking", (5, 10, 20, 50, 100, 200)), ("thinking", (1, 2, 3, 5))):
        if name not in pools:
            continue
        P = pools[name]
        for k in ks:
            cand = {f"count>={t}": (lambda t, k: lambda p: p.counts(k) >= t)(t, k) for t in range(1, k + 1)}
            rule, _ = tune(P, [p for p in sub_tune if p in P], cand)
            test = [p for p in sub_test if p in P]
            rows.append({"answers": name, "k": k, "rule": rule,
                         "tokens_per_paper": float(np.mean([P[p].cum_tokens[k - 1] for p in test])),
                         **evaluate(P, test, cand[rule])})
    if rows:
        sel = pd.DataFrame([fmt(r) for r in rows])
        sel.to_csv(out / "deep_selection.csv", index=False)
        lines += ["## Frequency selection at matched compute (subset test papers)", "", sel.to_markdown(index=False), ""]

    (out / "deep_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main(*sys.argv[1:])
