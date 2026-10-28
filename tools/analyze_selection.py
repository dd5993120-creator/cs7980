"""Selection on the candidate pools: does P(True) beat frequency?

    python tools/analyze_selection.py data/outputs/g4_bf16

Needs <run>/ptrue (scores), data/mc_items/ptrue_candidates.jsonl.gz, and
<run>/analysis/papers.csv from analyze_run.py. Writes
<run>/analysis/selection_summary.md and selection_by_group.csv.

Descriptive only: no threshold is tuned, so nothing here touches a test split.
    pairwise AUC  within one paper, the share of (true, false) candidate pairs
                  in which the true one scores higher (ties count half)
    top-1         the share of papers whose highest-scoring candidate is true,
                  among papers with at least one true candidate in the pool
    per 100       true / false names among the top k candidates, per 100 papers
"""

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gts.generate import read_chunks  # noqa: E402
from gts.mc import read_items  # noqa: E402
from gts.paths import MC_ITEMS_DIR  # noqa: E402

FLOOR = -30.0  # log-probability given to "yes"/"no" when absent from the top 20
BIN_GROUP = {b: "0-32" for b in ["0-2", "3-4", "5-8", "9-16", "17-32"]} | \
            {b: "33-512" for b in ["33-64", "65-128", "129-256", "257-512"]} | \
            {b: "513+" for b in ["513-1024", "1025-2048", "2049-4096", "4097+"]}


def p_true(scores: dict) -> float:
    y, n = scores.get("yes", FLOOR), scores.get("no", FLOOR)
    return 1.0 / (1.0 + math.exp(n - y))


def pairwise_auc(group: pd.DataFrame, col: str) -> tuple[float, int]:
    t = group.loc[group.is_true, col].to_numpy()
    f = group.loc[~group.is_true, col].to_numpy()
    if len(t) == 0 or len(f) == 0:
        return 0.0, 0
    wins = (t[:, None] > f[None, :]).sum() + 0.5 * (t[:, None] == f[None, :]).sum()
    return float(wins), len(t) * len(f)


def summarise(df: pd.DataFrame, n_papers: int, named_default: float) -> dict:
    row = {"papers": n_papers, "default_named_per100": round(100 * named_default / n_papers, 2)}
    for col in ("count", "p_true"):
        wins = pairs = 0
        for _, g in df.groupby("paper_id"):
            w, p = pairwise_auc(g, col)
            wins, pairs = wins + w, pairs + p
        row[f"auc_{col}"] = round(wins / pairs, 3) if pairs else float("nan")
    with_true = df.groupby("paper_id").is_true.transform("any")
    sub = df[with_true]
    for col in ("count", "p_true"):
        top = sub.sort_values([col, "rank"], ascending=[False, True]).groupby("paper_id").head(1)
        row[f"top1_{col}"] = round(top.is_true.mean(), 3) if len(top) else float("nan")
        for k in (1, 3):
            topk = df.sort_values([col, "rank"], ascending=[False, True]).groupby("paper_id").head(k)
            row[f"top{k}_{col}_true_per100"] = round(100 * topk.is_true.sum() / n_papers, 2)
            row[f"top{k}_{col}_false_per100"] = round(100 * (~topk.is_true).sum() / n_papers, 2)
    row["papers_with_true_candidate"] = int(sub.paper_id.nunique())
    return row


def main(run_dir: str):
    run = Path(run_dir)
    cands = pd.DataFrame(read_items(MC_ITEMS_DIR / "ptrue_candidates.jsonl.gz"))
    scores = pd.DataFrame(read_chunks(run / "ptrue"))
    df = cands.merge(scores, on="record_id", how="inner")
    df["p_true"] = df["yes_no"].map(p_true)
    papers = pd.read_csv(run / "analysis" / "papers.csv", dtype={"paper_id": str})
    df = df.merge(papers[["paper_id", "citation_bin", "label_real_team", "m"]], on="paper_id")
    df["bin_group"] = df.citation_bin.map(BIN_GROUP)
    papers["bin_group"] = papers.citation_bin.map(BIN_GROUP)

    rows = []
    for name, sel in [("all", lambda d: d), *[(f"label={lab}", (lambda lab: lambda d: d[d.label_real_team == lab])(lab))
                                              for lab in ("recalled", "known", "notsure", "unknown")],
                      *[(f"bins {b}", (lambda b: lambda d: d[d.bin_group == b])(b)) for b in ("0-32", "33-512", "513+")]]:
        sub_c, sub_p = sel(df), sel(papers)
        if len(sub_p):
            rows.append({"group": name, **summarise(sub_c, len(sub_p), sub_p.m.sum())})
    table = pd.DataFrame(rows)
    table.to_csv(run / "analysis" / "selection_by_group.csv", index=False)
    lines = ["# Selection on the candidate pools", "",
             f"{len(df)} candidates scored ({df.paper_id.nunique()} papers, top {df['rank'].max() + 1} per paper); "
             f"{int(df.is_true.sum())} are true authors.", "",
             "Pairwise AUC (true vs false candidates of the same paper), by score:", "",
             table[["group", "papers", "papers_with_true_candidate", "auc_count", "auc_p_true",
                    "top1_count", "top1_p_true"]].to_markdown(index=False), "",
             "True and false names among the top k candidates, per 100 papers (default answer for reference):", "",
             table[["group", "default_named_per100", "top1_count_true_per100", "top1_p_true_true_per100",
                    "top3_count_true_per100", "top3_p_true_true_per100", "top3_count_false_per100",
                    "top3_p_true_false_per100"]].to_markdown(index=False), ""]
    (run / "analysis" / "selection_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1])
