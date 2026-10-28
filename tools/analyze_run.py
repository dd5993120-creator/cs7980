"""Analyse a downloaded Colab run.

    python tools/analyze_run.py data/outputs/g4_bf16

Reads <run>/greedy, <run>/samples and <run>/mc_<level> (whichever exist) and
writes <run>/analysis/: CSV tables and summary.md. Numbers only; no claims.

Definitions
    named      a true author the answer names at L1-L4 (the paper's matcher)
    labels     team rules on our data: recalled = default HR2 <= 0.5; otherwise
               known / notsure / unknown = true list chosen in >=3 / 2 / <=1 of
               the 4 option orders of the real-team test
    reachable  a true author the default answer misses but at least one of the
               50 samples names
    chance     the same counts for the authors of 3 partner papers (same field
               and citation bin) against this paper's samples
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gts.coverage import author_table, coverage_curve  # noqa: E402
from gts.data import BIN_LABELS, load_papers, load_shipped_greedy  # noqa: E402
from gts.generate import read_chunks  # noqa: E402
from gts.matching import score_answer  # noqa: E402
from gts.mc import read_items  # noqa: E402
from gts.paths import MC_ITEMS_DIR  # noqa: E402

KS = (1, 5, 10, 20, 50)


def per100(x: float, n: int) -> float:
    return round(100 * x / n, 2) if n else float("nan")


def greedy_scores(papers: pd.DataFrame, greedy: dict[str, str]) -> pd.DataFrame:
    rows = []
    for rec in papers.itertuples(index=False):
        s = score_answer(rec.gt_str, greedy.get(rec.paper_id, ""))
        rows.append({"paper_id": rec.paper_id, "citation_bin": rec.citation_bin, "field": rec.field,
                     "m": s["n_match"], "h": s["n_hal"], "u": s["n_unretr"], "hr2": s["hr2"],
                     "n_gt": s["n_gt_authors"], "n_pred": s["n_pred_authors"]})
    return pd.DataFrame(rows)


def mc_results(run: Path, level: str) -> pd.DataFrame | None:
    out_dir = run / f"mc_{level}"
    if not out_dir.exists():
        return None
    items = pd.DataFrame(read_items(MC_ITEMS_DIR / f"{level}.jsonl.gz"))
    res = pd.DataFrame(read_chunks(out_dir))
    if len(res) != len(items):
        print(f"skipping {level}: only {len(res)} of {len(items)} items scored so far")
        return None
    df = items[["item_id", "paper_id", "condition", "order", "correct_label"]].merge(res, on="item_id")
    df["chosen"] = df["letter_logprobs"].map(lambda d: max(d, key=d.get) if d else None)
    df["correct"] = df["chosen"] == df["correct_label"]
    df["level"] = level
    return df


def orders_won(mc: pd.DataFrame) -> pd.DataFrame:
    return (mc.groupby(["level", "condition", "paper_id"])
              .agg(a=("correct", "sum"), n_orders=("correct", "size")).reset_index())


def label(hr2: float, a: float) -> str:
    if hr2 <= 0.5:
        return "recalled"
    if np.isnan(a):
        return "unscored"
    return "known" if a >= 3 else "notsure" if a == 2 else "unknown"


def main(run_dir: str):
    t0 = time.time()
    run = Path(run_dir)
    out = run / "analysis"
    out.mkdir(exist_ok=True)
    papers = load_papers()
    lines = [f"# Run analysis: {run.name}", ""]
    meta_path = run / "run_meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        lines += [f"Model {meta.get('model')} ({meta.get('precision')}) on {meta.get('gpu_name')}, "
                  f"vLLM {meta.get('vllm')}.", ""]

    # 1. default answers
    greedy = {r["paper_id"]: r["texts"][0] for r in read_chunks(run / "greedy")}
    g = greedy_scores(papers, greedy)
    g.to_csv(out / "greedy_scores.csv", index=False)
    shipped = load_shipped_greedy().set_index("paper_id")
    same = np.mean([greedy[p].strip() == str(shipped.at[p, "generated_text"]).strip() for p in greedy])
    lines += ["## Default (greedy) answers", "",
              f"- {len(g)} papers; true authors named: {g.m.sum()} ({per100(g.m.sum(), len(g))} per 100 papers); "
              f"paper's own bf16 run: {shipped.n_match.sum()}.",
              f"- Papers with HR2 <= 0.5 (recalled): {(g.hr2 <= 0.5).sum()} (paper's run: {(shipped.hr2 <= 0.5).sum()}).",
              f"- Mean HR2 {g.hr2.mean():.4f} (paper's run {shipped.hr2.mean():.4f}); "
              f"answers identical to the paper's run: {same:.1%}.", ""]

    # 2. labels from the real-team test
    mcs = [df for df in (mc_results(run, "real_team"), mc_results(run, "own_sample")) if df is not None]
    won = orders_won(pd.concat(mcs)) if mcs else pd.DataFrame(columns=["level", "condition", "paper_id", "a"])
    std = won[won.condition == "standard"].pivot(index="paper_id", columns="level", values="a")
    g = g.merge(std, left_on="paper_id", right_index=True, how="left")
    for level in ("real_team", "own_sample"):
        if level in g:
            g[f"label_{level}"] = [label(h, a) for h, a in zip(g.hr2, g[level])]
    if mcs:
        lines += ["## Multiple choice", "",
                  "| level | condition | papers | per-question accuracy | right in >= 3 of 4 orders |",
                  "|---|---|---|---|---|"]
        for df in mcs:
            w = orders_won(df)
            for cond, sub in df.groupby("condition"):
                ww = w[w.condition == cond]
                lines.append(f"| {df.level.iloc[0]} | {cond} | {ww.paper_id.nunique()} | "
                             f"{sub.correct.mean():.3f} | {(ww.a >= 3).mean():.3f} |")
        lines += ["", "Chance: 0.25 per question; about 0.05 for >= 3 of 4 orders.", ""]
        for level in ("real_team", "own_sample"):
            col = f"label_{level}"
            if col in g:
                counts = g[col].value_counts()
                lines.append(f"- Labels from {level}: " + ", ".join(
                    f"{k} {counts.get(k, 0)}" for k in ("recalled", "known", "notsure", "unknown", "unscored")))
        if "label_real_team" in g and "label_own_sample" in g:
            ct = pd.crosstab(g.label_real_team, g.label_own_sample)
            ct.to_csv(out / "labels_real_team_vs_own_sample.csv")
            lines += ["", "Real-team label (rows) vs own-sample label (columns):", "", ct.to_markdown()]
        lines.append("")
        pd.concat(mcs).drop(columns=["letter_logprobs"]).to_csv(out / "mc_items_scored.csv.gz", index=False)
    g.to_csv(out / "papers.csv", index=False)

    # 3. coverage from the samples
    if (run / "samples").exists():
        samples = {r["paper_id"]: r["texts"] for r in read_chunks(run / "samples")}
        t1 = time.time()
        table = author_table(papers, samples, greedy, n_partners=3, topic=True)
        table.to_csv(out / "author_hits.csv.gz", index=False)
        print(f"author table: {len(table)} rows in {time.time() - t1:.0f}s")
        curve = coverage_curve(table, ks=KS)
        curve.to_csv(out / "coverage_by_bin.csv", index=False)
        n_texts = sum(len(t) for t in samples.values())
        distinct = np.mean([len(set(t)) for t in samples.values()])
        lines += ["## Sampling", "",
                  f"- {len(samples)} papers, {n_texts} sampled lists; on average {distinct:.1f} distinct lists of 50.", ""]

        true = table[table.kind == "true"].copy()
        true["reach"] = (~true.in_default) & (true.hits >= 1)
        p_chance = {kind: table[table.kind == kind].groupby("paper_id")["hits"].apply(lambda h: (h >= 1).mean())
                    for kind in ("chance", "chance_topic")}
        per_paper = true.groupby("paper_id").agg(k=("hits", "size"), named=("in_default", "sum"),
                                                  reach=("reach", "sum"),
                                                  missed=("in_default", lambda s: (~s).sum()))
        per_paper["chance_reach"] = per_paper.missed * p_chance["chance"].reindex(per_paper.index).fillna(0)
        per_paper["topic_reach"] = per_paper.missed * p_chance["chance_topic"].reindex(per_paper.index).fillna(0)
        per_paper = per_paper.join(g.set_index("paper_id")[["citation_bin"] + [c for c in g if c.startswith("label_")]])
        per_paper.to_csv(out / "reach_by_paper.csv")

        def net_ci(sub, baseline="chance_reach", n_boot=1000):
            """95% bootstrap interval of net reachable authors per 100 papers, resampling papers."""
            net = (sub.reach - sub[baseline]).to_numpy()
            gen = np.random.default_rng(0)
            boots = [100 * net[gen.integers(0, len(net), len(net))].mean() for _ in range(n_boot)]
            return f"[{np.percentile(boots, 2.5):.2f}, {np.percentile(boots, 97.5):.2f}]"

        def unit_rows(frame, by):
            rows = []
            for key, sub in frame.groupby(by, sort=False, observed=True):
                n = len(sub)
                rows.append({by: key, "papers": n,
                             "named_by_default_per100": per100(sub.named.sum(), n),
                             "reachable_per100": per100(sub.reach.sum(), n),
                             "chance_reachable_per100": per100(sub.chance_reach.sum(), n),
                             "net_reachable_per100": per100(sub.reach.sum() - sub.chance_reach.sum(), n),
                             "net_95ci": net_ci(sub),
                             "topic_chance_per100": per100(sub.topic_reach.sum(), n),
                             "net_of_topic_per100": per100(sub.reach.sum() - sub.topic_reach.sum(), n),
                             "net_of_topic_95ci": net_ci(sub, "topic_reach"),
                             "papers_with_reachable_author": int((sub.reach > 0).sum())})
            return pd.DataFrame(rows)

        overall = unit_rows(per_paper.assign(all="all"), "all")
        lines += ["### True authors per 100 papers (overall)", "", overall.to_markdown(index=False), ""]

        by_bin = unit_rows(per_paper.assign(citation_bin=pd.Categorical(per_paper.citation_bin, BIN_LABELS)
                                            ).sort_values("citation_bin"), "citation_bin")
        by_bin.to_csv(out / "reach_by_bin.csv", index=False)
        lines += ["### True authors per 100 papers (all papers)", "", by_bin.to_markdown(index=False), ""]
        if "label_real_team" in per_paper:
            by_label = unit_rows(per_paper, "label_real_team")
            by_label.to_csv(out / "reach_by_label.csv", index=False)
            lines += ["### True authors per 100 papers, by real-team label", "", by_label.to_markdown(index=False), ""]
        k50 = curve[curve.k == 50].set_index("citation_bin").reindex(BIN_LABELS)
        lines += ["### Coverage@50 per true author", "",
                  k50[["true", "chance", "net", "chance_topic", "net_topic"]].round(4).to_markdown(), ""]

    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1])
