"""Session 3: the whole recognition ladder (L0-L5), catch trials, the yes/no and
likelihood formats, and likelihood scores for selection.

    python tools/analyze_session3.py data/outputs/g4_bf16 data/outputs/s2 data/outputs/s3

Reads the multiple-choice results of all three sessions and writes
data/outputs/g4_bf16/analysis/session3_summary.md with CSVs (session3_*.csv).
Sections whose outputs are missing are skipped. Numbers only; no claims.

Definitions
    recognised      the true list chosen in >= 3 of the 4 cyclic orders (the team rule)
    chance-corrected accuracy   (accuracy - 1/4) / (3/4), per question
    pairwise AUC    the share of (true, wrong) pairs within a paper in which the true
                    option scores higher, pooled over papers, ties counting 1/2. In
                    multiple choice an option's score is its letter's probability
                    averaged over the four orders, so position bias cancels.
    d', c           yes/no: z(hit rate) - z(false-alarm rate), answering Yes when
                    P(Yes) > P(No); c = -(z(H) + z(F)) / 2, negative means a bias to Yes
    hard-known      recognised at every level L1-L5 (L3t included) and not recalled
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyze_selection import p_true  # noqa: E402
from gts.data import BIN_LABELS  # noqa: E402
from gts.generate import read_chunks  # noqa: E402
from gts.mc import read_items  # noqa: E402
from gts.paths import MC_ITEMS_DIR  # noqa: E402
from gts.prompts import OPTION_LABELS  # noqa: E402
from gts.splits import load_split  # noqa: E402

# (ladder level, item file, session whose outputs hold it)
LADDER = [("L0", "random_field", 3), ("L1", "real_team", 1), ("L2", "real_team_year", 2),
          ("L3", "similar_title", 3), ("L3t", "similar_topic", 3), ("L4", "collab_slot", 3),
          ("L4r", "collab_swap", 3), ("L5", "own_sample", 1)]
HARD = ("L1", "L2", "L3", "L3t", "L4", "L5")
LEVEL_OF = {name: lvl for lvl, name, _ in LADDER}


def md(df: pd.DataFrame) -> str:
    return df.to_markdown(index=False, floatfmt=".3f")


def mc_frame(run: Path, name: str) -> pd.DataFrame | None:
    out = run / f"mc_{name}"
    if not out.exists():
        return None
    items = pd.DataFrame(read_items(MC_ITEMS_DIR / f"{name}.jsonl.gz"))
    res = pd.DataFrame(read_chunks(out))
    if len(res) < len(items):
        print(f"skipping {name}: {len(res)} of {len(items)} items scored")
        return None
    keep = [c for c in ("item_id", "paper_id", "condition", "order", "correct_label", "none_label") if c in items]
    df = items[keep].merge(res[["item_id", "letter_logprobs"]], on="item_id")
    df["chosen"] = df["letter_logprobs"].map(lambda d: max(d, key=d.get) if d else None)
    df["correct"] = df["chosen"] == df["correct_label"]
    return df


def option_probs(df: pd.DataFrame) -> pd.DataFrame:
    """Per (condition, paper), each base option's letter probability averaged over the
    four orders. Base option i sits at letter (i + order) % 4; option 0 is the true list."""
    rows = []
    for rec in df.itertuples(index=False):
        lp = rec.letter_logprobs or {}
        rows.append((rec.condition, rec.paper_id,
                     *(float(np.exp(lp.get(OPTION_LABELS[(i + rec.order) % 4], -np.inf))) for i in range(4))))
    out = pd.DataFrame(rows, columns=["condition", "paper_id", 0, 1, 2, 3])
    return out.groupby(["condition", "paper_id"]).mean().reset_index()


def auc(true_scores, wrong_scores) -> float:
    """Pooled within-paper AUC: true_scores (n,), wrong_scores (n, m); NaN rows are dropped."""
    t = np.asarray(true_scores, dtype=float)[:, None]
    w = np.asarray(wrong_scores, dtype=float)
    ok = ~np.isnan(t[:, 0]) & ~np.isnan(w).any(axis=1)
    t, w = t[ok], w[ok]
    return float(((t > w).sum() + 0.5 * (t == w).sum()) / w.size) if w.size else float("nan")


def pooled_auc(df: pd.DataFrame, col: str) -> float:
    """Candidate AUC as in analyze_selection: (true, false) pairs within each paper, pooled."""
    wins = pairs = 0.0
    for _, g in df.groupby("paper_id"):
        t, f = g.loc[g.is_true, col].to_numpy(), g.loc[~g.is_true, col].to_numpy()
        if len(t) and len(f):
            wins += (t[:, None] > f[None, :]).sum() + 0.5 * (t[:, None] == f[None, :]).sum()
            pairs += len(t) * len(f)
    return wins / pairs if pairs else float("nan")


def paired_boot(df: pd.DataFrame, col: str, base: str, n_boot: int = 2000) -> dict:
    """Bootstrap over papers of (col - base) for the pooled AUC and the top-1 share."""
    per = []
    for pid, g in df.groupby("paper_id"):
        t, f = g[g.is_true], g[~g.is_true]
        row = {"pairs": len(t) * len(f), "has_true": len(t) > 0}
        for c in (col, base):
            tv, fv = t[c].to_numpy(), f[c].to_numpy()
            row[f"wins_{c}"] = float((tv[:, None] > fv[None, :]).sum() + 0.5 * (tv[:, None] == fv[None, :]).sum())
            top = g.sort_values([c, "rank"], ascending=[False, True]).iloc[0]
            row[f"top_{c}"] = float(top.is_true)
        per.append(row)
    per = pd.DataFrame(per)
    gen = np.random.default_rng(0)
    auc_d, top_d = [], []
    has = per[per.has_true]
    for _ in range(n_boot):
        s = per.iloc[gen.integers(0, len(per), len(per))]
        auc_d.append((s[f"wins_{col}"].sum() - s[f"wins_{base}"].sum()) / s.pairs.sum())
        h = has.iloc[gen.integers(0, len(has), len(has))]
        top_d.append(h[f"top_{col}"].mean() - h[f"top_{base}"].mean())
    auc_gain = (per[f"wins_{col}"].sum() - per[f"wins_{base}"].sum()) / per.pairs.sum()
    top_gain = has[f"top_{col}"].mean() - has[f"top_{base}"].mean()
    lo, hi = np.percentile(auc_d, [2.5, 97.5])
    tlo, thi = np.percentile(top_d, [2.5, 97.5])
    return {"AUC gain": auc_gain, "AUC 95% CI": f"[{lo:+.3f}, {hi:+.3f}]",
            "top-1 gain": top_gain, "top-1 95% CI": f"[{tlo:+.3f}, {thi:+.3f}]"}


def top1(df: pd.DataFrame, col: str) -> float:
    """Among papers with a true candidate, the share whose top-scored candidate is true (ties: rank)."""
    sub = df[df.groupby("paper_id").is_true.transform("any")]
    top = sub.sort_values([col, "rank"], ascending=[False, True]).groupby("paper_id").head(1)
    return float(top.is_true.mean()) if len(top) else float("nan")


def label(hr2: float, won: float) -> str:
    if hr2 <= 0.5:
        return "recalled"
    if np.isnan(won):
        return "unscored"
    return "known" if won >= 3 else "notsure" if won == 2 else "unknown"


# --------------------------------------------------------------------------- ladder

def ladder(runs: dict[int, Path], papers: pd.DataFrame, lines: list[str], out: Path) -> pd.DataFrame:
    rows, bins, won_cols = [], [], {}
    for lvl, name, session in LADDER:
        df = mc_frame(runs[session], name)
        if df is None:
            continue
        probs = option_probs(df)
        won = df.groupby(["condition", "paper_id"]).correct.sum()
        for condition, g in df.groupby("condition"):
            p = probs[probs.condition == condition]
            acc = g.correct.mean()
            rows.append({"level": lvl, "distractors": name, "condition": condition,
                         "papers": g.paper_id.nunique(), "accuracy": acc, "chance_corrected": (acc - 0.25) / 0.75,
                         ">=3 of 4": (won[condition] >= 3).mean(),
                         "AUC": auc(p[0], p[[1, 2, 3]])})
        std = won["standard"]
        won_cols[lvl] = std
        p = probs[probs.condition == "standard"].set_index("paper_id")
        bin_of = papers.set_index("paper_id").citation_bin
        for b, ids in p.groupby(bin_of.reindex(p.index)).groups.items():
            bins.append({"level": lvl, "citation_bin": b, ">=3 of 4": (std.reindex(ids) >= 3).mean(),
                         "AUC": auc(p.loc[ids, 0], p.loc[ids, [1, 2, 3]])})
    table = pd.DataFrame(rows)
    table.to_csv(out / "session3_ladder.csv", index=False)
    lines += ["## 1. The recognition ladder", "",
              "Per-question accuracy (chance 0.25), chance-corrected accuracy, the share of papers whose true list is",
              "chosen in >= 3 of 4 orders (guessing: about 0.05), and the pairwise AUC of the true list against the",
              "three wrong ones (chance 0.5). Controls on the 1,023-paper subset.", "", md(table), ""]
    if bins:
        by_bin = pd.DataFrame(bins)
        by_bin["citation_bin"] = pd.Categorical(by_bin.citation_bin, BIN_LABELS, ordered=True)
        by_bin = by_bin.sort_values(["level", "citation_bin"])
        by_bin.to_csv(out / "session3_ladder_by_bin.csv", index=False)
        for metric in (">=3 of 4", "AUC"):
            wide = by_bin.pivot(index="citation_bin", columns="level", values=metric).reset_index()
            lines += [f"**{metric} by citation bin** (standard condition):", "", md(wide), ""]
    return pd.DataFrame(won_cols)


def labels_section(won: pd.DataFrame, papers: pd.DataFrame, lines: list[str], out: Path) -> pd.DataFrame:
    p = papers.set_index("paper_id")
    labs = pd.DataFrame({lvl: [label(h, w) for h, w in zip(p.hr2, won[lvl].reindex(p.index))]
                         for lvl in won.columns}, index=p.index)
    counts = labs.apply(lambda s: s.value_counts()).fillna(0).astype(int).T.reset_index(names="level")
    lines += ["## 2. Labels at each level", "",
              "Team rules: recalled = default HR2 <= 0.5; otherwise known / notsure / unknown = the true list chosen",
              "in >= 3 / 2 / <= 1 of the 4 orders.", "", md(counts), ""]
    if all(lvl in labs for lvl in HARD):
        labs["hard_known"] = (labs[list(HARD)] == "known").all(axis=1)
        by_bin = labs.join(p.citation_bin).groupby("citation_bin").hard_known.agg(["sum", "mean"])
        by_bin = by_bin.reindex(BIN_LABELS).reset_index()
        lines += [f"**Hard-known** (known at every level {', '.join(HARD)}): {int(labs.hard_known.sum())} papers "
                  f"({labs.hard_known.mean():.1%}); known at L1: {int((labs.L1 == 'known').sum())}.", "",
                  md(by_bin.rename(columns={"sum": "hard-known papers", "mean": "share"})), ""]
    recog = won.reindex(p.index) >= 3
    rows = []
    for lvl in won.columns:
        ct = pd.crosstab(p.hr2 <= 0.5, recog[lvl])
        rows.append({"level": lvl, "recalled & recognised": int(ct.loc[True, True]) if True in ct.columns else 0,
                     "recalled, not recognised": int(ct.loc[True, False]) if False in ct.columns else 0,
                     "not recalled, recognised": int(ct.loc[False, True]) if True in ct.columns else 0,
                     "neither": int(ct.loc[False, False]) if False in ct.columns else 0})
    lines += ["**Reliability check** (proposal): recalled papers should almost all be recognised.", "",
              md(pd.DataFrame(rows)), ""]
    labs.reset_index(names="paper_id").to_csv(out / "session3_labels.csv", index=False)
    return labs


# --------------------------------------------------------------------------- catch trials

def catch_section(run3: Path, papers: pd.DataFrame, labs: pd.DataFrame, lines: list[str], out: Path) -> None:
    df = mc_frame(run3, "catch")
    if df is None:
        return
    df["picked_none"] = df.chosen == df.none_label
    wt, ct = df[df.condition == "with_true"], df[df.condition == "catch"]
    per_q = pd.DataFrame([
        {"items": "true list present", "true list chosen": wt.correct.mean(), "None chosen": wt.picked_none.mean(),
         "a wrong list chosen": 1 - wt.correct.mean() - wt.picked_none.mean()},
        {"items": "true list removed (catch)", "true list chosen": np.nan, "None chosen": ct.picked_none.mean(),
         "a wrong list chosen": 1 - ct.picked_none.mean()},
    ])
    won = df.groupby(["condition", "paper_id"]).correct.sum().unstack(0)
    paper = pd.DataFrame({"hit": won["with_true"] >= 3, "reject": won["catch"] >= 3})
    paper["both"] = paper.hit & paper.reject
    paper = paper.join(labs["L1"].rename("L1 label")).join(papers.set_index("paper_id").citation_bin)
    by_label = paper.groupby("L1 label")[["hit", "reject", "both"]].mean().join(
        paper.groupby("L1 label").size().rename("papers")).reset_index()
    by_bin = paper.groupby("citation_bin")[["hit", "reject", "both"]].mean().reindex(BIN_LABELS).reset_index()
    paper.reset_index(names="paper_id").to_csv(out / "session3_catch_by_paper.csv", index=False)
    lines += ["## 3. Catch trials (L1 distractors, with \"None of the above\")", "",
              "Per question:", "", md(per_q), "",
              "Per paper, >= 3 of 4 orders: hit = the true list chosen when present; reject = None chosen when the",
              "true list is removed; both = a recognition that survives the catch trial.", "",
              md(by_label), "", md(by_bin), ""]


# --------------------------------------------------------------------------- yes/no

def yesno_section(run3: Path, papers: pd.DataFrame, lines: list[str], out: Path) -> None:
    if not (run3 / "yesno_lists").exists():
        return
    recs = pd.DataFrame(read_items(MC_ITEMS_DIR / "yesno_lists.jsonl.gz"))[
        ["record_id", "paper_id", "level", "list_index", "is_true"]]
    df = recs.merge(pd.DataFrame(read_chunks(run3 / "yesno_lists")), on="record_id")
    df["p_yes"] = df.yes_no.map(p_true)
    df["yes"] = df.p_yes > 0.5
    rows, bins = [], []
    for level, g in df.groupby("level"):
        n_t, n_f = int(g.is_true.sum()), int((~g.is_true).sum())
        hit, fa = g[g.is_true].yes.mean(), g[~g.is_true].yes.mean()
        z_h = norm.ppf(np.clip(hit, 0.5 / n_t, 1 - 0.5 / n_t))
        z_f = norm.ppf(np.clip(fa, 0.5 / n_f, 1 - 0.5 / n_f))
        wide = g.pivot(index="paper_id", columns="list_index", values="p_yes")
        rows.append({"level": LEVEL_OF.get(level, level), "distractors": level, "papers": len(wide),
                     "hit rate": hit, "false-alarm rate": fa, "d'": z_h - z_f, "c": -(z_h + z_f) / 2,
                     "AUC": auc(wide[0], wide[[1, 2, 3]]),
                     "Yes to the true list only": ((wide[0] > 0.5) & (wide[[1, 2, 3]] <= 0.5).all(axis=1)).mean()})
        bin_of = papers.set_index("paper_id").citation_bin.reindex(wide.index)
        for b, ids in wide.groupby(bin_of).groups.items():
            bins.append({"level": LEVEL_OF.get(level, level), "citation_bin": b,
                         "AUC": auc(wide.loc[ids, 0], wide.loc[ids, [1, 2, 3]])})
    table = pd.DataFrame(rows).sort_values("level")
    table.to_csv(out / "session3_yesno.csv", index=False)
    wide_bins = pd.DataFrame(bins).pivot(index="citation_bin", columns="level", values="AUC").reindex(BIN_LABELS)
    lines += ["## 4. Yes/no recognition (one author list per question)", "",
              "Answer = Yes when P(Yes) > P(No). AUC uses P(Yes) within each paper.", "", md(table), "",
              "**AUC by citation bin:**", "", md(wide_bins.reset_index()), ""]


# --------------------------------------------------------------------------- list likelihood

def likelihood_section(run3: Path, papers: pd.DataFrame, lines: list[str], out: Path) -> None:
    if not (run3 / "score_lists").exists():
        return
    recs = pd.DataFrame(read_items(MC_ITEMS_DIR / "score_lists.jsonl.gz"))[
        ["record_id", "paper_id", "level", "condition", "list_index"]]
    df = recs.merge(pd.DataFrame(read_chunks(run3 / "score_lists")), on="record_id")

    def wide(condition, value):
        return df[df.condition == condition].pivot_table(index=["level", "paper_id"], columns="list_index", values=value)

    shown, hidden, ntok = wide("standard", "logprob"), wide("hidden", "logprob"), wide("standard", "n_tokens")
    scores = {"log P(list | title)": shown, "per token": shown / ntok, "PMI (title shown - hidden)": shown - hidden}
    rows, bins = [], []
    for level in sorted(df.level.unique()):
        for name, s in scores.items():
            s_l = s.loc[level]
            rows.append({"level": LEVEL_OF.get(level, level), "distractors": level, "score": name, "papers": len(s_l),
                         "AUC": auc(s_l[0], s_l[[1, 2, 3]]),
                         "true list scores highest": (s_l[0] > s_l[[1, 2, 3]].max(axis=1)).mean()})
        pmi = scores["PMI (title shown - hidden)"].loc[level]
        bin_of = papers.set_index("paper_id").citation_bin.reindex(pmi.index)
        for b, ids in pmi.groupby(bin_of).groups.items():
            bins.append({"level": LEVEL_OF.get(level, level), "citation_bin": b,
                         "PMI AUC": auc(pmi.loc[ids, 0], pmi.loc[ids, [1, 2, 3]])})
    table = pd.DataFrame(rows)
    table.to_csv(out / "session3_likelihood.csv", index=False)
    wide_bins = pd.DataFrame(bins).pivot(index="citation_bin", columns="level", values="PMI AUC").reindex(BIN_LABELS)
    lines += ["## 5. Likelihood recognition (each list as the answer to the open question)", "",
              md(table), "", "**PMI AUC by citation bin:**", "", md(wide_bins.reset_index()), ""]


# --------------------------------------------------------------------------- selection

def selection_section(run1: Path, run3: Path, lines: list[str], out: Path) -> None:
    if not (run3 / "score_cands").exists():
        return
    from sklearn.linear_model import LogisticRegression

    cands = pd.DataFrame(read_items(MC_ITEMS_DIR / "ptrue_candidates.jsonl.gz"))[
        ["record_id", "paper_id", "rank", "count", "is_true"]]
    cands = cands.merge(pd.DataFrame(read_chunks(run1 / "ptrue"))[["record_id", "yes_no"]], on="record_id")
    cands["p_true"] = cands.yes_no.map(p_true)
    lp = dict((r["record_id"], r["logprob"]) for r in read_chunks(run3 / "score_cands"))
    key = cands.paper_id + "|" + cands["rank"].astype(str)
    cands["fwd"] = ("author|title|" + key).map(lp)
    cands["prior"] = ("author|none|" + key).map(lp)
    cands["pmi"] = cands.fwd - cands.prior
    cands["rev"] = ("title|author|" + key).map(lp)
    cands["rev_pmi"] = cands.rev - ("title|none|" + cands.paper_id).map(lp)
    cands["log_count"] = np.log(cands["count"])
    cands["logit_p_true"] = np.log(cands.p_true.clip(1e-12, 1 - 1e-12) / (1 - cands.p_true.clip(1e-12, 1 - 1e-12)))
    split = load_split()
    cands["split"] = cands.paper_id.map(split)
    names = {"count": "sample count (of 50)", "p_true": "P(True)", "fwd": "log P(a | title)",
             "prior": "log P(a), no title", "pmi": "PMI: log P(a | title) - log P(a)",
             "rev": "log P(title | a), reference list", "rev_pmi": "log P(title | a) - log P(title)"}
    rows = []
    for split_name, sub in (("all", cands), ("test", cands[cands.split == "test"])):
        for col, label_ in names.items():
            rows.append({"papers": split_name, "score": label_, "AUC": pooled_auc(sub, col), "top-1 true": top1(sub, col)})
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    tune, test = cands[cands.split == "tune"], cands[cands.split == "test"].copy()
    combos = {"count + PMI": ["log_count", "pmi"], "count + reverse": ["log_count", "rev_pmi"],
              "count + P(True)": ["log_count", "logit_p_true"],
              "count + PMI + reverse + P(True)": ["log_count", "pmi", "rev_pmi", "logit_p_true"]}
    for name, combo in combos.items():
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(tune[combo], tune.is_true)
        test[name] = model.decision_function(test[combo])
        rows.append({"papers": "test", "score": "logistic on tune: " + name,
                     "AUC": pooled_auc(test, name), "top-1 true": top1(test, name)})
    table = pd.DataFrame(rows)
    table.to_csv(out / "session3_selection.csv", index=False)
    boot = pd.DataFrame([paired_boot(test, name, "count") | {"combination": name} for name in combos])
    boot.to_csv(out / "session3_selection_boot.csv", index=False)

    # the same number of names as the ">= 7 of 50 samples" rule, kept by each score across all test papers
    n_papers = test.paper_id.nunique()
    budget = int((test["count"] >= 7).sum())
    keep_rows = []
    for col in ["count", "p_true", "pmi", "rev_pmi", *combos]:
        kept = test.sort_values([col, "count", "rank"], ascending=[False, False, True]).head(budget)
        keep_rows.append({"score": names.get(col, col), "names kept": budget,
                          "true per 100 papers": 100 * kept.is_true.sum() / n_papers,
                          "false per 100 papers": 100 * (~kept.is_true).sum() / n_papers})
    keep = pd.DataFrame(keep_rows)
    keep.to_csv(out / "session3_selection_budget.csv", index=False)
    lines += ["## 6. Selection scores for the top-10 sampled candidates", "",
              "Pooled within-paper AUC (true vs false candidates) and the share of papers with a true candidate whose",
              "top-scored candidate is true. Combinations are fitted on the tune split and scored on test once.", "",
              md(table), "",
              "**Paired bootstrap on test papers** (2,000 resamples): each combination minus the count alone.", "",
              md(boot[["combination", "AUC gain", "AUC 95% CI", "top-1 gain", "top-1 95% CI"]]), "",
              f"**At a fixed budget** ({n_papers} test papers): each score keeps the same number of names as",
              f"\">= 7 of 50 samples\" ({budget}), ranked across all test papers' top-10 candidates.", "",
              md(keep), ""]


def main(run1: str, run2: str, run3: str):
    runs = {1: Path(run1), 2: Path(run2), 3: Path(run3)}
    out = runs[1] / "analysis"
    papers = pd.read_csv(out / "papers.csv", dtype={"paper_id": str})
    lines = ["# Session 3: the recognition ladder, catch trials, formats and selection scores", ""]
    won = ladder(runs, papers, lines, out)
    labs = labels_section(won, papers, lines, out)
    catch_section(runs[3], papers, labs, lines, out)
    yesno_section(runs[3], papers, lines, out)
    likelihood_section(runs[3], papers, lines, out)
    selection_section(runs[1], runs[3], lines, out)
    (out / "session3_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main(*sys.argv[1:])
