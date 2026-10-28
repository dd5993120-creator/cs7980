"""Tuned selection, compute curve, CoT, candidate MC, and robustness checks.

    python tools/analyze_session2.py data/outputs/g4_bf16 [data/outputs/s2]

The first run directory holds session 1 (greedy, 50 samples); the optional
second holds session 2 (cot/, mc_cand_mc/, mc_real_team_year/, samples_t07/).
Writes <run1>/analysis/session2_summary.md and CSV tables next to it.

Rules are tuned on the "tune" split (minimum mean HR2) and scored once on
"test". Tuning uses a fast pool-based count of (M, H, U): a selected name is a
match if it names a true author at L1-L4, and two names of one author count
once. The final test numbers are re-scored with the paper's matcher
(score_answer on the joined names), which should agree to within rounding.
"""

from __future__ import annotations

import math
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gts.coverage import author_table, coverage_at_k  # noqa: E402
from gts.data import BIN_LABELS, load_papers  # noqa: E402
from gts.generate import read_chunks  # noqa: E402
from gts.matching import display_name, gt_names, matched_gt_indices, name_key, score_answer, split_answer  # noqa: E402
from gts.mc import control_subset, read_items  # noqa: E402
from gts.paths import MC_ITEMS_DIR  # noqa: E402
from gts.prompts import split_thinking  # noqa: E402
from gts.splits import load_split  # noqa: E402

KS = (1, 2, 3, 5, 10, 20, 30, 50)


# ------------------------------------------------------------------ pools

class Pool:
    """One paper's sampled names: presence per sample, truth per name."""

    def __init__(self, texts: list[str], n_tokens: list[int], gt_list: list[str]):
        first, forms, present = {}, defaultdict(Counter), defaultdict(set)
        for i, text in enumerate(texts):
            for raw in split_answer(text):
                key = name_key(raw)
                if not key[0]:
                    continue
                present[key].add(i)
                first.setdefault(key, (i, len(first)))
                forms[key][display_name(raw)] += 1
        self.keys = sorted(present, key=lambda k: first[k])
        self.form = {k: forms[k].most_common(1)[0][0] for k in self.keys}
        self.presence = np.zeros((len(self.keys), len(texts)), dtype=bool)
        for r, k in enumerate(self.keys):
            self.presence[r, list(present[k])] = True
        self.gt_index = []
        for k in self.keys:
            hit = matched_gt_indices(gt_list, self.form[k])
            self.gt_index.append(min(hit) if hit else -1)
        self.gt_index = np.array(self.gt_index, dtype=int)
        self.n_gt = len(gt_list)
        self.cum_tokens = np.cumsum(n_tokens)

    def counts(self, k: int) -> np.ndarray:
        return self.presence[:, :k].sum(axis=1)

    def mhu(self, selected: np.ndarray) -> tuple[int, int, int]:
        hits = self.gt_index[selected]
        m = len(set(hits[hits >= 0].tolist()))
        return m, int(selected.sum()) - m, self.n_gt - m


def hr2(m, h, u):
    return (h + u) / (m + h + u)


def load_pools(run: Path, samples: dict, gt_str: dict) -> dict[str, Pool]:
    """Build every paper's pool once (about ten minutes) and cache it next to the analysis."""
    import pickle
    cache = Path(run) / "analysis" / "pools.pkl"
    if cache.exists():
        with open(cache, "rb") as f:
            return pickle.load(f)
    pools = {pid: Pool(r["texts"], r["n_tokens"], gt_names(gt_str[pid])) for pid, r in samples.items()}
    with open(cache, "wb") as f:
        pickle.dump(pools, f)
    return pools


def evaluate(pools: dict[str, Pool], ids, select) -> dict:
    """Mean HR2 and named / false names per 100 papers for one selection rule."""
    mm = hh = 0
    hrs = []
    for pid in ids:
        m, h, u = pools[pid].mhu(select(pools[pid]))
        mm, hh = mm + m, hh + h
        hrs.append(hr2(m, h, u))
    n = len(hrs)
    return {"mean_hr2": float(np.mean(hrs)), "recalled_share": float(np.mean(np.array(hrs) <= 0.5)),
            "true_per100": 100 * mm / n, "false_per100": 100 * hh / n}


def background(pools: dict[str, Pool], fields: dict[str, str]) -> dict[str, np.ndarray]:
    """Leave-one-out mean count of each name over the other papers of the same field."""
    total: dict[tuple[str, tuple], int] = defaultdict(int)
    n_field = Counter(fields[p] for p in pools)
    for pid, pool in pools.items():
        for key, c in zip(pool.keys, pool.counts(pool.presence.shape[1])):
            total[(fields[pid], key)] += int(c)
    out = {}
    for pid, pool in pools.items():
        f, own = fields[pid], pool.counts(pool.presence.shape[1])
        out[pid] = np.array([(total[(f, k)] - c) / max(n_field[f] - 1, 1) for k, c in zip(pool.keys, own)])
    return out


def tune(pools, tune_ids, rules: dict) -> tuple[str, dict]:
    scored = {name: evaluate(pools, tune_ids, fn) for name, fn in rules.items()}
    best = min(scored, key=lambda name: scored[name]["mean_hr2"])
    return best, scored[best]


def exact_scores(pools, ids, select, papers_by_id) -> dict:
    """The same rule re-scored with the paper's matcher on the joined names."""
    mm = hh = 0
    hrs = []
    for pid in ids:
        pool = pools[pid]
        names = [pool.form[k] for k, s in zip(pool.keys, select(pool)) if s]
        s = score_answer(papers_by_id[pid], "; ".join(names))
        mm, hh = mm + s["n_match"], hh + s["n_hal"]
        hrs.append(s["hr2"])
    n = len(hrs)
    return {"mean_hr2": float(np.mean(hrs)), "recalled_share": float(np.mean(np.array(hrs) <= 0.5)),
            "true_per100": 100 * mm / n, "false_per100": 100 * hh / n}


def answer_scores(answers: dict[str, str], ids, papers_by_id) -> dict:
    mm = hh = 0
    hrs = []
    for pid in ids:
        s = score_answer(papers_by_id[pid], answers.get(pid, ""))
        mm, hh = mm + s["n_match"], hh + s["n_hal"]
        hrs.append(s["hr2"])
    n = len(hrs)
    return {"mean_hr2": float(np.mean(hrs)), "recalled_share": float(np.mean(np.array(hrs) <= 0.5)),
            "true_per100": 100 * mm / n, "false_per100": 100 * hh / n}


def fmt(row: dict) -> dict:
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}


_REFUSAL = re.compile(
    r"(?i)\b(i (do not|don't|cannot|can't|am unable|am not able)|unable to (provide|find|identify)|"
    r"no (information|record|data) (on|about|for)|not (able|aware) )")


def is_refusal(answer: str) -> bool:
    """An answer that declines ("I cannot provide the authors ...") instead of naming anyone."""
    return bool(_REFUSAL.search(answer or ""))


# ------------------------------------------------------------------ main

def main(run1: str, run2: str | None = None):
    t0 = time.time()
    run1 = Path(run1)
    out = run1 / "analysis"
    out.mkdir(exist_ok=True)
    papers = load_papers()
    gt_str = dict(zip(papers.paper_id, papers.gt_str))
    fields = dict(zip(papers.paper_id, papers.field))
    bins = dict(zip(papers.paper_id, papers.citation_bin))
    split = load_split()
    tune_ids = sorted(p for p, s in split.items() if s == "tune")
    test_ids = sorted(p for p, s in split.items() if s == "test")
    labels = pd.read_csv(run1 / "analysis" / "papers.csv", dtype={"paper_id": str}).set_index("paper_id")
    lines = ["# Session 2 analysis", "",
             f"Split: {len(tune_ids)} tune / {len(test_ids)} test papers (stratified by field x citation bin).", ""]

    samples = {r["paper_id"]: r for r in read_chunks(run1 / "samples")}
    greedy = {r["paper_id"]: r for r in read_chunks(run1 / "greedy")}
    t1 = time.time()
    pools = load_pools(run1, samples, gt_str)
    print(f"pools ready in {time.time() - t1:.0f}s")
    bg = background(pools, fields)

    # --- 1. tuned selection at k = 50
    rules = {}
    for t in range(1, 51):
        rules[f"count>={t}"] = (lambda t: lambda p: p.counts(50) >= t)(t)
    for m in range(1, 11):
        rules[f"top{m}"] = (lambda m: lambda p: np.isin(np.arange(len(p.keys)),
                                                        np.argsort(-p.counts(50), kind="stable")[:m]))(m)
    bg_of = {id(p): bg[pid] for pid, p in pools.items()}  # rules receive the pool, not its id

    def bg_rule(t, a):
        return lambda p: p.counts(50) - a * bg_of[id(p)] >= t

    bg_rules = {f"count-{a}*bg>={t}": bg_rule(t, a) for t in range(1, 51) for a in (0.5, 1, 2, 4, 8)}

    best_freq, _ = tune(pools, tune_ids, {k: v for k, v in rules.items() if k.startswith("count")})
    best_top, _ = tune(pools, tune_ids, {k: v for k, v in rules.items() if k.startswith("top")})
    best_bg, _ = tune(pools, tune_ids, bg_rules)
    all_rules = rules | bg_rules
    papers_by_id = gt_str
    rows = [{"method": "default answer (greedy)", "tokens_per_paper": np.mean([greedy[p]["n_tokens"][0] for p in test_ids]),
             **answer_scores({p: greedy[p]["texts"][0] for p in greedy}, test_ids, papers_by_id)}]
    for name in (best_freq, best_top, best_bg):
        rows.append({"method": f"{name} of 50 samples (tuned)",
                     "tokens_per_paper": np.mean([pools[p].cum_tokens[49] for p in test_ids]),
                     **exact_scores(pools, test_ids, all_rules[name], papers_by_id)})
    for t in (25, 10, 5):  # fixed operating points from session 1, for reference
        rows.append({"method": f"count>={t} of 50 samples (fixed)",
                     "tokens_per_paper": np.mean([pools[p].cum_tokens[49] for p in test_ids]),
                     **exact_scores(pools, test_ids, all_rules[f"count>={t}"], papers_by_id)})

    # --- 2. CoT (session 2)
    cot = {}
    cot_dirs = [Path(run2) / d for d in ("cot_test", "cot_tune")] if run2 else []
    if any(d.exists() for d in cot_dirs):
        recs = {r["paper_id"]: r for d in cot_dirs if d.exists() for r in read_chunks(d)}
        cot_answers, finished, cot_tokens = {}, {}, {}
        for pid, r in recs.items():
            _, answer, done = split_thinking(r["texts"][0])
            cot_answers[pid], finished[pid], cot_tokens[pid] = answer, done, r["n_tokens"][0]
        cot_test = [p for p in test_ids if p in recs]
        if cot_test:
            abstained = {p: ("" if is_refusal(a) else a) for p, a in cot_answers.items()}
            tok = np.mean([cot_tokens[p] for p in cot_test])
            rows.append({"method": f"thinking mode (CoT), 1 answer; refusals scored as names [{len(cot_test)} test papers]",
                         "tokens_per_paper": tok, **answer_scores(cot_answers, cot_test, papers_by_id)})
            rows.append({"method": f"thinking mode (CoT), 1 answer; refusals as no answer [{len(cot_test)} test papers]",
                         "tokens_per_paper": tok, **answer_scores(abstained, cot_test, papers_by_id)})
            greedy_refusals = sum(is_refusal(greedy[p]["texts"][0]) for p in test_ids)
            cot = {"papers": len(recs), "test_papers": len(cot_test),
                   "unfinished_share": 1 - np.mean([finished[p] for p in cot_test]),
                   "refusal_share": float(np.mean([is_refusal(cot_answers[p]) for p in cot_test])),
                   "greedy_refusals": int(greedy_refusals),
                   "median_tokens": float(np.median([cot_tokens[p] for p in cot_test]))}
    table = pd.DataFrame([fmt(r) for r in rows])
    table.to_csv(out / "selection_tuned_test.csv", index=False)
    lines += ["## Selection, tuned on tune, scored on test", "",
              f"Chosen on tune: {best_freq}, {best_top}, {best_bg} (bg = mean count of the name over the other "
              "papers of the field). Mean HR2: lower is better; recalled = HR2 <= 0.5.", "",
              table.to_markdown(index=False), ""]
    if cot:
        lines += [f"CoT: {cot['test_papers']} test papers; median {cot['median_tokens']:.0f} tokens; "
                  f"{cot['unfinished_share']:.1%} hit the token limit before answering (scored as no names); "
                  f"{cot['refusal_share']:.1%} of answers are refusals (\"I cannot provide the authors...\"), "
                  f"which the paper's scoring splits into one or more false names. Greedy answers with a "
                  f"refusal on the test split: {cot['greedy_refusals']}.", ""]

    # --- 3. compute curve: first k samples, threshold tuned per k
    curve = []
    for k in KS:
        cand = {f"count>={t}": (lambda t, k: lambda p: p.counts(k) >= t)(t, k) for t in range(1, k + 1)}
        name, _ = tune(pools, tune_ids, cand)
        curve.append({"k": k, "rule": name, "tokens_per_paper": float(np.mean([pools[p].cum_tokens[k - 1] for p in test_ids])),
                      **evaluate(pools, test_ids, cand[name])})
    curve = pd.DataFrame([fmt(r) for r in curve])
    curve.to_csv(out / "compute_curve.csv", index=False)
    lines += ["## Compute curve (first k samples, threshold tuned per k; fast pool scoring on test)", "",
              curve.to_markdown(index=False), ""]

    # --- 4. coverage by author position (session 1, all papers)
    hits = pd.read_csv(run1 / "analysis" / "author_hits.csv.gz", dtype={"paper_id": str})
    true = hits[hits.kind == "true"].copy()
    true["position"] = np.where(true.author_index == 0, "first author", "other authors")
    true["label"] = true.paper_id.map(labels["label_real_team"])
    pos = (true.groupby(["position"]).agg(authors=("hits", "size"), named_by_default=("in_default", "mean"),
                                         in_any_sample=("hits", lambda h: (h >= 1).mean())).round(4))
    pos_known = (true[true.label == "known"].groupby("position")
                 .agg(authors=("hits", "size"), named_by_default=("in_default", "mean"),
                      in_any_sample=("hits", lambda h: (h >= 1).mean())).round(4))
    lines += ["## Coverage by author position", "", "All papers:", "", pos.to_markdown(), "",
              "Real-team known papers:", "", pos_known.to_markdown(), ""]

    # --- 5. candidate MC (session 2)
    if run2 and (Path(run2) / "mc_cand_mc").exists():
        items = pd.DataFrame(read_items(MC_ITEMS_DIR / "cand_mc.jsonl.gz"))
        res = pd.DataFrame(read_chunks(Path(run2) / "mc_cand_mc"))
        df = items.merge(res, on="item_id")
        df["chosen"] = df.letter_logprobs.map(lambda d: max(d, key=d.get) if d else None)
        df["any_true"] = df.true_letters.map(bool)
        df["model_correct"] = [c in cl for c, cl in zip(df.chosen, df.correct_letters)]
        top_letter = [ "ABCD"[r.index(0)] for r in df.ranks]  # the most frequent candidate's letter
        df["top1_correct"] = [t in tl for t, tl in zip(top_letter, df.true_letters)]
        df["chose_e"] = df.chosen == "E"
        rows = []
        for group, sub in [("pool has a true author", df[df.any_true]), ("pool has none", df[~df.any_true])]:
            rows.append({"items": group, "n_items": len(sub), "model_correct": sub.model_correct.mean(),
                         "most_frequent_correct": sub.top1_correct.mean() if group.startswith("pool has a") else float("nan"),
                         "model_chose_E": sub.chose_e.mean()})
        cm = pd.DataFrame([fmt(r) for r in rows])
        cm.to_csv(out / "cand_mc_summary.csv", index=False)
        lines += ["## Candidate multiple choice (top-4 sampled names + 'none of these')", "",
                  cm.to_markdown(index=False), "",
                  "most_frequent_correct: picking the candidate sampled most often, on the same items.", ""]

    # --- 5b. year-matched real teams (L2) against the session-1 real-team level
    mc_rows = []
    for level, run in (("real_team", run1), ("real_team_year", Path(run2) if run2 else None)):
        if run is None or not (run / f"mc_{level}").exists():
            continue
        items = pd.DataFrame(read_items(MC_ITEMS_DIR / f"{level}.jsonl.gz"))
        res = pd.DataFrame(read_chunks(run / f"mc_{level}"))
        if len(res) != len(items):
            continue
        df = items[["item_id", "paper_id", "condition", "correct_label"]].merge(res, on="item_id")
        df["correct"] = df.letter_logprobs.map(lambda d: max(d, key=d.get) if d else None) == df.correct_label
        won = df.groupby(["condition", "paper_id"]).correct.sum()
        for cond, sub in df.groupby("condition"):
            mc_rows.append({"level": level, "condition": cond, "papers": sub.paper_id.nunique(),
                            "per_question": sub.correct.mean(), ">=3 of 4": (won.loc[cond] >= 3).mean()})
    if mc_rows:
        mt = pd.DataFrame([fmt(r) for r in mc_rows])
        mt.to_csv(out / "mc_year_matched.csv", index=False)
        lines += ["## Real-team distractors with and without year matching (+-2 years)", "",
                  mt.to_markdown(index=False), ""]

    # --- 6. robustness: T=0.7 / top-p 0.8 / top-k 20 versus T=1 on the control subset
    if run2 and (Path(run2) / "samples_t07").exists():
        s07 = {r["paper_id"]: r["texts"] for r in read_chunks(Path(run2) / "samples_t07")}
        ids = sorted(s07)
        sub_papers = papers[papers.paper_id.isin(ids)]
        g = {p: greedy[p]["texts"][0] for p in ids}
        rows = []
        for name, smp in [("T=1.0, top-p 1, no top-k (main run)", {p: samples[p]["texts"] for p in ids}),
                          ("T=0.7, top-p 0.8, top-k 20", s07)]:
            tab = author_table(sub_papers, smp, g, n_partners=3, topic=True)
            row = {"sampling": name, "papers": len(ids)}
            for kind in ("true", "chance", "chance_topic"):
                h = tab[tab.kind == kind].hits
                row[f"cov50_{kind}"] = float((h >= 1).mean())
                row[f"cov10_{kind}"] = float(np.mean([coverage_at_k(50, int(c), 10) for c in h]))
            row["distinct_lists"] = float(np.mean([len(set(smp[p])) for p in ids]))
            rows.append(row)
        rb = pd.DataFrame([fmt(r) for r in rows])
        rb.to_csv(out / "robustness_temperature.csv", index=False)
        lines += ["## Robustness: sampling settings (1,023-paper control subset)", "", rb.to_markdown(index=False), ""]

    (out / "session2_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(*sys.argv[1:])
