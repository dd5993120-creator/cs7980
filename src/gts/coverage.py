"""Coverage of the true authors by sampled answers, and its chance baseline.

For one true author, ``c`` of the ``n`` sampled lists name it (L1-L4). The
probability that at least one of ``k`` samples names it is estimated without
bias by 1 - C(n-c, k) / C(n, k) (Chen et al., 2021, the pass@k estimator).

Common surnames match by luck as k grows, so every paper's samples are also
matched against the authors of partner papers from the same field and
citation bin. That gives the coverage a model with no knowledge of the paper
would reach; we report coverage net of it.
"""

from __future__ import annotations

from math import comb

import numpy as np
import pandas as pd

from gts.dua_matcher import MATCH_LEVEL_THRESHOLD, _match_level_value
from gts.matching import clean_generated_text, could_match, get_matcher, gt_names, matched_gt_indices
from gts.mc import paper_rng


def coverage_at_k(n: int, c: int, k: int) -> float:
    """Unbiased P(at least one of k samples hits) from c hits among n samples."""
    if not 0 <= c <= n or not 1 <= k <= n:
        raise ValueError(f"need 0 <= c <= n and 1 <= k <= n, got n={n}, c={c}, k={k}")
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def hit_counts(gt_list: list[str], texts: list[str]) -> list[int]:
    """For each true author, how many of the texts name it at L1-L4."""
    return multi_hit_counts([gt_list], texts)[0]


def multi_hit_counts(gt_lists: list[list[str]], texts: list[str]) -> list[list[int]]:
    """hit_counts for several author lists against the same texts, splitting each text once."""
    m = get_matcher()
    split = []
    for text in texts:
        cleaned = clean_generated_text(text)
        split.append((cleaned, m.splitter.split(cleaned)))
    out = []
    for gt_list in gt_lists:
        counts = [0] * len(gt_list)
        for cleaned, names in split:
            if not could_match(gt_list, names, m):
                continue
            for x in m.find_matches(gt_list, cleaned)["matches"]:
                if _match_level_value(x["match_level"]) <= MATCH_LEVEL_THRESHOLD:
                    counts[x["gt_index"]] += 1
        out.append(counts)
    return out


def _usable(papers: pd.DataFrame, i: int, j: int) -> bool:
    """A partner must share no author with the paper (surname-strict), or its
    authors would be true knowledge, not chance. Skipped when no author lists."""
    if "authors" not in papers:
        return True
    from gts.mc import conflicts
    return not conflicts(papers.iat[j, papers.columns.get_loc("authors")],
                         papers.iat[i, papers.columns.get_loc("authors")])


def partners(papers: pd.DataFrame, n_partners: int = 3) -> dict[str, list[str]]:
    """For every paper, ``n_partners`` random other papers from the same field and citation bin."""
    out = {}
    for _, group in papers.groupby(["field", "citation_bin"], sort=True):
        group = group.sort_values("paper_id").reset_index(drop=True)
        ids = group["paper_id"].tolist()
        for i, pid in enumerate(ids):
            gen = paper_rng(pid, "chance_partner")
            chosen = []
            for j in gen.permutation(len(ids)):
                if j != i and _usable(group, i, j):
                    chosen.append(ids[j])
                    if len(chosen) == n_partners:
                        break
            out[pid] = chosen
    return out


def topic_partners(papers: pd.DataFrame, n_partners: int = 3) -> dict[str, list[str]]:
    """For every paper, the most similar titles in its field (TF-IDF cosine).

    A stricter chance baseline than random partners: it also removes what
    knowing who works on the topic would predict, so what remains is closer
    to memory of this particular paper.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    out = {}
    for _, group in papers.groupby("field", sort=True):
        group = group.sort_values("paper_id").reset_index(drop=True)
        ids = group["paper_id"].tolist()
        tfidf = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit_transform(group["title"].astype(str))
        sims = (tfidf @ tfidf.T).toarray()
        for i, pid in enumerate(ids):
            chosen = []
            for j in np.argsort(-sims[i], kind="stable"):
                if j != i and _usable(group, i, j):
                    chosen.append(ids[j])
                    if len(chosen) == n_partners:
                        break
            out[pid] = chosen
    return out


def author_table(papers: pd.DataFrame, samples: dict[str, list[str]],
                 greedy: dict[str, str], n_partners: int = 3, topic: bool = False) -> pd.DataFrame:
    """One row per (paper, true author): hits among the samples and in the default answer,
    plus the same counts for partner papers' authors (the chance baselines).

    ``kind`` is "true" for the paper's own authors, "chance" for a random
    same-field, same-bin partner's, and "chance_topic" (with ``topic=True``)
    for a most-similar-title partner's; chance rows carry the partner's author
    but this paper's samples.
    """
    by_id = papers.set_index("paper_id")
    gt = {pid: gt_names(by_id.at[pid, "gt_str"]) for pid in by_id.index}
    subset = papers.loc[papers["paper_id"].isin(samples)]
    part = {"chance": partners(subset, n_partners)}
    if topic:
        part["chance_topic"] = topic_partners(papers, n_partners)
    rows = []
    for pid, texts in samples.items():
        meta = {"paper_id": pid, "field": by_id.at[pid, "field"],
                "citation_bin": by_id.at[pid, "citation_bin"], "n_samples": len(texts)}
        default_hits = matched_gt_indices(gt[pid], greedy.get(pid, ""))
        tagged = [(kind, q) for kind, mapping in part.items() for q in mapping.get(pid, [])]
        own, *others = multi_hit_counts([gt[pid], *(gt[q] for _, q in tagged)], texts)
        for i, c in enumerate(own):
            rows.append(meta | {"kind": "true", "author_index": i, "hits": c,
                                "in_default": i in default_hits})
        for (kind, partner), counts in zip(tagged, others):
            for i, c in enumerate(counts):
                rows.append(meta | {"kind": kind, "partner_id": partner, "author_index": i,
                                    "hits": c, "in_default": False})
    return pd.DataFrame(rows)


def coverage_curve(table: pd.DataFrame, ks=(1, 2, 5, 10, 20, 50)) -> pd.DataFrame:
    """Mean coverage@k per citation bin for true and chance authors, and their difference."""
    rows = []
    for (bin_label, kind), group in table.groupby(["citation_bin", "kind"]):
        n = int(group["n_samples"].iloc[0])
        for k in ks:
            if k > n:
                continue
            cov = np.mean([coverage_at_k(n, int(c), k) for c in group["hits"]])
            rows.append({"citation_bin": bin_label, "kind": kind, "k": k, "coverage": cov})
    curve = pd.DataFrame(rows).pivot_table(index=["citation_bin", "k"], columns="kind", values="coverage")
    curve["net"] = curve["true"] - curve["chance"]
    if "chance_topic" in curve:
        curve["net_topic"] = curve["true"] - curve["chance_topic"]
    return curve.reset_index()
