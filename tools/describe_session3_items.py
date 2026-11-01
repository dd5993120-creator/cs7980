"""How the session 3 distractors came out: L3 tiers, title similarity and prominence gaps
(against L1 and L2 donors), and how many L4 swaps use real collaborators.

    python tools/describe_session3_items.py

Writes data/outputs/g4_bf16/analysis/session3_items.md.
"""

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gts.data import load_papers  # noqa: E402
from gts.mc import read_items, team_prominence  # noqa: E402
from gts.paths import DATA_DIR, MC_ITEMS_DIR, OUTPUTS_DIR  # noqa: E402

import gzip  # noqa: E402
import json  # noqa: E402


def first_items(name: str) -> list[dict]:
    return [i for i in read_items(MC_ITEMS_DIR / f"{name}.jsonl.gz") if i["condition"] == "standard" and i["order"] == 0]


def main():
    from sklearn.feature_extraction.text import TfidfVectorizer

    papers = load_papers()
    with gzip.open(DATA_DIR / "external" / "s2_papers.jsonl.gz", "rt", encoding="utf-8") as f:
        prominence = {r["paper_id"]: team_prominence(r["authors"]) for r in map(json.loads, f)}
    sim = {}
    for _, group in papers.groupby("field"):
        ids = sorted(group["paper_id"])
        titles = group.set_index("paper_id").loc[ids, "title"].astype(str)
        m = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit_transform(titles)
        sims = (m @ m.T).toarray()
        index = {p: i for i, p in enumerate(ids)}
        sim[group["field"].iloc[0]] = (index, sims)
    field = dict(zip(papers["paper_id"], papers["field"]))

    def donor_stats(items):
        s, g = [], []
        for it in items:
            index, sims = sim[field[it["paper_id"]]]
            for d in it["donors"]:
                s.append(sims[index[it["paper_id"]], index[d]] if d in index else np.nan)
                pa, pb = prominence.get(it["paper_id"], np.nan), prominence.get(d, np.nan)
                g.append(abs(pa - pb))
        s, g = np.array(s), np.array(g)
        return {"donors": len(s), "title similarity, median": np.nanmedian(s), "title similarity, mean": np.nanmean(s),
                "prominence gap, median": np.nanmedian(g), "prominence gap <= 0.25": np.nanmean(g[~np.isnan(g)] <= 0.25)}

    rows = [{"level": lvl, **donor_stats(first_items(name))}
            for lvl, name in (("L1", "real_team"), ("L2", "real_team_year"), ("L3", "similar_title"),
                              ("L3t", "similar_topic"))]
    l3 = first_items("similar_title")
    tiers = Counter(r for it in l3 for r in it["donor_rules"])
    l4 = first_items("collab_swap")
    l4_rules = Counter(r for it in l4 for r in it["swap_rules"])
    full = sum(it["swap_rules"].count("collaborator") == 3 for it in l4)
    lines = ["# How the session 3 distractors came out", "",
             "Donor teams: TF-IDF title similarity to the paper (same field) and the prominence gap",
             "(|log(1 + h-index)| of each team's best-known member).", "",
             pd.DataFrame(rows).to_markdown(index=False, floatfmt=".3f"), "",
             "L3 donor tiers (most constrained first):", "",
             pd.DataFrame(sorted(tiers.items(), key=lambda kv: -kv[1]), columns=["tier", "donors"]).to_markdown(index=False), "",
             f"L4: {l4_rules['collaborator']} of {sum(l4_rules.values())} swapped-in names are real collaborators "
             f"({l4_rules['collaborator'] / sum(l4_rules.values()):.1%}); {full} of {len(l4)} papers "
             f"({full / len(l4):.1%}) have all three; the rest are random field names.", ""]
    out = OUTPUTS_DIR / "g4_bf16" / "analysis" / "session3_items.md"
    out.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
