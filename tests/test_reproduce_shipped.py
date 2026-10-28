"""Our matcher wrapper must reproduce the shipped Qwen3-32B scores row for row.

If this fails, every number we report would sit on a different scale from the
paper's Qwen3-32B baseline.
"""

import pandas as pd

from gts.data import load_papers, load_shipped_greedy
from gts.matching import score_answer

COMPARED = ["n_match", "n_hal", "n_unretr", "n_gt_authors", "n_pred_authors", "predicted_authors"]


def test_reproduces_every_shipped_row():
    papers = load_papers()[["paper_id", "gt_str"]]
    shipped = load_shipped_greedy()
    merged = shipped.merge(papers, on="paper_id", how="left", validate="one_to_one")
    assert merged["gt_str"].notna().all(), "a shipped paper_id is missing from the sample"
    assert len(merged) == 9108

    rows = []
    for rec in merged.itertuples(index=False):
        ours = score_answer(rec.gt_str, "" if pd.isna(rec.generated_text) else rec.generated_text)
        rows.append({k: ours[k] for k in COMPARED} | {"hr2": ours["hr2"]})
    ours = pd.DataFrame(rows)

    shipped_pred = merged["predicted_authors"].fillna("")
    mismatched = (
        (ours["n_match"] != merged["n_match"])
        | (ours["n_hal"] != merged["n_hal"])
        | (ours["n_unretr"] != merged["n_unretr"])
        | (ours["n_gt_authors"] != merged["n_gt_authors"])
        | (ours["n_pred_authors"] != merged["n_pred_authors"])
        | (ours["predicted_authors"] != shipped_pred)
        | ((ours["hr2"] - merged["hr2"]).abs() > 1e-6)
    )
    if mismatched.any():
        sample = merged.loc[mismatched, ["paper_id", "n_match", "n_hal", "n_unretr"]].head(10)
        raise AssertionError(f"{int(mismatched.sum())} rows differ:\n{sample}\n{ours[mismatched].head(10)}")
