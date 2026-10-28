"""A tune/test split of the 9,108 papers, stratified by field x citation bin.

Every threshold or setting is chosen on "tune"; "test" is scored once, for the
reported numbers. Membership is a hash of the paper id, so it is reproducible.
"""

from __future__ import annotations

import hashlib

import pandas as pd

from gts.paths import DATA_DIR

SPLIT_FILE = DATA_DIR / "splits" / "split_v1.csv"


def make_split(papers: pd.DataFrame, tune_frac: float = 0.3) -> pd.DataFrame:
    rows = []
    for _, group in papers.groupby(["field", "citation_bin"], sort=True):
        ids = sorted(group["paper_id"], key=lambda pid: hashlib.sha256(f"split|{pid}".encode()).hexdigest())
        n_tune = round(len(ids) * tune_frac)
        rows += [(pid, "tune" if i < n_tune else "test") for i, pid in enumerate(ids)]
    return pd.DataFrame(rows, columns=["paper_id", "split"]).sort_values("paper_id").reset_index(drop=True)


def load_split(path=SPLIT_FILE) -> dict[str, str]:
    df = pd.read_csv(path, dtype={"paper_id": str})
    return dict(zip(df["paper_id"], df["split"]))
