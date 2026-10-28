"""The 9,108-paper evaluation sample and the shipped Qwen3-32B greedy run.

Both files come from the Dreaming Up Authors release (data/release/):
    eval_sample_9108.csv.gz          one row per paper; authors as a JSON list
    qwen3_32b_greedy_scored.tsv.gz   the paper's Section 7 run: Qwen3-32B, bf16,
                                     thinking off, greedy, max 96 new tokens,
                                     scored with the vendored matcher
"""

from __future__ import annotations

import json

import pandas as pd

from gts.paths import RELEASE_DIR

SAMPLE_FILE = RELEASE_DIR / "eval_sample_9108.csv.gz"
SHIPPED_GREEDY_FILE = RELEASE_DIR / "qwen3_32b_greedy_scored.tsv.gz"

BIN_LABELS = [
    "0-2", "3-4", "5-8", "9-16", "17-32", "33-64", "65-128",
    "129-256", "257-512", "513-1024", "1025-2048", "2049-4096", "4097+",
]
FIELDS = [
    "medicine", "biology", "chemistry", "computer science",
    "materials science", "physics", "psychology", "mathematics",
]


def load_papers(path=SAMPLE_FILE) -> pd.DataFrame:
    """One row per paper, with ``authors`` as a list and ``gt_str`` as one string.

    ``gt_str`` joins the authors with ", ", the form the shipped Qwen3-32B run
    was scored against (dua.probing.prompts._authors_to_answer).
    """
    df = pd.read_csv(path, dtype={"paper_id": str, "record_id": str})
    df["authors"] = df["authors"].map(json.loads)
    df["gt_str"] = df["authors"].map(", ".join)
    df["bin_index"] = df["citation_bin"].map(BIN_LABELS.index)
    return df


def load_shipped_greedy(path=SHIPPED_GREEDY_FILE) -> pd.DataFrame:
    """The paper's Qwen3-32B greedy answers with their (M, H, U) scores."""
    return pd.read_csv(path, sep="\t", dtype={"paper_id": str})
