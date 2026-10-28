"""Paired comparison on the subset's test papers, at about 3,200 tokens per paper:
five CoT answers + frequency (count >= 2) versus 100 ordinary samples + frequency
(count >= 9); both thresholds were chosen on the subset's tune papers by analyze_deep.py.

    python tools/paired_deep.py data/outputs/s2
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyze_session2 import Pool  # noqa: E402
from gts.data import load_papers  # noqa: E402
from gts.generate import read_chunks  # noqa: E402
from gts.matching import gt_names, score_answer  # noqa: E402
from gts.mc import control_subset  # noqa: E402
from gts.prompts import split_thinking  # noqa: E402
from gts.splits import load_split  # noqa: E402


def per_paper(pools, ids, k, t, gt_str):
    rows = []
    for pid in ids:
        pool = pools[pid]
        names = [pool.form[key] for key, s in zip(pool.keys, pool.counts(k) >= t) if s]
        s = score_answer(gt_str[pid], "; ".join(names))
        rows.append({"paper_id": pid, "hr2": s["hr2"], "true": s["n_match"], "false": s["n_hal"]})
    return pd.DataFrame(rows).set_index("paper_id")


def ci(d, n_boot=5000):
    gen = np.random.default_rng(0)
    boots = [d[gen.integers(0, len(d), len(d))].mean() for _ in range(n_boot)]
    return f"{d.mean():+.4f} [{np.percentile(boots, 2.5):+.4f}, {np.percentile(boots, 97.5):+.4f}]"


def main(run2: str):
    run2 = Path(run2)
    papers = load_papers()
    gt_str = dict(zip(papers.paper_id, papers.gt_str))
    split = load_split()
    ids = [p for p in sorted(control_subset(papers, per_stratum=10)) if split[p] == "test"]
    s200 = {r["paper_id"]: r for r in read_chunks(run2 / "samples200") if r["paper_id"] in set(ids)}
    c5 = {r["paper_id"]: r for r in read_chunks(run2 / "cot5") if r["paper_id"] in set(ids)}
    plain = {p: Pool(r["texts"][:100], r["n_tokens"][:100], gt_names(gt_str[p])) for p, r in s200.items()}
    think = {p: Pool([split_thinking(t)[1] for t in r["texts"]], r["n_tokens"], gt_names(gt_str[p]))
             for p, r in c5.items()}
    a = per_paper(think, ids, 5, 2, gt_str)
    b = per_paper(plain, ids, 100, 9, gt_str)
    lines = [f"# CoT x5 + frequency vs 100 samples + frequency ({len(ids)} subset test papers)", "",
             f"- mean HR2: {a.hr2.mean():.4f} vs {b.hr2.mean():.4f}; difference {ci((a.hr2 - b.hr2).to_numpy())}",
             f"- true per 100: {100 * a.true.mean():.1f} vs {100 * b.true.mean():.1f}; "
             f"difference {ci(100 * (a.true - b.true).to_numpy())}",
             f"- false per 100: {100 * a.false.mean():.1f} vs {100 * b.false.mean():.1f}; "
             f"difference {ci(100 * (a.false - b.false).to_numpy())}"]
    out = Path(__file__).resolve().parents[1] / "data" / "outputs" / "g4_bf16" / "analysis" / "paired_deep.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1])
