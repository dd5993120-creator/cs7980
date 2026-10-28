"""True and false names per 100 known papers (the steering unit) on the test split.

    python tools/known_unit.py data/outputs/g4_bf16 data/outputs/s2

Known = our real-team label (team rules on our data). Rules are the ones chosen on the
tune split by analyze_session2.py / paired_tests.py: count >= 3 of the first 20 samples,
count >= 7 of 50. Bootstrap intervals resample known test papers.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analyze_session2 import is_refusal, load_pools  # noqa: E402
from gts.data import load_papers  # noqa: E402
from gts.generate import read_chunks  # noqa: E402
from gts.prompts import split_thinking  # noqa: E402
from paired_tests import per_paper_answers, per_paper_rule  # noqa: E402
from gts.splits import load_split  # noqa: E402


def ci(d, n_boot=5000):
    gen = np.random.default_rng(0)
    boots = [d[gen.integers(0, len(d), len(d))].mean() for _ in range(n_boot)]
    return np.percentile(boots, 2.5), np.percentile(boots, 97.5)


def main(run1: str, run2: str):
    run1, run2 = Path(run1), Path(run2)
    papers = load_papers()
    gt_str = dict(zip(papers.paper_id, papers.gt_str))
    labels = pd.read_csv(run1 / "analysis" / "papers.csv", dtype={"paper_id": str}).set_index("paper_id")
    split = load_split()
    rows = []
    for label_col in ("label_real_team", "label_own_sample"):
        ids = sorted(p for p, s in split.items() if s == "test" and labels.at[p, label_col] == "known")
        samples = {r["paper_id"]: r for r in read_chunks(run1 / "samples")}
        pools = load_pools(run1, samples, gt_str)
        greedy = {r["paper_id"]: r["texts"][0] for r in read_chunks(run1 / "greedy")}
        cot = {}
        for r in read_chunks(run2 / "cot_test"):
            a = split_thinking(r["texts"][0])[1]
            cot[r["paper_id"]] = "" if is_refusal(a) else a
        methods = {
            "default answer": per_paper_answers(greedy, ids, gt_str),
            "CoT, 1 answer": per_paper_answers(cot, ids, gt_str),
            ">= 3 of 20 samples": per_paper_rule(pools, ids, lambda p: p.counts(20) >= 3, gt_str),
            ">= 7 of 50 samples": per_paper_rule(pools, ids, lambda p: p.counts(50) >= 7, gt_str),
        }
        base = methods["default answer"]
        for name, df in methods.items():
            gain = (100 * (df.true - base.true)).to_numpy()
            lo, hi = ci(gain)
            rows.append({"known by": label_col.replace("label_", ""), "papers": len(ids), "method": name,
                         "true_per100": round(100 * df.true.mean(), 1), "false_per100": round(100 * df.false.mean(), 1),
                         "extra_true_vs_default": f"{gain.mean():+.1f} [{lo:+.1f}, {hi:+.1f}]"})
    table = pd.DataFrame(rows)
    out = run1 / "analysis" / "known_unit_test.md"
    out.write_text("# Steering unit on known test papers\n\n" + table.to_markdown(index=False) + "\n", encoding="utf-8")
    print(table.to_string(index=False))


if __name__ == "__main__":
    main(*sys.argv[1:])
