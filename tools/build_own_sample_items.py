"""Build the own-sample multiple-choice items from a downloaded sampling run.

    python tools/build_own_sample_items.py data/outputs/g4_bf16

Reads <run>/samples/chunk_*.jsonl.gz and writes data/mc_items/own_sample.jsonl.gz:
four cyclic orders per paper, plus the title-hidden and title-swapped controls
on the same stratified subset as the real-team items.
"""

import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gts.data import load_papers  # noqa: E402
from gts.generate import read_chunks  # noqa: E402
from gts.mc import build_own_sample_items, control_subset, verify_items, write_items  # noqa: E402
from gts.paths import MC_ITEMS_DIR  # noqa: E402


def main(run_dir: str):
    t0 = time.time()
    papers = load_papers()
    samples = {r["paper_id"]: r["texts"] for r in read_chunks(Path(run_dir) / "samples")}
    if len(samples) != len(papers):
        raise SystemExit(f"samples cover {len(samples)} of {len(papers)} papers; finish the run first")
    controls = control_subset(papers, per_stratum=10)
    items = build_own_sample_items(papers, samples, controls)
    report = verify_items(items, papers)
    standard = [i for i in items if i["order"] == 0 and i["condition"] == "standard"]
    report["papers_with_any_fallback"] = sum("real_team_fallback" in i["distractor_rules"] for i in standard)
    report["papers_all_from_samples"] = sum(all(r.startswith("sampled") for r in i["distractor_rules"]) for i in standard)
    report["wrong_lists_per_paper"] = dict(sorted(Counter(min(i["n_wrong_lists"], 10) for i in standard).items()))
    print(json.dumps(report, indent=2, default=str))
    if report["distractor_groups_overlapping_truth"] or report["items_with_duplicate_options"] \
            or report["groups_without_4_orders"]:
        raise SystemExit("integrity check failed; items not written")
    out = MC_ITEMS_DIR / "own_sample.jsonl.gz"
    write_items(items, out)
    print(f"wrote {len(items)} items to {out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1])
