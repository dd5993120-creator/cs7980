"""Build the real-team multiple-choice items for the first Colab session.

    python tools/build_real_team_items.py

Writes data/mc_items/real_team.jsonl.gz: four cyclic orders per paper for all
9,108 papers, plus the title-hidden and title-swapped controls on a stratified
subset (up to 10 papers per field x citation bin).
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gts.data import load_papers  # noqa: E402
from gts.mc import build_real_team_items, control_subset, verify_items, write_items  # noqa: E402
from gts.paths import MC_ITEMS_DIR  # noqa: E402


def main():
    t0 = time.time()
    papers = load_papers()
    controls = control_subset(papers, per_stratum=10)
    items = build_real_team_items(papers, controls)
    report = verify_items(items, papers)
    report["n_control_papers"] = len(controls)
    print(json.dumps(report, indent=2, default=str))
    if report["distractor_groups_overlapping_truth"] or report["items_with_duplicate_options"] \
            or report["groups_without_4_orders"]:
        raise SystemExit("integrity check failed; items not written")
    out = MC_ITEMS_DIR / "real_team.jsonl.gz"
    write_items(items, out)
    print(f"wrote {len(items)} items to {out} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
