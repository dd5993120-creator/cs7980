"""Build everything session 2 needs on the GPU.

    python tools/build_session2_items.py data/outputs/g4_bf16

Writes
    data/splits/split_v1.csv               30% tune / 70% test, stratified by field x bin
    data/mc_items/real_team_year.jsonl.gz  real teams published within 2 years of the paper (L2)
    data/mc_items/cand_mc.jsonl.gz         the model's top-4 candidates plus "none of these"
"""

import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gts.data import load_papers  # noqa: E402
from gts.generate import read_chunks  # noqa: E402
from gts.mc import build_cand_mc_items, build_real_team_items, control_subset, verify_items, write_items  # noqa: E402
from gts.paths import MC_ITEMS_DIR  # noqa: E402
from gts.splits import SPLIT_FILE, make_split  # noqa: E402


def main(run_dir: str):
    t0 = time.time()
    papers = load_papers()

    split = make_split(papers)
    SPLIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    split.to_csv(SPLIT_FILE, index=False)
    print("split:", split.split.value_counts().to_dict())

    items = build_real_team_items(papers, control_subset(papers, per_stratum=10),
                                  level="real_team_year", year_window=2)
    report = verify_items(items, papers)
    print(json.dumps({"real_team_year": report}, indent=1, default=str))
    if report["distractor_groups_overlapping_truth"] or report["items_with_duplicate_options"] \
            or report["groups_without_4_orders"]:
        raise SystemExit("real_team_year failed its integrity check")
    write_items(items, MC_ITEMS_DIR / "real_team_year.jsonl.gz")

    samples = {r["paper_id"]: r["texts"] for r in read_chunks(Path(run_dir) / "samples")}
    cand = build_cand_mc_items(papers, samples)
    first = [i for i in cand if i["order"] == 0]
    print(json.dumps({"cand_mc": {
        "items": len(cand), "papers": len(first),
        "papers_with_a_true_candidate": sum(bool(i["true_letters"]) for i in first),
        "true_candidates_per_paper": dict(Counter(len(i["true_letters"]) for i in first)),
        "position_of_correct": dict(Counter("".join(i["correct_letters"]) for i in cand if i["true_letters"])),
    }}, indent=1, default=str))
    write_items(cand, MC_ITEMS_DIR / "cand_mc.jsonl.gz")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1])
