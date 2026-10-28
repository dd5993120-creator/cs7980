"""Write the top candidates of every paper's pool for P(True) scoring.

    python tools/build_ptrue_candidates.py data/outputs/g4_bf16

Reads <run>/samples and writes data/mc_items/ptrue_candidates.jsonl.gz, one
record per (paper, candidate) with whether the candidate is a true author.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gts.data import load_papers  # noqa: E402
from gts.generate import read_chunks  # noqa: E402
from gts.matching import gt_names  # noqa: E402
from gts.mc import write_items  # noqa: E402
from gts.paths import MC_ITEMS_DIR  # noqa: E402
from gts.selection import candidate_pool, label_candidates  # noqa: E402


def main(run_dir: str):
    t0 = time.time()
    papers = load_papers().set_index("paper_id")
    samples = {r["paper_id"]: r["texts"] for r in read_chunks(Path(run_dir) / "samples")}
    records = []
    for pid in sorted(samples):
        paper = papers.loc[pid]
        for c in label_candidates(candidate_pool(samples[pid]), gt_names(paper["gt_str"])):
            records.append({"record_id": f"{pid}|{c['rank']}", "paper_id": pid, "title": paper["title"],
                            "year": int(paper["year"]), **c})
    n_true = sum(r["is_true"] for r in records)
    print(json.dumps({"papers": len(samples), "candidates": len(records), "true_candidates": n_true,
                      "papers_with_a_true_candidate": len({r["paper_id"] for r in records if r["is_true"]})}))
    write_items(records, MC_ITEMS_DIR / "ptrue_candidates.jsonl.gz")
    print(f"wrote {MC_ITEMS_DIR / 'ptrue_candidates.jsonl.gz'} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1])
