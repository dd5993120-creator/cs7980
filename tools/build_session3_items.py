"""Build everything session 3 needs on the GPU: the rest of the recognition ladder,
catch trials, the yes/no and likelihood formats, and likelihood scores for selection.

    python tools/build_session3_items.py data/outputs/g4_bf16

Needs data/external/s2_papers.jsonl.gz and s2_collaborators.jsonl.gz (tools/fetch_s2.py).
Writes data/mc_items/:
    random_field.jsonl.gz    L0: random names from the paper's field
    similar_title.jsonl.gz   L3: real teams of the most similar titles, of similar prominence
    similar_topic.jsonl.gz   L3t: the most similar titles only, any year (python ... --only=similar_topic)
    collab_swap.jsonl.gz     L4r: one author swapped for a real collaborator of the first author,
                             a position drawn per distractor (the release's design; it leaks)
    collab_slot.jsonl.gz     L4: the same, all three swaps at one shared position (--only=collab_slot)
    catch.jsonl.gz           catch trials on the L1 distractors, with "None of the above"
    yesno_lists.jsonl.gz     a yes/no question on every option list of L1, L4 and L5
    score_lists.jsonl.gz     the likelihood of every option list of L1 and L5, title shown and hidden
    score_cands.jsonl.gz     likelihood scores for each paper's top-10 sampled candidates
"""

import gzip
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from gts.data import load_papers  # noqa: E402
from gts.generate import read_chunks  # noqa: E402
from gts.matching import name_key  # noqa: E402
from gts.mc import (CollaboratorSwapBuilder, RandomFieldBuilder, SimilarTitleBuilder,  # noqa: E402
                    build_catch_items, build_level_items, build_list_score_records, build_yesno_records,
                    control_subset, read_items, team_prominence, verify_items, write_items)
from gts.paths import DATA_DIR, MC_ITEMS_DIR  # noqa: E402
from gts.selection import build_candidate_score_records  # noqa: E402


def read_jsonl(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def checked(name: str, items: list[dict], papers) -> list[dict]:
    report = verify_items(items, papers)
    print(json.dumps({name: report}, indent=1, default=str), flush=True)
    if report["distractor_groups_overlapping_truth"] or report["items_with_duplicate_options"] \
            or report["groups_without_4_orders"]:
        raise SystemExit(f"{name} failed its integrity check")
    write_items(items, MC_ITEMS_DIR / f"{name}.jsonl.gz")
    return items


def first_of_each(items: list[dict]) -> list[dict]:
    return [i for i in items if i["condition"] == "standard" and i["order"] == 0]


def similar_topic(papers, control_ids, prominence) -> None:
    """L3t: the most similar titles only (teams at least as long, truncated), any year."""
    items = checked("similar_topic", build_level_items(
        papers, control_ids, "similar_topic",
        SimilarTitleBuilder(papers, prominence, topic_only=True).distractors), papers)
    first = first_of_each(items)
    sims = [s for i in first for s in i["title_similarity"]]
    print(json.dumps({"similar_topic": {
        "donor_tiers": dict(Counter(r for i in first for r in i["donor_rules"])),
        "title_similarity_mean": round(float(np.mean(sims)), 3),
        "title_similarity_median": round(float(np.median(sims)), 3),
    }}, indent=1), flush=True)


def collab_slot(papers, control_ids, collaborators) -> None:
    """L4: the three collaborator swaps share one position (see CollaboratorSwapBuilder)."""
    items = checked("collab_slot", build_level_items(
        papers, control_ids, "collab_slot",
        CollaboratorSwapBuilder(papers, collaborators, shared_position=True).distractors), papers)
    first = first_of_each(items)
    assert all(len(set(i["swap_positions"])) == 1 for i in first)
    print(json.dumps({"collab_slot": {
        "options_from_collaborators": sum(i["swap_rules"].count("collaborator") for i in first),
        "papers_with_3_collaborators": sum(i["swap_rules"].count("collaborator") == 3 for i in first),
        "papers": len(first)}}, indent=1), flush=True)


def main(run_dir: str, only: str | None = None):
    t0 = time.time()
    papers = load_papers()
    control_ids = control_subset(papers, per_stratum=10)
    external = DATA_DIR / "external"
    s2 = {r["paper_id"]: r["authors"] for r in read_jsonl(external / "s2_papers.jsonl.gz")}
    prominence = {pid: team_prominence(a) for pid, a in s2.items()}
    if only == "similar_topic":
        similar_topic(papers, control_ids, prominence)
        print(f"done in {time.time() - t0:.0f}s")
        return
    # collaborators only where Semantic Scholar's first author is the ground truth's first author
    first_true = {pid: name_key(a[0]) for pid, a in zip(papers["paper_id"], papers["authors"]) if a}
    same_first = {pid for pid, a in s2.items()
                  if a and a[0].get("name") and pid in first_true and name_key(a[0]["name"]) == first_true[pid]}
    collaborators = {r["paper_id"]: [name for name, _ in r["collaborators"]] if r["paper_id"] in same_first else []
                     for r in read_jsonl(external / "s2_collaborators.jsonl.gz")}
    print(f"prominence known for {sum(not np.isnan(v) for v in prominence.values())} papers; "
          f"first author agrees with Semantic Scholar for {len(same_first)}; "
          f"collaborators for {sum(bool(v) for v in collaborators.values())}", flush=True)
    if only == "collab_slot":
        collab_slot(papers, control_ids, collaborators)
        # the collab_swap items that were run must rebuild unchanged
        rebuilt = build_level_items(papers, control_ids, "collab_swap",
                                    CollaboratorSwapBuilder(papers, collaborators).distractors)
        if rebuilt != read_items(MC_ITEMS_DIR / "collab_swap.jsonl.gz"):
            raise SystemExit("collab_swap no longer rebuilds as it was run")
        print(f"collab_swap rebuilds unchanged; done in {time.time() - t0:.0f}s")
        return

    l0 = checked("random_field", build_level_items(
        papers, control_ids, "random_field", RandomFieldBuilder(papers).distractors), papers)
    print(f"L0 done ({time.time() - t0:.0f}s)", flush=True)

    l3 = checked("similar_title", build_level_items(
        papers, control_ids, "similar_title", SimilarTitleBuilder(papers, prominence).distractors), papers)
    first = first_of_each(l3)
    sims = [s for i in first for s in i["title_similarity"]]
    gaps = [g for i in first for g in i["prominence_gap"] if g is not None]
    print(json.dumps({"similar_title": {
        "donor_tiers": dict(Counter(r for i in first for r in i["donor_rules"])),
        "title_similarity_median": round(float(np.median(sims)), 3),
        "prominence_gap_median": round(float(np.median(gaps)), 3),
        "prominence_gap_le_0.25": round(float(np.mean(np.array(gaps) <= 0.25)), 3),
    }}, indent=1), flush=True)
    print(f"L3 done ({time.time() - t0:.0f}s)", flush=True)
    similar_topic(papers, control_ids, prominence)

    l4 = checked("collab_swap", build_level_items(
        papers, control_ids, "collab_swap", CollaboratorSwapBuilder(papers, collaborators).distractors), papers)
    first = first_of_each(l4)
    print(json.dumps({"collab_swap": {
        "papers_with_3_collaborators": sum(i["swap_rules"].count("collaborator") == 3 for i in first),
        "options_from_collaborators": sum(i["swap_rules"].count("collaborator") for i in first),
        "papers": len(first),
    }}, indent=1), flush=True)
    print(f"L4 done ({time.time() - t0:.0f}s)", flush=True)
    collab_slot(papers, control_ids, collaborators)

    real_team = read_items(MC_ITEMS_DIR / "real_team.jsonl.gz")
    own_sample = read_items(MC_ITEMS_DIR / "own_sample.jsonl.gz")
    checked("catch", build_catch_items(papers, real_team), papers)

    yesno = build_yesno_records({"real_team": real_team, "collab_swap": l4, "own_sample": own_sample})
    write_items(yesno, MC_ITEMS_DIR / "yesno_lists.jsonl.gz")
    lists = build_list_score_records({"real_team": real_team, "own_sample": own_sample})
    write_items(lists, MC_ITEMS_DIR / "score_lists.jsonl.gz")
    print(f"yes/no records {len(yesno)}, list scoring records {len(lists)} ({time.time() - t0:.0f}s)", flush=True)

    samples = {r["paper_id"]: r["texts"] for r in read_chunks(Path(run_dir) / "samples")}
    cands = build_candidate_score_records(papers, samples)
    # the same pools as the P(True) run, so the scores line up candidate by candidate
    ptrue = {r["record_id"]: r for r in read_items(MC_ITEMS_DIR / "ptrue_candidates.jsonl.gz")}
    mismatched = sum(1 for r in cands if r["kind"] == "author" and r["title"] is not None
                     and (ptrue.get(f"{r['paper_id']}|{r['rank']}", {}).get("name") != r["name_shown"]
                          or ptrue[f"{r['paper_id']}|{r['rank']}"]["is_true"] != r["is_true"]))
    if mismatched:
        raise SystemExit(f"{mismatched} candidates differ from ptrue_candidates")
    write_items(cands, MC_ITEMS_DIR / "score_cands.jsonl.gz")
    print(f"candidate scoring records {len(cands)}, kinds {dict(Counter(r['record_id'].split('|')[0] + '|' + r['record_id'].split('|')[1] for r in cands))}")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    # optional second argument "--only=similar_topic" builds just that level
    main(sys.argv[1], sys.argv[2].split("=", 1)[1] if len(sys.argv) > 2 else None)
