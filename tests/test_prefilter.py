"""The could_match prefilter must never change a matching result."""

import numpy as np
import pandas as pd

from gts.data import load_papers, load_shipped_greedy
from gts.dua_matcher import MATCH_LEVEL_THRESHOLD, _match_level_value
from gts.matching import clean_generated_text, get_matcher, gt_names, matched_gt_indices, overlaps_any, split_answer


def full_indices(gt_list, answer):
    result = get_matcher().find_matches(gt_list, clean_generated_text(answer))
    return {m["gt_index"] for m in result["matches"] if _match_level_value(m["match_level"]) <= MATCH_LEVEL_THRESHOLD}


def full_overlap(names, others):
    return bool(names and others and get_matcher().find_matches(others, "; ".join(names))["matches"])


def _pairs():
    papers = load_papers()
    answers = load_shipped_greedy().set_index("paper_id")["generated_text"].fillna("")
    papers["answer"] = papers["paper_id"].map(answers)
    rng = np.random.default_rng(0)
    own = list(zip(papers["gt_str"], papers["answer"]))
    # cross pairs inside a field, where surnames collide most often
    cross = []
    for _, group in papers.groupby("field"):
        idx = group.index.to_numpy()
        for i, j in zip(rng.choice(idx, 3000), rng.choice(idx, 3000)):
            cross.append((papers.at[i, "gt_str"], papers.at[j, "answer"]))
    return own + cross


def test_prefilter_keeps_every_l4_match():
    diffs = [(g, a) for g, a in _pairs() if matched_gt_indices(gt_names(g), a) != full_indices(gt_names(g), a)]
    assert not diffs, diffs[:5]


def test_prefilter_keeps_every_overlap():
    diffs = []
    for g, a in _pairs():
        names, truth = split_answer(a), gt_names(g)
        if overlaps_any(names, truth) != full_overlap(names, truth):
            diffs.append((g, a))
    assert not diffs, diffs[:5]


def test_particle_and_swap_cases():
    assert matched_gt_indices(["J. D. van der Waals"], "Johannes Waals") == {0}
    assert matched_gt_indices(["R. Yehuda"], "Yehuda R") == {0}
    assert overlaps_any(["Adam Wurman"], ["Peter Wurman"])  # L5 (surname only) counts as overlap
    assert not overlaps_any(["K. Smith"], ["Jane Smith"])  # contradicting initials block even L5
