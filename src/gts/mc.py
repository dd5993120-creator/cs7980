"""Multiple-choice recognition items: distractors, option orders and controls.

Levels (where the three wrong options come from):
    real_team    the author lists of three other papers in the same field with
                 the same number of authors (the team test's design)
    own_sample   wrong author lists the model itself sampled for this paper;
                 built once the sampling run is back (see build_own_sample_items)

Conditions:
    standard       the paper's own title and year
    title_hidden   the title replaced by "[title not shown]"; should stay at chance
    title_swapped  the title and year of another paper from the same field and
                   citation bin, with this paper's options; picking this paper's
                   true list should stay at chance

Every item is asked in the four cyclic orders, so the true list sits once at
each of A-D and position bias cancels. Every name is rendered as "F. Lastname"
(gts.matching.display_name), so no option stands out by format.

A paper counts as recognised when the model picks the true list in at least
three of the four orders; pure guessing gets there for about 5% of papers.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from gts.matching import display_name, name_key, overlaps_any, split_answer
from gts.prompts import HIDDEN_TITLE, OPTION_LABELS

N_ORDERS = 4


def paper_rng(paper_id: str, *salt) -> np.random.Generator:
    """An independent, reproducible stream per paper and purpose."""
    key = "|".join([str(paper_id), *map(str, salt)]).encode("utf-8")
    return np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:8], "big"))



def conflicts(names: list[str], truth: list[str]) -> bool:
    """True if any name could be taken for a true author: a match at any level
    (L5 included), or the same surname even with a different initial. Stricter
    than scoring, as in the paper's own distractor construction."""
    if overlaps_any(names, truth):
        return True
    surnames = {name_key(t)[0] for t in truth} - {""}
    return any(name_key(n)[0] in surnames for n in names)

def format_list(names: list[str]) -> list[str]:
    return [display_name(n) for n in names]


def cyclic_orders(true_option: list[str], distractors: list[list[str]]):
    """Yield (order, options); in order r the true list is option r (A=0 ... D=3)."""
    base = [true_option, *distractors]
    for r in range(N_ORDERS):
        yield r, base[-r:] + base[:-r] if r else list(base)


def _items_for(paper: dict, level: str, condition: str, true_option: list[str],
               distractors: list[list[str]], title: str, year, meta: dict) -> list[dict]:
    items = []
    for order, options in cyclic_orders(true_option, distractors):
        items.append({
            "item_id": f"{level}|{condition}|{paper['paper_id']}|o{order}",
            "paper_id": paper["paper_id"],
            "level": level,
            "condition": condition,
            "order": order,
            "correct_label": OPTION_LABELS[order],
            "title": title,
            "year": int(year),
            "options": options,
            **meta,
        })
    return items


# --------------------------------------------------------------------------
# real_team
# --------------------------------------------------------------------------

class RealTeamBuilder:
    """Distractors from the real author teams of other papers in the sample.

    Donors come from the same field with the same number of authors. Where
    fewer than three such papers are usable (long author lists), longer teams
    from the same field are truncated to the same length, and as a last
    resort other fields are used; the item records which rule applied.
    A donor is unusable if any of its authors matches a true author at any
    level, L5 included, or if it renders identically to an option already
    chosen.

    With ``year_window``, donors published within that many years of the
    paper come first (the L2 level), so the era of the names is no cue;
    papers without three such donors fall back to the rules above.
    """

    def __init__(self, papers: pd.DataFrame, year_window: int | None = None):
        self.year_window = year_window
        self.papers = {r["paper_id"]: r for r in papers.to_dict("records")}
        by_field_k: dict[tuple[str, int], list[str]] = defaultdict(list)
        for pid, rec in self.papers.items():
            by_field_k[(rec["field"], len(rec["authors"]))].append(pid)
        self.by_field_k = {key: sorted(ids) for key, ids in by_field_k.items()}

    def _candidates(self, paper: dict):
        """Donor ids in preference order, each tagged with the rule that admitted it."""
        field, k, pid = paper["field"], len(paper["authors"]), paper["paper_id"]
        gen = paper_rng(pid, "real_team")

        def shuffled(ids):
            ids = [i for i in ids if i != pid]
            return [ids[j] for j in gen.permutation(len(ids))]

        longer = sorted(kk for (f, kk) in self.by_field_k if f == field and kk > k)
        if self.year_window is not None:
            def near(d):
                return abs(int(self.papers[d]["year"]) - int(paper["year"])) <= self.year_window
            yield from ((d, "same_length_year") for d in shuffled(self.by_field_k.get((field, k), [])) if near(d))
            for kk in longer:
                yield from ((d, "truncated_year") for d in shuffled(self.by_field_k[(field, kk)]) if near(d))
        yield from ((d, "same_length") for d in shuffled(self.by_field_k.get((field, k), [])))
        for kk in longer:
            yield from ((d, "truncated") for d in shuffled(self.by_field_k[(field, kk)]))
        others = sorted((f, kk) for (f, kk) in self.by_field_k if f != field and kk >= k)
        for key in others:
            yield from ((d, "other_field") for d in shuffled(self.by_field_k[key]))

    def distractors(self, paper: dict) -> tuple[list[list[str]], list[str], list[str]]:
        """Three formatted distractor lists, the donor ids and the rule for each."""
        k = len(paper["authors"])
        true_fmt = format_list(paper["authors"])
        chosen, donors, rules = [], [], []
        for donor_id, rule in self._candidates(paper):
            donor_authors = self.papers[donor_id]["authors"][:k]
            if conflicts(donor_authors, paper["authors"]):
                continue
            option = format_list(donor_authors)
            # the model sees the formatted names, so check those too
            if option == true_fmt or option in chosen or conflicts(option, paper["authors"]):
                continue
            chosen.append(option)
            donors.append(donor_id)
            rules.append(rule)
            if len(chosen) == 3:
                return chosen, donors, rules
        raise ValueError(f"fewer than 3 usable donors for paper {paper['paper_id']}")


def control_subset(papers: pd.DataFrame, per_stratum: int = 10) -> list[str]:
    """Up to ``per_stratum`` papers from every field x citation bin, chosen reproducibly."""
    chosen = []
    for _, group in papers.groupby(["field", "citation_bin"], sort=True):
        ids = sorted(group["paper_id"], key=lambda pid: hashlib.sha256(f"ctrl|{pid}".encode()).hexdigest())
        chosen.extend(ids[:per_stratum])
    return sorted(chosen)


def swap_partner(paper: dict, papers: pd.DataFrame, exclude_authors: list[list[str]]) -> dict:
    """Another paper from the same field and bin whose authors overlap none of the options."""
    pool = papers[(papers["field"] == paper["field"])
                  & (papers["citation_bin"] == paper["citation_bin"])
                  & (papers["paper_id"] != paper["paper_id"])]
    pool = pool.sort_values("paper_id").to_dict("records")
    gen = paper_rng(paper["paper_id"], "title_swap")
    for j in gen.permutation(len(pool)):
        cand = pool[j]
        if not any(conflicts(cand["authors"], names) for names in exclude_authors):
            return cand
    raise ValueError(f"no swap partner for paper {paper['paper_id']}")


def build_real_team_items(papers: pd.DataFrame, control_ids: list[str], level: str = "real_team",
                          year_window: int | None = None) -> list[dict]:
    """Standard items for every paper, plus both controls on ``control_ids``."""
    builder = RealTeamBuilder(papers, year_window=year_window)
    control_ids = set(control_ids)
    items = []
    for paper in papers.sort_values("paper_id").to_dict("records"):
        distractors, donors, rules = builder.distractors(paper)
        true_fmt = format_list(paper["authors"])
        meta = {"donors": donors, "donor_rules": rules}
        items += _items_for(paper, level, "standard", true_fmt, distractors,
                            paper["title"], paper["year"], meta)
        if paper["paper_id"] in control_ids:
            items += _items_for(paper, level, "title_hidden", true_fmt, distractors,
                                HIDDEN_TITLE, paper["year"], meta)
            donor_authors = [builder.papers[d]["authors"] for d in donors]
            partner = swap_partner(paper, papers, [paper["authors"], *donor_authors])
            items += _items_for(paper, level, "title_swapped", true_fmt, distractors,
                                partner["title"], partner["year"],
                                meta | {"title_from": partner["paper_id"]})
    return items


# --------------------------------------------------------------------------
# own_sample
# --------------------------------------------------------------------------

def _keys(option: list[str]) -> set[tuple[str, str]]:
    return {name_key(n) for n in option}


def wrong_lists(truth: list[str], texts: list[str]) -> tuple[list[tuple[list[str], int]], list[str]]:
    """The model's sampled lists that share no name with the truth at any level.

    Returns the distinct lists, formatted as the options will show them and
    most frequent first (ties: first seen), with their counts; and the pool of
    their names, most frequent first.
    """
    list_counts: Counter = Counter()
    first_seen: dict = {}
    name_counts: Counter = Counter()
    for i, text in enumerate(texts):
        raw = split_answer(text)
        if not raw or conflicts(raw, truth):
            continue
        formatted, seen = [], set()
        for name in format_list(raw):
            key = name_key(name)
            if key[0] and key not in seen:
                seen.add(key)
                formatted.append(name)
        if not formatted or conflicts(formatted, truth):
            continue
        key = tuple(formatted)
        list_counts[key] += 1
        first_seen.setdefault(key, i)
        for name in formatted:
            name_counts[name] += 1
    ordered = sorted(list_counts, key=lambda k: (-list_counts[k], first_seen[k]))
    pool = [n for n, _ in name_counts.most_common()]
    return [(list(k), list_counts[k]) for k in ordered], pool


def own_sample_distractors(truth: list[str], texts: list[str]) -> tuple[list[list[str]], list[str], dict]:
    """Up to three wrong options built from the model's own samples for this paper.

    The model's most frequent wrong lists come first, cut or padded (with its
    other wrong names, most frequent first) to the true list's length. The
    three options share no name with the truth or with each other, as three
    real teams would. If the samples do not yield three such lists, the
    remaining names of the pool are recombined; the caller fills any slot
    still empty from real teams.
    """
    k = len(truth)
    true_fmt = format_list(truth)
    lists, pool = wrong_lists(truth, texts)
    chosen, rules, used = [], [], set()

    def accept(option, rule):
        keys = _keys(option)
        if len(option) != k or len(keys) != k or option == true_fmt or keys & used:
            return False
        chosen.append(option)
        rules.append(rule)
        used.update(keys)
        return True

    for names, _count in lists:
        if len(chosen) == 3:
            break
        if len(names) >= k:
            accept(names[:k], "sampled" if len(names) == k else "sampled_cut")
        else:
            fill = [n for n in pool if n not in names and name_key(n) not in used]
            accept(names + fill[: k - len(names)], "sampled_padded")
    spare = [n for n in pool if name_key(n) not in used]
    while len(chosen) < 3 and len(spare) >= k:
        option, spare = spare[:k], spare[k:]
        accept(option, "pooled")
    return chosen, rules, {"n_wrong_lists": len(lists), "n_pool_names": len(pool)}


def build_own_sample_items(papers: pd.DataFrame, samples: dict[str, list[str]],
                           control_ids: list[str]) -> list[dict]:
    """Standard items for every sampled paper, plus both controls on ``control_ids``."""
    real_team = RealTeamBuilder(papers)
    control_ids = set(control_ids)
    items = []
    for paper in papers.sort_values("paper_id").to_dict("records"):
        texts = samples.get(paper["paper_id"])
        if texts is None:
            continue
        distractors, rules, stats = own_sample_distractors(paper["authors"], texts)
        if len(distractors) < 3:
            used = set().union(*(_keys(o) for o in distractors)) if distractors else set()
            for option in real_team.distractors(paper)[0]:
                if len(distractors) == 3:
                    break
                if not _keys(option) & used:
                    distractors.append(option)
                    rules.append("real_team_fallback")
                    used |= _keys(option)
        if len(distractors) < 3:
            raise ValueError(f"could not build three distractors for {paper['paper_id']}")
        true_fmt = format_list(paper["authors"])
        meta = {"distractor_rules": rules, **stats}
        items += _items_for(paper, "own_sample", "standard", true_fmt, distractors,
                            paper["title"], paper["year"], meta)
        if paper["paper_id"] in control_ids:
            items += _items_for(paper, "own_sample", "title_hidden", true_fmt, distractors,
                                HIDDEN_TITLE, paper["year"], meta)
            partner = swap_partner(paper, papers, [paper["authors"], *distractors])
            items += _items_for(paper, "own_sample", "title_swapped", true_fmt, distractors,
                                partner["title"], partner["year"],
                                meta | {"title_from": partner["paper_id"]})
    return items


# --------------------------------------------------------------------------
# candidate multiple choice (selection among the model's own top candidates)
# --------------------------------------------------------------------------

def build_cand_mc_items(papers: pd.DataFrame, samples: dict[str, list[str]]) -> list[dict]:
    """Four cyclic orders of each paper's four most frequent sampled names, plus
    "E: None of these". Correct letters: any true author, or E when none is.

    Unlike the recognition levels, the options come from the method itself, so
    this measures selection: can the model point at a true author among its
    own candidates better than picking the most frequent one?
    """
    from gts.matching import gt_names
    from gts.prompts import CAND_LABELS, cand_mc_user_text
    from gts.selection import candidate_pool, label_candidates

    items = []
    for paper in papers.sort_values("paper_id").to_dict("records"):
        pid = paper["paper_id"]
        texts = samples.get(pid)
        if texts is None:
            continue
        pool = label_candidates(candidate_pool(texts, n=4), gt_names(paper["gt_str"]))
        if len(pool) < 4:
            continue
        for order in range(N_ORDERS):
            rotated = pool[-order:] + pool[:-order] if order else list(pool)
            names = [c["name"] for c in rotated]
            true_letters = [OPTION_LABELS[i] for i, c in enumerate(rotated) if c["is_true"]]
            items.append({
                "item_id": f"cand_mc|standard|{pid}|o{order}",
                "paper_id": pid, "level": "cand_mc", "condition": "standard",
                "order": order, "title": paper["title"], "year": int(paper["year"]),
                "labels": list(CAND_LABELS), "options": names,
                "user_text": cand_mc_user_text(paper["title"], paper["year"], names),
                "ranks": [c["rank"] for c in rotated], "counts": [c["count"] for c in rotated],
                "true_letters": true_letters, "correct_letters": true_letters or ["E"],
            })
    return items


# --------------------------------------------------------------------------
# checks and I/O
# --------------------------------------------------------------------------

def verify_items(items: list[dict], papers: pd.DataFrame) -> dict:
    """Integrity checks: distractors disjoint from the truth, no repeated
    options, the true list at each position equally often, four orders per
    paper and condition."""
    authors = dict(zip(papers["paper_id"], papers["authors"]))
    overlapping = duplicate_options = 0
    positions = Counter()
    per_group = Counter()
    for item in items:
        positions[item["correct_label"]] += 1
        per_group[(item["level"], item["condition"], item["paper_id"])] += 1
        opts = [tuple(o) for o in item["options"]]
        if len(set(opts)) != 4:
            duplicate_options += 1
        if item["order"] == 0:
            truth = authors[item["paper_id"]]
            wrong = item["options"][1:]
            # display names are what the model sees; check them against the raw truth
            if any(conflicts(opt, truth) for opt in wrong):
                overlapping += 1
    total = sum(positions.values()) or 1
    return {
        "n_items": len(items),
        "by_condition": dict(Counter(f"{i['level']}/{i['condition']}" for i in items)),
        "distractor_groups_overlapping_truth": overlapping,
        "items_with_duplicate_options": duplicate_options,
        "correct_position_share": {k: round(v / total, 4) for k, v in sorted(positions.items())},
        "groups_without_4_orders": sum(1 for v in per_group.values() if v != N_ORDERS),
        "distractor_rules": dict(Counter(
            r for i in items if i["order"] == 0 and i["condition"] == "standard"
            for r in i.get("donor_rules", i.get("distractor_rules", [])))),
    }


def write_items(items: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def read_items(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
