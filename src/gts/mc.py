"""Multiple-choice recognition items: distractors, option orders and controls.

Levels (where the three wrong options come from), the proposal's ladder L0-L5:
    random_field    L0  random names from the paper's field (the first test's design)
    real_team       L1  the author lists of three other papers in the same field with
                        the same number of authors (the team test's design)
    real_team_year  L2  the same, published within two years of the paper
    similar_title   L3  the same, from the most similar titles, with teams of similar
                        prominence (their best-known member's h-index)
    similar_topic   L3t the most similar titles only, any year (isolates the topic cue)
    collab_swap     L4  the true list with one author replaced by a real collaborator
                        of the first author
    own_sample      L5  wrong author lists the model itself sampled for this paper;
                        built once the sampling run is back (see build_own_sample_items)

Catch trials (build_catch_items) add "None of the above" as an option and remove
the true list from half of the items, which measures false recognition.

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
# random_field (L0), similar_title (L3), collab_swap (L4)
# --------------------------------------------------------------------------

def build_level_items(papers: pd.DataFrame, control_ids: list[str], level: str, distractor_fn) -> list[dict]:
    """Standard items for every paper, plus both controls on ``control_ids``.

    ``distractor_fn(paper)`` returns the three formatted wrong options and a
    dict of metadata that every item of the paper records.
    """
    control_ids = set(control_ids)
    items = []
    for paper in papers.sort_values("paper_id").to_dict("records"):
        distractors, meta = distractor_fn(paper)
        true_fmt = format_list(paper["authors"])
        items += _items_for(paper, level, "standard", true_fmt, distractors,
                            paper["title"], paper["year"], meta)
        if paper["paper_id"] in control_ids:
            items += _items_for(paper, level, "title_hidden", true_fmt, distractors,
                                HIDDEN_TITLE, paper["year"], meta)
            partner = swap_partner(paper, papers, [paper["authors"], *distractors])
            items += _items_for(paper, level, "title_swapped", true_fmt, distractors,
                                partner["title"], partner["year"],
                                meta | {"title_from": partner["paper_id"]})
    return items


class RandomFieldBuilder:
    """Random names from the paper's field (L0), as the release's random_field.

    The pool is every author name on the field's papers in the sample. A drawn
    name may not share a surname with a true author, or a (surname, initial)
    key with another drawn name, so the three lists of k names are disjoint
    from the truth and from each other.
    """

    def __init__(self, papers: pd.DataFrame):
        pools: dict[str, set[str]] = defaultdict(set)
        for field, authors in zip(papers["field"], papers["authors"]):
            pools[field].update(n for n in (" ".join(str(a).split()) for a in authors) if n)
        self.pools = {f: sorted(names) for f, names in pools.items()}

    def draw(self, paper: dict, n: int, salt: str, used: set | None = None) -> list[str]:
        """``n`` formatted names. ``used`` holds (surname, initial) keys to avoid
        and gains the keys of the names drawn."""
        truth = paper["authors"]
        surnames = {name_key(t)[0] for t in truth} - {""}
        used = set() if used is None else used
        pool = self.pools[paper["field"]]
        picked = []
        for j in paper_rng(paper["paper_id"], salt).permutation(len(pool)):
            shown = display_name(pool[j])
            key = name_key(shown)
            if not key[0] or key in used or key[0] in surnames or conflicts([pool[j], shown], truth):
                continue
            used.add(key)
            picked.append(shown)
            if len(picked) == n:
                return picked
        raise ValueError(f"field pool exhausted for paper {paper['paper_id']}")

    def distractors(self, paper: dict) -> tuple[list[list[str]], dict]:
        k = len(paper["authors"])
        names = self.draw(paper, 3 * k, "random_field")
        return [names[i * k:(i + 1) * k] for i in range(3)], {}


def team_prominence(s2_authors: list[dict] | None) -> float:
    """log(1 + h-index) of the team's best-known member; NaN when no h-index is known."""
    hs = [a["hIndex"] for a in (s2_authors or []) if a and a.get("hIndex") is not None]
    return float(np.log1p(max(hs))) if hs else float("nan")


class SimilarTitleBuilder(RealTeamBuilder):
    """Real teams of the most similar titles in the field (L3).

    Donors are ranked by the TF-IDF cosine similarity of their titles to the
    paper's, within tiers that relax one constraint at a time: the same author
    count, published within two years and a team prominence (see
    team_prominence) within the first, then the second tolerance; longer teams
    truncated; no prominence constraint; no year constraint. The real-team
    rules (other fields) are the last resort. Each donor records its tier, its
    title similarity and its prominence gap.
    """

    PROMINENCE_TOLERANCE = (0.25, 0.5)

    def __init__(self, papers: pd.DataFrame, prominence: dict[str, float], year_window: int = 2,
                 topic_only: bool = False):
        """``topic_only`` gives the L3t variant: the most similar titles from teams at
        least as long (truncated), with no year or prominence constraint. Within the
        sample, year and team-size constraints leave few topical neighbours, so this
        variant isolates the topic cue (compare with L1), and L3 the prominence cue
        (compare with L2)."""
        from sklearn.feature_extraction.text import TfidfVectorizer

        super().__init__(papers, year_window=year_window)
        self.prominence = prominence
        self.topic_only = topic_only
        self.fields = {}
        for field, group in papers.groupby("field", sort=True):
            ids = sorted(group["paper_id"])
            titles = group.set_index("paper_id").loc[ids, "title"].astype(str)
            tfidf = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit_transform(titles)
            sims = (tfidf @ tfidf.T).toarray().astype(np.float32)
            self.fields[field] = (ids, sims, {p: i for i, p in enumerate(ids)})

    def _gap(self, a: str, b: str) -> float:
        pa, pb = self.prominence.get(a, np.nan), self.prominence.get(b, np.nan)
        return float("inf") if np.isnan(pa) or np.isnan(pb) else abs(pa - pb)

    def distractors(self, paper: dict) -> tuple[list[list[str]], dict]:
        pid, k, year = paper["paper_id"], len(paper["authors"]), int(paper["year"])
        ids, sims, index = self.fields[paper["field"]]
        i = index[pid]
        ranked = [j for j in np.argsort(-sims[i], kind="stable") if j != i]
        tol1, tol2 = self.PROMINENCE_TOLERANCE

        def near(d):
            return abs(int(self.papers[d]["year"]) - year) <= self.year_window

        def same(d):
            return len(self.papers[d]["authors"]) == k

        def longer(d):
            return len(self.papers[d]["authors"]) > k

        tiers = [("most_similar_title", lambda d: same(d) or longer(d))] if self.topic_only else [
            ("same_length_year_prominence", lambda d: same(d) and near(d) and self._gap(pid, d) <= tol1),
            ("same_length_year_prominence2", lambda d: same(d) and near(d) and self._gap(pid, d) <= tol2),
            ("truncated_year_prominence2", lambda d: longer(d) and near(d) and self._gap(pid, d) <= tol2),
            ("same_length_year", lambda d: same(d) and near(d)),
            ("truncated_year", lambda d: longer(d) and near(d)),
            ("same_length", same),
            ("truncated", longer),
        ]
        true_fmt = format_list(paper["authors"])
        chosen, meta = [], {"donors": [], "donor_rules": [], "title_similarity": [], "prominence_gap": []}

        def take(d, rule, sim):
            donor_authors = self.papers[d]["authors"][:k]
            if d in meta["donors"] or conflicts(donor_authors, paper["authors"]):
                return False
            option = format_list(donor_authors)
            if option == true_fmt or option in chosen or conflicts(option, paper["authors"]):
                return False
            chosen.append(option)
            gap = self._gap(pid, d)
            meta["donors"].append(d)
            meta["donor_rules"].append(rule)
            meta["title_similarity"].append(round(float(sim), 4))
            meta["prominence_gap"].append(None if np.isinf(gap) else round(gap, 4))
            return len(chosen) == 3

        for rule, ok in tiers:
            for j in ranked:
                if ok(ids[j]) and take(ids[j], rule, sims[i, j]):
                    return chosen, meta
        for d, rule in self._candidates(paper):
            if take(d, rule, 0.0):
                return chosen, meta
        raise ValueError(f"fewer than 3 usable donors for paper {pid}")


class CollaboratorSwapBuilder:
    """The true list with one author swapped for a real collaborator (L4).

    As the release's collaborator_swap: each replacement is a co-author of the
    true first author (Semantic Scholar, tools/fetch_s2.py); the first author
    stays in place and one other position changes (the only one for a single
    author). Unlike the release, which draws three collaborators at random,
    the three most frequent usable ones are taken, so the swaps are the most
    plausible teams. Below three usable collaborators, random field names fill
    the rest (the release's same_field_swap fallback), recorded per option.

    With ``shared_position`` (collab_slot, the L4 used for labels) all three
    distractors replace the same position, so the four options share the same
    k-1 names and differ only there. With a position drawn per distractor (the
    release's design, collab_swap) the true list is the one whose every name
    recurs in other options, which can be found without knowing the paper.
    """

    def __init__(self, papers: pd.DataFrame, collaborators: dict[str, list[str]], shared_position: bool = False):
        self.field = RandomFieldBuilder(papers)
        self.collaborators = collaborators
        self.shared_position = shared_position

    def distractors(self, paper: dict) -> tuple[list[list[str]], dict]:
        pid, truth = paper["paper_id"], paper["authors"]
        k = len(truth)
        true_fmt = format_list(truth)
        surnames = {name_key(t)[0] for t in truth} - {""}
        used = {name_key(n) for n in true_fmt}
        names, rules = [], []
        for raw in self.collaborators.get(pid, []):
            shown = display_name(raw)
            key = name_key(shown)
            if not key[0] or key in used or key[0] in surnames or conflicts([raw, shown], truth):
                continue
            used.add(key)
            names.append(shown)
            rules.append("collaborator")
            if len(names) == 3:
                break
        if len(names) < 3:
            extra = self.field.draw(paper, 3 - len(names), "collab_swap_fallback", used)
            names += extra
            rules += ["field_name"] * len(extra)
        gen = paper_rng(pid, "collab_swap")
        # drawn only when shared, so collab_swap items rebuild exactly as they were run
        shared = int(gen.integers(1, k)) if self.shared_position and k > 1 else 0
        options, positions = [], []
        for name in names:
            position = 0 if k == 1 else shared if self.shared_position else int(gen.integers(1, k))
            option = list(true_fmt)
            option[position] = name
            options.append(option)
            positions.append(position)
        return options, {"swap_rules": rules, "swap_positions": positions}


# --------------------------------------------------------------------------
# catch trials and the other recognition formats (yes/no, likelihood)
# --------------------------------------------------------------------------

def build_catch_items(papers: pd.DataFrame, real_team_items: list[dict]) -> list[dict]:
    """Catch trials on the real-team (L1) distractors, with "None of the above" as an option.

    with_true  the true list, the first two real-team distractors and None
    catch      the three real-team distractors and None; None is correct

    Both are asked in the four cyclic orders, so every option, None included,
    sits once at each letter.
    """
    from gts.prompts import NONE_OPTION, mc_none_user_text

    base = {it["paper_id"]: it for it in real_team_items
            if it["condition"] == "standard" and it["order"] == 0}
    items = []
    for paper in papers.sort_values("paper_id").to_dict("records"):
        ref = base[paper["paper_id"]]
        true, wrong = ref["options"][0], ref["options"][1:]
        for condition, options, correct in (("with_true", [true, wrong[0], wrong[1], NONE_OPTION], 0),
                                            ("catch", [wrong[0], wrong[1], wrong[2], NONE_OPTION], 3)):
            for order in range(N_ORDERS):
                rotated = options[-order:] + options[:-order] if order else list(options)
                items.append({
                    "item_id": f"catch|{condition}|{paper['paper_id']}|o{order}",
                    "paper_id": paper["paper_id"], "level": "catch", "condition": condition,
                    "order": order, "correct_label": OPTION_LABELS[(correct + order) % N_ORDERS],
                    "none_label": OPTION_LABELS[(3 + order) % N_ORDERS],
                    "title": paper["title"], "year": int(paper["year"]), "options": rotated,
                    "user_text": mc_none_user_text(paper["title"], paper["year"], rotated),
                })
    return items


def standard_option_sets(items: list[dict]) -> dict[str, dict]:
    """Per paper, the order-0 standard item of a level: options[0] is the true list."""
    return {it["paper_id"]: it for it in items if it["condition"] == "standard" and it["order"] == 0}


def build_yesno_records(items_by_level: dict[str, list[dict]]) -> list[dict]:
    """One yes/no question per option list of each level's standard items."""
    from gts.prompts import YESNO_SYSTEM, yesno_list_user_text

    records = []
    for level, items in items_by_level.items():
        for pid, it in sorted(standard_option_sets(items).items()):
            for j, option in enumerate(it["options"]):
                records.append({
                    "record_id": f"{level}|{pid}|{j}", "paper_id": pid, "level": level,
                    "list_index": j, "is_true": j == 0, "system": YESNO_SYSTEM,
                    "user_text": yesno_list_user_text(option, it["title"], it["year"]),
                })
    return records


def build_list_score_records(items_by_level: dict[str, list[dict]]) -> list[dict]:
    """Each option list of each level as the answer to the open question, with the
    title shown ("standard") and hidden ("hidden"); see gts.prompts.score_texts."""
    records = []
    for level, items in items_by_level.items():
        for pid, it in sorted(standard_option_sets(items).items()):
            for j, option in enumerate(it["options"]):
                for condition, title in (("standard", it["title"]), ("hidden", HIDDEN_TITLE)):
                    records.append({
                        "record_id": f"list|{level}|{condition}|{pid}|{j}", "kind": "list",
                        "paper_id": pid, "level": level, "condition": condition, "list_index": j,
                        "is_true": j == 0, "title": title, "year": it["year"], "names": option,
                    })
    return records


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
    paper and condition.

    collab_swap distractors share all but one name with the truth by design:
    there the check is that exactly one position differs and its new name
    could not be taken for a true author. Catch items are checked for repeats
    and positions only."""
    authors = dict(zip(papers["paper_id"], papers["authors"]))
    overlapping = duplicate_options = 0
    positions = Counter()
    per_group = Counter()
    for item in items:
        positions[item["correct_label"]] += 1
        per_group[(item["level"], item["condition"], item["paper_id"])] += 1
        opts = [tuple(o) if isinstance(o, list) else (o,) for o in item["options"]]
        if len(set(opts)) != 4:
            duplicate_options += 1
        if item["order"] == 0 and item["level"] != "catch":
            truth = authors[item["paper_id"]]
            true_fmt, wrong = item["options"][0], item["options"][1:]
            if item["level"] in ("collab_swap", "collab_slot"):
                for opt in wrong:
                    diff = [i for i, (a, b) in enumerate(zip(opt, true_fmt)) if a != b]
                    if len(opt) != len(true_fmt) or len(diff) != 1 or conflicts([opt[diff[0]]], truth):
                        overlapping += 1
                        break
            # display names are what the model sees; check them against the raw truth
            elif any(conflicts(opt, truth) for opt in wrong):
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
            for r in i.get("donor_rules", i.get("distractor_rules", i.get("swap_rules", []))))),
    }


def write_items(items: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def read_items(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]
