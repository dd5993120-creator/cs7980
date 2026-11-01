"""Selection on the candidate pools: frequency, P(True) and likelihood scores.

Each paper's pool is the distinct names across its 50 sampled lists (merged on
surname and first initial), ranked by how many lists name them. The top
candidates are then scored by asking the model whether each one is an author
(Kadavath et al., 2022, P(True) on single candidates). The key ablation is
whether P(True) separates true from false candidates better than frequency.
Session 3 adds likelihood scores of the same candidates: a PMI-corrected
log P(author | title) and the authors-to-title direction.
"""

from __future__ import annotations

from collections import Counter

from gts.matching import display_name, matched_gt_indices, name_key, split_answer

N_CANDIDATES = 10

PTRUE_SYSTEM = (
    "You are a knowledgeable assistant. Answer factually and concisely. "
    "Answer with a single word: Yes or No."
)


def ptrue_user_text(name: str, title: str, year) -> str:
    return (f"Is {name} one of the authors of the paper titled '{title}' "
            f"which was published in {year}? Answer Yes or No.")


def candidate_pool(texts: list[str], n: int = N_CANDIDATES) -> list[dict]:
    """The n names that the most sampled lists contain (ties: first seen).

    Each list counts a name once; surface variants of one (surname, initial)
    key are merged and shown in their most frequent form ("name", as
    "F. Lastname"; "raw", as the model wrote it).
    """
    lists_with, first_seen, forms, raw_forms = Counter(), {}, {}, {}
    for i, text in enumerate(texts):
        seen = set()
        for raw in split_answer(text):
            key = name_key(raw)
            if not key[0] or key in seen:
                continue
            seen.add(key)
            lists_with[key] += 1
            first_seen.setdefault(key, i)
            forms.setdefault(key, Counter())[display_name(raw)] += 1
            raw_forms.setdefault(key, Counter())[" ".join(raw.split())] += 1
    ranked = sorted(lists_with, key=lambda k: (-lists_with[k], first_seen[k]))[:n]
    return [{"name": forms[k].most_common(1)[0][0], "raw": raw_forms[k].most_common(1)[0][0],
             "count": lists_with[k], "rank": r}
            for r, k in enumerate(ranked)]


def label_candidates(candidates: list[dict], gt_list: list[str]) -> list[dict]:
    """Mark each candidate that names a true author at L1-L4."""
    out = []
    for c in candidates:
        hit = matched_gt_indices(gt_list, c["name"])
        out.append(c | {"is_true": bool(hit), "gt_index": min(hit) if hit else None})
    return out


def apa_author(name: str) -> str:
    """"P. Shi" -> "Shi, P.", as an author appears in a reference list."""
    first, _, last = name.partition(". ")
    return f"{last}, {first}." if last and len(first) == 1 else name


def build_candidate_score_records(papers, samples: dict[str, list[str]], n: int = N_CANDIDATES) -> list[dict]:
    """Likelihood records for each paper's top candidates (gts.prompts.score_texts).

    author|title   the candidate, as the model writes it most often, as the answer
                   to "name one author of the paper titled ..."
    author|none    the same answer without the title: the name's prior, so the
                   difference is a PMI, log P(a | t) - log P(a)
    title|author   the title after the candidate in a reference-list entry
                   ("Shi, P. et al. (2014). <title>."), the authors-to-title direction
    title|none     the title after "Anonymous (2014).", once per paper
    """
    from gts.matching import gt_names

    rows = []
    for paper in papers.sort_values("paper_id").to_dict("records"):
        pid, texts = paper["paper_id"], samples.get(paper["paper_id"])
        if texts is None:
            continue
        pool = label_candidates(candidate_pool(texts, n), gt_names(paper["gt_str"]))
        base = {"paper_id": pid, "year": int(paper["year"])}
        for c in pool:
            cand = base | {"rank": c["rank"], "name_shown": c["name"], "count": c["count"], "is_true": c["is_true"]}
            rows.append(cand | {"record_id": f"author|title|{pid}|{c['rank']}", "kind": "author",
                                "title": paper["title"], "name": c["raw"]})
            rows.append(cand | {"record_id": f"author|none|{pid}|{c['rank']}", "kind": "author",
                                "title": None, "name": c["raw"]})
            rows.append(cand | {"record_id": f"title|author|{pid}|{c['rank']}", "kind": "title",
                                "title": paper["title"], "author": apa_author(c["name"])})
        if pool:
            rows.append(base | {"record_id": f"title|none|{pid}", "kind": "title", "rank": None,
                                "title": paper["title"], "author": None})
    return rows
