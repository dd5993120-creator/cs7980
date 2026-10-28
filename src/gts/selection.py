"""Selection on the candidate pools: frequency and P(True).

Each paper's pool is the distinct names across its 50 sampled lists (merged on
surname and first initial), ranked by how many lists name them. The top
candidates are then scored by asking the model whether each one is an author
(Kadavath et al., 2022, P(True) on single candidates). The key ablation is
whether P(True) separates true from false candidates better than frequency.
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
    key are merged and shown in their most frequent form.
    """
    lists_with, first_seen, forms = Counter(), {}, {}
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
    ranked = sorted(lists_with, key=lambda k: (-lists_with[k], first_seen[k]))[:n]
    return [{"name": forms[k].most_common(1)[0][0], "count": lists_with[k], "rank": r}
            for r, k in enumerate(ranked)]


def label_candidates(candidates: list[dict], gt_list: list[str]) -> list[dict]:
    """Mark each candidate that names a true author at L1-L4."""
    out = []
    for c in candidates:
        hit = matched_gt_indices(gt_list, c["name"])
        out.append(c | {"is_true": bool(hit), "gt_index": min(hit) if hit else None})
    return out
