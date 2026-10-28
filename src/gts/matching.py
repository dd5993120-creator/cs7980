"""Name matching on the scale of the shipped Qwen3-32B run.

The vendored matcher (gts.dua_matcher) is the one that scored the shipped run.
This module adds a parse cache, so scoring hundreds of thousands of sampled
lists stays fast, and the helpers the analyses need:

    score_answer        the paper's per-answer (M, H, U) and rates
    gt_names            the ground-truth list exactly as the scorer sees it
    matched_gt_indices  which true authors an answer names (L1-L4)
    name_key            (surname, first initial): the identity L4 matches on
    display_name        one name rendered as "F. Lastname" in ASCII
"""

from __future__ import annotations

import re
from functools import lru_cache

from nameparser import HumanName
from unidecode import unidecode

from gts.dua_matcher import MATCH_LEVEL_THRESHOLD, NameMatcher, _match_level_value, score_authors


class CachedMatcher(NameMatcher):
    """NameMatcher whose per-name parse is cached.

    find_matches only copies the parsed dict and never mutates it, so sharing
    one parse across calls is safe.
    """

    def __init__(self, cache_size: int = 1 << 20):
        super().__init__()
        self._parse_cached = lru_cache(maxsize=cache_size)(super()._parse_single_name)

    def _parse_single_name(self, name_str: str):
        return self._parse_cached(name_str)


_MATCHER: CachedMatcher | None = None


def get_matcher() -> CachedMatcher:
    global _MATCHER
    if _MATCHER is None:
        _MATCHER = CachedMatcher()
    return _MATCHER


def clean_generated_text(text: str) -> str:
    """Strip Qwen3 artifacts before scoring.

    Copied from dua.probing.stages.stage2_score.clean_generated_text, which
    produced the shipped scores.
    """
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"(?i)^(the\s+)?authors?\s*(of[^:]*)?:\s*", "", text.strip())
    text = text.strip().strip('"').strip("'").lstrip(";,，；").strip()
    return text


def score_answer(gt_str: str, answer: str, matcher: CachedMatcher | None = None) -> dict:
    """(M, H, U), HR2/4/5 and name counts for one answer, as stage2_score computed them."""
    return score_authors(gt_str, clean_generated_text(answer), matcher or get_matcher())


def gt_names(gt_str: str, matcher: CachedMatcher | None = None) -> list[str]:
    """The ground-truth list as score_answer splits it; indices refer to this list."""
    return (matcher or get_matcher()).splitter.split(gt_str) if gt_str else []


def _name_words(name: str, matcher: CachedMatcher) -> frozenset[str]:
    """Words of the name's alpha-only normalisation, as the matcher compares them."""
    return frozenset(matcher._parse_single_name(name)["normalized_alpha"].split())


def _anchor_words(name: str, matcher: CachedMatcher) -> frozenset[str]:
    """Words any match with this (ground-truth) name must share with the other name.

    L1-L3 need equal full names; L3.5-L5 and both rescues need the other
    name's (possibly rearranged) surname to equal this surname, or both
    particle-stripped surnames to be equal and non-empty. The stripped surname
    is a subset of the surname, so in every case the other name contains every
    word of this stripped surname (else of the surname, else of the whole name).
    """
    parsed = matcher._parse_single_name(name)
    alpha = parsed["parsed_alpha"]
    words = alpha["last_stripped"] or alpha["last"] or parsed["normalized_alpha"]
    return frozenset(words.split())


def could_match(gt_list: list[str], names: list[str], matcher: CachedMatcher | None = None) -> bool:
    """A necessary condition for any match between the two lists, at any level.

    Cheap, and false for most pairs of unrelated lists, so the full matcher
    only runs where a match is possible.
    """
    m = matcher or get_matcher()
    anchors = [_anchor_words(g, m) for g in gt_list]
    words = frozenset().union(*(_name_words(n, m) for n in names)) if names else frozenset()
    return any(a and a <= words for a in anchors)


def matched_gt_indices(gt_list: list[str], answer: str,
                       matcher: CachedMatcher | None = None) -> set[int]:
    """Indices into ``gt_list`` of the true authors the answer names at L1-L4."""
    m = matcher or get_matcher()
    cleaned = clean_generated_text(answer)
    if not could_match(gt_list, m.splitter.split(cleaned), m):
        return set()
    result = m.find_matches(gt_list, cleaned)
    return {x["gt_index"] for x in result["matches"]
            if _match_level_value(x["match_level"]) <= MATCH_LEVEL_THRESHOLD}


def split_answer(answer: str, matcher: CachedMatcher | None = None) -> list[str]:
    """The individual names in an answer, as the scorer splits them."""
    return (matcher or get_matcher()).splitter.split(clean_generated_text(answer))


def name_key(name: str, matcher: CachedMatcher | None = None) -> tuple[str, str]:
    """(particle-stripped surname, first initial), lower case.

    Two names with the same key match at L4 or better in the normal
    orientation. The swap and bibliographic rescues are not covered, so use
    the full matcher where an exact guarantee is needed.
    """
    parsed = (matcher or get_matcher())._parse_single_name(name)["parsed_alpha"]
    surname = parsed["last_stripped"] or parsed["last"]
    first = parsed["first"]
    return surname, (first[0] if first else "")


@lru_cache(maxsize=1 << 18)
def display_name(name: str) -> str:
    """Render a name as "F. Lastname" in ASCII, so no option stands out by format.

    Ground truth mixes "S. Bree" with "Suzanne Duijst" and keeps diacritics,
    while the model writes full ASCII names; normalising every option to one
    format removes that cue.
    """
    spaced = re.sub(r"([A-Z])\.([A-Z])", r"\1. \2", unidecode(name).strip())
    parsed = HumanName(spaced)
    last = " ".join(parsed.last.split())
    first = parsed.first.strip(" .")
    if not last:
        return " ".join(spaced.split())
    return f"{first[0].upper()}. {last}" if first else last


def overlaps_any(names: list[str], others: list[str],
                 matcher: CachedMatcher | None = None) -> bool:
    """True if any name in ``names`` matches any name in ``others`` at any level, L5 included.

    Used to keep distractors disjoint from the truth: stricter than scoring,
    which ignores L5.
    """
    if not names or not others:
        return False
    m = matcher or get_matcher()
    if not could_match(others, names, m):
        return False
    return bool(m.find_matches(others, "; ".join(names))["matches"])
