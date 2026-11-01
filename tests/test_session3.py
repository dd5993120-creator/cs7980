import pandas as pd

from gts.generate import score_token_ids
from gts.matching import name_key
from gts.mc import (CollaboratorSwapBuilder, RandomFieldBuilder, SimilarTitleBuilder, build_catch_items,
                    build_level_items, build_list_score_records, build_real_team_items, build_yesno_records,
                    conflicts, team_prominence, verify_items)
from gts.prompts import NONE_OPTION, score_texts
from gts.selection import apa_author, build_candidate_score_records


def _papers(rows):
    return pd.DataFrame([{"paper_id": f"p{i}", "citations": i, "citation_bin": "0-2", "field": "physics", **r}
                         for i, r in enumerate(rows)])


def _toy():
    names = [["Alice Brown", "Carl Dent"], ["Eva Fox", "Gil Hart"], ["Ian Jones", "Kai Lund"],
             ["Max Nye", "Ola Park"], ["Quinn Ross", "Sam Tate"], ["Uma Vance", "Will Xu"]]
    return _papers([{"title": f"Paper {i}", "year": 2000 + i, "authors": a} for i, a in enumerate(names)])


def test_random_field_lists_are_disjoint_from_truth_and_each_other():
    papers = _toy()
    builder = RandomFieldBuilder(papers)
    paper = papers.iloc[0].to_dict()
    options, _ = builder.distractors(paper)
    assert [len(o) for o in options] == [2, 2, 2]
    assert not any(conflicts(o, paper["authors"]) for o in options)
    keys = [name_key(n) for o in options for n in o]
    assert len(set(keys)) == 6
    assert builder.distractors(paper)[0] == options  # reproducible
    report = verify_items(build_level_items(papers, ["p1"], "random_field", builder.distractors), papers)
    assert report["distractor_groups_overlapping_truth"] == 0 and report["groups_without_4_orders"] == 0


def _similar_title_papers():
    rows = [("Electronic band structure of bilayer graphene", 2010), ("Band structure of graphene bilayers", 2011),
            ("Bilayer graphene band structure revisited", 2020), ("Protein folding kinetics", 2010),
            ("Graphene electronic properties", 2012), ("Superconductivity in cuprates", 2009)]
    authors = [["Alice Brown", "Carl Dent"], ["Eva Fox", "Gil Hart"], ["Ian Jones", "Kai Lund"],
               ["Max Nye", "Ola Park"], ["Quinn Ross", "Sam Tate"], ["Uma Vance", "Will Xu"]]
    return _papers([{"title": t, "year": y, "authors": a} for (t, y), a in zip(rows, authors)])


def test_similar_title_ranks_by_title_within_year_and_prominence():
    papers = _similar_title_papers()
    prominence = {f"p{i}": 3.0 for i in range(6)}
    paper = papers.iloc[0].to_dict()
    _, meta = SimilarTitleBuilder(papers, prominence).distractors(paper)
    assert meta["donors"][:2] == ["p1", "p4"]           # most similar titles first
    assert "p2" not in meta["donors"]                    # similar, but published ten years later
    assert set(meta["donor_rules"]) == {"same_length_year_prominence"}
    # a famous team is passed over while equally prominent ones remain
    prominence["p4"] = 5.0
    _, meta = SimilarTitleBuilder(papers, prominence).distractors(paper)
    assert meta["donors"][0] == "p1" and "p4" not in meta["donors"]


def test_similar_topic_ignores_year_and_prominence():
    papers = _similar_title_papers()
    prominence = {f"p{i}": 3.0 for i in range(6)} | {"p2": 9.0}
    _, meta = SimilarTitleBuilder(papers, prominence, topic_only=True).distractors(papers.iloc[0].to_dict())
    # the three most similar titles, whatever the year: p2 (2020) shares "bilayer", p1 has "bilayers"
    assert meta["donors"] == ["p2", "p1", "p4"]
    assert set(meta["donor_rules"]) == {"most_similar_title"}


def test_team_prominence_uses_the_best_known_member():
    assert round(team_prominence([{"hIndex": 3}, {"hIndex": None}, {"hIndex": 19}]), 6) == round(float(__import__("math").log1p(19)), 6)
    assert team_prominence([{"hIndex": None}]) != team_prominence([{"hIndex": None}])  # NaN


def test_collab_swap_changes_exactly_one_non_first_author():
    papers = _papers([{"title": "Paper 0", "year": 2000, "authors": ["Alice Brown", "Carl Dent", "Eve Ford"]},
                      {"title": "Paper 1", "year": 2001, "authors": ["Gil Hart", "Ian Jones", "Kai Lund"]},
                      {"title": "Paper 2", "year": 2002, "authors": ["Max Nye", "Ola Park", "Sam Tate"]}])
    collaborators = {"p0": ["C. Dent", "Gina Hall", "Alice Brown", "Ivan Ince", "Jack Kerr"],
                     "p1": ["Rita Moss"]}
    builder = CollaboratorSwapBuilder(papers, collaborators)
    options, meta = builder.distractors(papers.iloc[0].to_dict())
    true_fmt = ["A. Brown", "C. Dent", "E. Ford"]
    assert [sorted(set(o) - set(true_fmt)) for o in options] == [["G. Hall"], ["I. Ince"], ["J. Kerr"]]
    assert all(o[0] == "A. Brown" for o in options) and all(p in (1, 2) for p in meta["swap_positions"])
    assert meta["swap_rules"] == ["collaborator"] * 3
    _, meta = builder.distractors(papers.iloc[1].to_dict())
    assert meta["swap_rules"] == ["collaborator", "field_name", "field_name"]
    slot = CollaboratorSwapBuilder(papers, collaborators, shared_position=True)
    options, meta = slot.distractors(papers.iloc[0].to_dict())
    assert len(set(meta["swap_positions"])) == 1
    p = meta["swap_positions"][0]
    assert all(o[:p] + o[p + 1:] == true_fmt[:p] + true_fmt[p + 1:] for o in options)
    report = verify_items(build_level_items(papers, [], "collab_swap", builder.distractors), papers)
    assert report["distractor_groups_overlapping_truth"] == 0 and report["items_with_duplicate_options"] == 0


def test_catch_items_hide_the_truth_and_rotate_none():
    papers = _toy()
    items = build_catch_items(papers, build_real_team_items(papers, control_ids=[]))
    p0 = [i for i in items if i["paper_id"] == "p0"]
    assert len(p0) == 8
    for it in p0:
        assert it["options"][ord(it["none_label"]) - 65] == NONE_OPTION
        has_truth = ["A. Brown", "C. Dent"] in it["options"]
        assert has_truth == (it["condition"] == "with_true")
        if it["condition"] == "catch":
            assert it["correct_label"] == it["none_label"]
        else:
            assert it["options"][ord(it["correct_label"]) - 65] == ["A. Brown", "C. Dent"]
        assert NONE_OPTION in it["user_text"] and "Paper 0" in it["user_text"]
    assert sorted(i["none_label"] for i in p0 if i["condition"] == "catch") == ["A", "B", "C", "D"]
    assert verify_items(items, papers)["items_with_duplicate_options"] == 0


def test_yesno_and_list_records_cover_every_option():
    papers = _toy()
    items = build_real_team_items(papers, control_ids=["p1"])
    yesno = build_yesno_records({"real_team": items})
    assert len(yesno) == 6 * 4 and sum(r["is_true"] for r in yesno) == 6
    assert all(r["user_text"].endswith("Answer Yes or No.") for r in yesno)
    lists = build_list_score_records({"real_team": items})
    assert len(lists) == 6 * 4 * 2
    assert {r["title"] for r in lists if r["condition"] == "hidden"} == {"[title not shown]"}


class FakeTokenizer:
    """Characters as tokens; the chat template is a plain string."""

    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        return "".join(f"<{m['role']}>{m['content']}" for m in messages) + "<assistant>"

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]


def test_score_token_ids_mark_the_continuation():
    tok = FakeTokenizer()
    records = [{"kind": "list", "title": "T", "year": 2014, "names": ["P. Shi", "L. Li"]},
               {"kind": "author", "title": None, "year": 2014, "name": "Peng Shi"},
               {"kind": "title", "title": "Reduced-order filters", "year": 2014, "author": "Shi, P."}]
    for rec, (ids, start) in zip(records, score_token_ids(tok, records)):
        prefix, continuation = score_texts(rec, tok)
        assert "".join(map(chr, ids[start:])) == continuation and "".join(map(chr, ids[:start])) == prefix
    assert score_texts(records[0], tok)[1] == "P. Shi; L. Li<|im_end|>"
    assert "a paper which was published in 2014" in score_texts(records[1], tok)[0]
    assert score_texts(records[2], tok) == ("References\n\nShi, P. et al. (2014).", " Reduced-order filters.")
    assert score_texts(records[2] | {"author": None}, tok)[0] == "References\n\nAnonymous (2014)."


def test_run_score_and_run_yes_no_with_a_fake_engine(tmp_path, monkeypatch):
    import sys
    import types

    from gts.generate import read_chunks, run_score, run_yes_no

    fake_vllm = types.ModuleType("vllm")
    fake_vllm.SamplingParams = lambda **kw: kw
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)

    class LP:
        def __init__(self, logprob, token=None):
            self.logprob, self.decoded_token = logprob, token

    class Engine:
        def get_tokenizer(self):
            return FakeTokenizer()

        def generate(self, prompts, params, use_tqdm=False):
            outs = []
            for p in prompts:
                ids = p["prompt_token_ids"] if isinstance(p, dict) else [0]
                first = {1: LP(-0.1, "Yes"), 2: LP(-2.3, " no")}
                outs.append(types.SimpleNamespace(
                    prompt_logprobs=[None] + [{t: LP(-0.5)} for t in ids[1:]],
                    outputs=[types.SimpleNamespace(text="Yes", logprobs=[first])]))
            return outs

    records = [{"record_id": "r1", "kind": "title", "title": "Ab", "year": 2014, "author": None}]
    run_score(Engine(), records, tmp_path / "score", log=lambda m: None)
    row = read_chunks(tmp_path / "score")[0]
    assert row["n_tokens"] == len(" Ab.") and abs(row["logprob"] + 0.5 * len(" Ab.")) < 1e-9
    run_yes_no(Engine(), [{"record_id": "q1", "user_text": "Is X?", "system": "S"}], tmp_path / "yn",
               log=lambda m: None)
    assert read_chunks(tmp_path / "yn")[0]["yes_no"] == {"yes": -0.1, "no": -2.3}


def test_apa_author():
    assert apa_author("P. Shi") == "Shi, P."
    assert apa_author("P. van der Berg") == "van der Berg, P."
    assert apa_author("Shi") == "Shi"


def test_candidate_score_records():
    papers = _toy().assign(gt_str=lambda d: d["authors"].map(", ".join))
    samples = {"p0": ["Alice Brown; Wei Zhang"] * 3 + ["A. Brown; Li Wang"]}
    recs = build_candidate_score_records(papers, samples)
    kinds = [r["record_id"].rsplit("|", 2)[0] if r["rank"] is not None else r["record_id"] for r in recs]
    assert len(recs) == 3 * 3 + 1
    by_id = {r["record_id"]: r for r in recs}
    assert by_id["author|title|p0|0"]["name"] == "Alice Brown" and by_id["author|title|p0|0"]["is_true"]
    assert by_id["author|none|p0|0"]["title"] is None
    assert by_id["title|author|p0|1"]["author"] == "Zhang, W." and not by_id["title|author|p0|1"]["is_true"]
    assert by_id["title|none|p0"]["author"] is None and kinds[-1] == "title|none|p0"
