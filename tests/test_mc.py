import pandas as pd

from gts.mc import (_keys, build_own_sample_items, build_real_team_items, cyclic_orders,
                    own_sample_distractors, verify_items, wrong_lists)

TRUTH = ["Alice Brown", "Carl Dent"]


def test_cyclic_orders_put_truth_once_at_each_position():
    orders = list(cyclic_orders(["T"], [["a"], ["b"], ["c"]]))
    assert [opts.index(["T"]) for _, opts in orders] == [0, 1, 2, 3]
    assert all(sorted(map(tuple, opts)) == sorted(map(tuple, orders[0][1])) for _, opts in orders)


def test_wrong_lists_drop_any_list_naming_a_true_author():
    texts = ["Wei Zhang; Li Wang", "Alice Brown; Wei Zhang", "Wei Zhang; Li Wang",
             "A. Brown", "Jun Li; Wei Zhang; Jun Li", "B. Dent"]
    lists, pool = wrong_lists(TRUTH, texts)
    # "Alice Brown", "A. Brown" name a true author; "B. Dent" is an L5 surname match, also excluded
    assert lists == [(["W. Zhang", "L. Wang"], 2), (["J. Li", "W. Zhang"], 1)]
    assert pool[0] == "W. Zhang"


def test_distractors_have_true_length_and_share_no_names():
    texts = ["Wei Zhang; Li Wang"] * 5 + ["Wei Zhang; Jun Li"] * 3 + ["Mei Chen; Hao Liu; Yu Sun"] * 2 \
        + ["Tao Xu"] + ["Ping Ma; Lei Gao"]
    options, rules, stats = own_sample_distractors(TRUTH, texts)
    assert len(options) == 3 and all(len(o) == 2 for o in options)
    keys = [_keys(o) for o in options]
    assert not (keys[0] & keys[1] or keys[0] & keys[2] or keys[1] & keys[2])
    assert options[0] == ["W. Zhang", "L. Wang"] and rules[0] == "sampled"
    assert options[1] == ["M. Chen", "H. Liu"] and rules[1] == "sampled_cut"
    assert stats["n_wrong_lists"] == 5


def _papers():
    rows = []
    for i, (authors, field) in enumerate([
        (["Alice Brown", "Carl Dent"], "physics"), (["Eva Fox", "Gil Hart"], "physics"),
        (["Ian Jones", "Kai Lund"], "physics"), (["Max Nye", "Ola Park"], "physics"),
        (["Quinn Ross", "Sam Tate"], "physics"), (["Uma Vance", "Will Xu"], "physics")]):
        rows.append({"paper_id": f"p{i}", "title": f"Paper {i}", "year": 2000 + i, "field": field,
                     "citations": i, "citation_bin": "0-2", "authors": authors})
    return pd.DataFrame(rows)


def test_own_sample_items_fall_back_to_real_teams_and_verify():
    papers = _papers()
    samples = {"p0": ["Wei Zhang; Li Wang"] * 50, "p1": ["Rui Song; Bo Hu", "Ann Lee; Tom Kim", "Jo Ray; Al Poe"] * 10}
    items = build_own_sample_items(papers, samples, control_ids=["p1"])
    report = verify_items(items, papers)
    assert report["distractor_groups_overlapping_truth"] == 0
    assert report["items_with_duplicate_options"] == 0
    assert report["groups_without_4_orders"] == 0
    p0 = [i for i in items if i["paper_id"] == "p0" and i["order"] == 0][0]
    assert p0["distractor_rules"][0] == "sampled" and "real_team_fallback" in p0["distractor_rules"]
    assert {i["condition"] for i in items if i["paper_id"] == "p1"} == {"standard", "title_hidden", "title_swapped"}


def test_real_team_items_on_toy_sample():
    papers = _papers()
    items = build_real_team_items(papers, control_ids=["p2"])
    report = verify_items(items, papers)
    assert report["n_items"] == 6 * 4 + 2 * 4
    assert report["distractor_groups_overlapping_truth"] == 0
