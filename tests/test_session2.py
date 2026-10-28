import pandas as pd

from gts.mc import RealTeamBuilder, build_cand_mc_items, build_real_team_items, verify_items
from gts.prompts import split_thinking
from gts.splits import make_split


SURNAMES = ["Abbott", "Barker", "Carver", "Dalton", "Easton", "Fisher", "Garner", "Hughes", "Irving", "Jarvis",
            "Keller", "Lawson", "Mercer", "Norris", "Osborn", "Palmer", "Quincy", "Ramsey", "Sutton", "Turner",
            "Upton", "Vaughn", "Walker", "Xavier", "Yates", "Zeller", "Archer", "Bishop", "Conway", "Dorsey",
            "Ellison", "Foster", "Graham", "Harper", "Ingram", "Jensen", "Kirby", "Lambert", "Monroe", "Nolan",
            "Oakley", "Porter", "Quinlan", "Reeves", "Sawyer", "Thorne", "Ulrich", "Vance", "Wilder", "Young",
            "Ziegler", "Ashby", "Bowen", "Crane", "Doyle", "Emery", "Flynn", "Grant", "Hayes", "Irwin",
            "Joyce", "Knox", "Lyons", "Moss", "Neal", "Owens", "Pratt", "Rhodes", "Stone", "Tate",
            "Underwood", "Voss", "Wells", "York", "Zane", "Avery", "Blake", "Cole", "Drake", "Ellis"]


def _papers(n=12):
    rows = []
    for i in range(n):
        a, b = f"Ann {SURNAMES[2 * i]}", f"Bob {SURNAMES[2 * i + 1]}"
        rows.append({"paper_id": f"p{i:02d}", "title": f"Paper {i}", "year": 1990 + 3 * i, "field": "physics",
                     "citations": i, "citation_bin": "0-2" if i % 2 else "3-4",
                     "authors": [a, b], "gt_str": f"{a}, {b}"})
    return pd.DataFrame(rows)


def test_year_window_prefers_near_donors():
    papers = _papers()
    builder = RealTeamBuilder(papers, year_window=3)
    paper = papers.iloc[5].to_dict()  # 2005; donors within 3 years: 2002 and 2008 only
    _, donors, rules = builder.distractors(paper)
    years = {papers.set_index("paper_id").at[d, "year"] for d, r in zip(donors, rules) if r == "same_length_year"}
    assert years <= {2002, 2008} and rules.count("same_length_year") == 2 and rules[2] == "same_length"


def test_year_level_items_verify():
    papers = _papers()
    items = build_real_team_items(papers, control_ids=["p01"], level="real_team_year", year_window=2)
    report = verify_items(items, papers)
    assert {i["level"] for i in items} == {"real_team_year"}
    assert report["distractor_groups_overlapping_truth"] == 0 and report["groups_without_4_orders"] == 0


def test_cand_mc_correct_letters():
    papers = _papers(3)
    samples = {"p00": ["Ann Abbott; Zed Ray", "Zed Ray; Quo Vad", "Mia Fox", "Ira Gold; Zed Ray"],
               "p01": ["Tom Hill", "Kai Lin; Tom Hill", "Una Bey", "Vic Orr"],
               "p02": ["Only One"]}
    items = build_cand_mc_items(papers, samples)
    by_paper = {}
    for it in items:
        by_paper.setdefault(it["paper_id"], []).append(it)
    assert set(by_paper) == {"p00", "p01"}  # p02 has fewer than four candidates
    for it in by_paper["p00"]:
        letter = "ABCD"[it["options"].index("A. Abbott")]
        assert it["correct_letters"] == [letter] and "E: None of these" in it["user_text"]
    assert all(it["correct_letters"] == ["E"] for it in by_paper["p01"])


def test_split_is_stratified_and_stable():
    papers = _papers(40)
    a, b = make_split(papers), make_split(papers)
    assert a.equals(b)
    merged = a.merge(papers[["paper_id", "citation_bin"]], on="paper_id")
    for _, g in merged.groupby("citation_bin"):
        assert (g.split == "tune").sum() == round(len(g) * 0.3)


def test_split_thinking():
    assert split_thinking("<think>\nhmm\n</think>\n\nA. Lee; B. Kim") == ("hmm", "A. Lee; B. Kim", True)
    assert split_thinking("<think>\nstill thinking")[1:] == ("", False)
