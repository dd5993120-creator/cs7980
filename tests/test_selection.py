import sys
import types
from types import SimpleNamespace

import pytest

from gts.generate import _yes_no_logprobs, read_chunks, run_ptrue
from gts.selection import candidate_pool, label_candidates


def lp(token, value):
    return SimpleNamespace(decoded_token=token, logprob=value)


def test_candidate_pool_counts_lists_not_mentions():
    texts = ["Wei Zhang; W. Zhang; Li Wang", "Wei Zhang", "Li Wang; Jun Li", "", "Jun Li"]
    pool = candidate_pool(texts, n=5)
    assert [(c["name"], c["count"]) for c in pool] == [("W. Zhang", 2), ("L. Wang", 2), ("J. Li", 2)]
    assert [c["rank"] for c in pool] == [0, 1, 2]


def test_label_candidates_marks_true_authors():
    pool = [{"name": "W. Zhang", "count": 3, "rank": 0}, {"name": "J. Li", "count": 1, "rank": 1}]
    out = label_candidates(pool, ["Wei Zhang", "Ann Bell"])
    assert [c["is_true"] for c in out] == [True, False]
    assert out[0]["gt_index"] == 0 and out[1]["gt_index"] is None


def test_yes_no_merges_case_and_space():
    scores = _yes_no_logprobs({1: lp("Yes", -0.2), 2: lp(" yes", -3.0), 3: lp("No", -1.7), 4: lp("Maybe", -4)})
    assert set(scores) == {"yes", "no"}
    assert scores["no"] == pytest.approx(-1.7)


def test_run_ptrue_writes_chunks(tmp_path, monkeypatch):
    module = types.ModuleType("vllm")
    module.SamplingParams = lambda **kw: SimpleNamespace(**kw)
    monkeypatch.setitem(sys.modules, "vllm", module)

    class Tok:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
            return messages[-1]["content"]

    class LLM:
        def get_tokenizer(self):
            return Tok()

        def generate(self, prompts, params, use_tqdm=False):
            assert "Is W. Zhang one of the authors" in prompts[0]
            return [SimpleNamespace(outputs=[SimpleNamespace(text="Yes", logprobs=[{1: lp("Yes", -0.1), 2: lp("No", -2.4)}])])
                    for _ in prompts]

    recs = [{"record_id": f"p|{i}", "name": "W. Zhang", "title": "T", "year": 2001} for i in range(3)]
    run_ptrue(LLM(), recs, tmp_path / "ptrue", chunk_size=2, log=lambda *_: None)
    rows = read_chunks(tmp_path / "ptrue")
    assert len(rows) == 3 and rows[0]["yes_no"]["yes"] == pytest.approx(-0.1)
