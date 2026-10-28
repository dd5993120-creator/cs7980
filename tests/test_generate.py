"""The Colab runners, exercised end to end with a stand-in for vLLM."""

import math
import sys
import types
from types import SimpleNamespace

import pytest

from gts.generate import _letter_logprobs, paper_seed, read_chunks, run_mc, run_open


@pytest.fixture(autouse=True)
def fake_vllm(monkeypatch):
    module = types.ModuleType("vllm")
    module.SamplingParams = lambda **kw: SimpleNamespace(**kw)
    monkeypatch.setitem(sys.modules, "vllm", module)


class FakeTokenizer:
    def apply_chat_template(self, messages, tokenize, add_generation_prompt, enable_thinking):
        assert enable_thinking is False and add_generation_prompt and not tokenize
        return " | ".join(m["content"] for m in messages)


def lp(token, value):
    return SimpleNamespace(decoded_token=token, logprob=value)


class FakeLLM:
    def __init__(self):
        self.calls = 0

    def get_tokenizer(self):
        return FakeTokenizer()

    def generate(self, prompts, params, use_tqdm=False):
        self.calls += 1
        outs = []
        for i, _ in enumerate(prompts):
            p = params[i] if isinstance(params, list) else params
            if getattr(p, "logprobs", None):
                first = {32: lp("B", -0.1), 33: lp("A", -2.5), 362: lp(" A", -3.0), 7: lp("The", -4.0)}
                outs.append(SimpleNamespace(outputs=[SimpleNamespace(text="B", logprobs=[first])]))
            else:
                outs.append(SimpleNamespace(outputs=[
                    SimpleNamespace(text=f"Name {j}", finish_reason="stop", token_ids=[1, 2, 3], logprobs=None)
                    for j in range(p.n)]))
        return outs


PAPERS = [{"paper_id": f"p{i}", "title": f"Title {i}", "year": 2000 + i} for i in range(5)]


def test_run_open_writes_resumable_chunks(tmp_path):
    llm = FakeLLM()
    run_open(llm, PAPERS, tmp_path / "samples", mode="samples", n=3, temperature=1.0, chunk_size=2, log=lambda *_: None)
    rows = read_chunks(tmp_path / "samples")
    assert [r["paper_id"] for r in rows] == [p["paper_id"] for p in PAPERS]
    assert all(len(r["texts"]) == 3 and r["n_tokens"] == [3, 3, 3] for r in rows)
    assert llm.calls == 3
    run_open(llm, PAPERS, tmp_path / "samples", mode="samples", n=3, temperature=1.0, chunk_size=2, log=lambda *_: None)
    assert llm.calls == 3, "existing chunks must be skipped on a rerun"


def test_run_mc_reads_letter_scores(tmp_path):
    items = [{"item_id": "x|o0", "title": "T", "year": 2001,
              "options": [["A. One"], ["B. Two"], ["C. Three"], ["D. Four"]]}]
    run_mc(FakeLLM(), items, tmp_path / "mc", log=lambda *_: None)
    (row,) = read_chunks(tmp_path / "mc")
    scores = row["letter_logprobs"]
    assert set(scores) == {"A", "B"}
    assert scores["B"] == pytest.approx(-0.1)
    assert scores["A"] == pytest.approx(math.log(math.exp(-2.5) + math.exp(-3.0)))


def test_letter_logprobs_ignores_other_tokens():
    assert _letter_logprobs({1: lp("Answer", -0.5), 2: lp("C", -1.0)}) == {"C": -1.0}


def test_paper_seed_is_stable_and_mode_specific():
    assert paper_seed("abc", "samples") == paper_seed("abc", "samples")
    assert paper_seed("abc", "samples") != paper_seed("abc", "greedy")
    assert 0 <= paper_seed("abc", "samples") < 2**31
