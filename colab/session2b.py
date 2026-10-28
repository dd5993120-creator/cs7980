# Session 2b, sent after colab/remote_steps.py (uses its globals: llm, log, logged, OUT, WORK).
# Deeper runs on the stratified 1,023-paper control subset, plus the tune-split CoT.
# Usage: append one call line, e.g. "session2b_part1()", and send with colab exec -f.


def _control_papers():
    from gts.data import load_papers
    from gts.mc import control_subset
    papers = load_papers()
    ids = set(control_subset(papers, per_stratum=10))
    return papers[papers["paper_id"].isin(ids)].sort_values("paper_id").to_dict("records")


@logged
def deep_samples_step(n: int = 200):
    """n samples per paper at T=1 (no truncation), to see where coverage saturates."""
    from gts.generate import run_open
    sub = _control_papers()
    t0 = time.time()
    run_open(llm, sub, f"{OUT}/samples{n}", mode=f"samples{n}", n=n, temperature=1.0,
             chunk_size=100, log=log)
    log(f"samples{n} done in {time.time() - t0:.0f}s for {len(sub)} papers")


@logged
def cot_sc_step(n: int = 5):
    """n thinking-mode answers per paper, for self-consistency over CoT."""
    from gts.generate import run_open
    sub = _control_papers()
    t0 = time.time()
    run_open(llm, sub, f"{OUT}/cot{n}", mode=f"cot{n}", n=n, temperature=0.6, chunk_size=100,
             log=log, thinking=True, sampling=COT_SAMPLING, max_tokens=3500)
    log(f"cot{n} done in {time.time() - t0:.0f}s for {len(sub)} papers")


@logged
def session2b_part1():
    cot_step("tune")
    deep_samples_step(200)
    log("session2b_part1 finished")


@logged
def session2b_part2():
    cot_sc_step(5)
    meta_step({"session": 2, "cot": {"temperature": 0.6, **COT_SAMPLING, "max_tokens": 3500},
               "samples_t07": {"temperature": 0.7, **T07_SAMPLING, "n": 50},
               "samples200": {"temperature": 1.0, "top_p": 1.0, "n": 200, "papers": "control subset"},
               "cot5": {"n": 5, "temperature": 0.6, **COT_SAMPLING, "papers": "control subset"}})
    pack("outputs_s2.tar.gz")
    log("ALL DONE")


log("session2b loaded")
