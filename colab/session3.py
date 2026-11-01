# Session 3, sent after colab/remote_steps.py (uses its globals: llm, log, logged, OUT, WORK,
# mc_step, meta_step, pack). The rest of the recognition ladder (L0, L3, L4), catch trials,
# the yes/no and likelihood formats, and likelihood scores for selection; items are built by
# tools/build_session3_items.py. Usage: append one call line, e.g. "session3_part1()", and send
# with colab exec -f, keeping the client connected (remote_steps.py explains why).


@logged
def load3(utilization: float = 0.88):
    """As load(), with some memory left free for the prompt log-probabilities of score_step."""
    global spec, llm
    from gts.generate import load_llm, model_spec
    spec = model_spec() | {"gpu_memory_utilization": utilization}
    log(f"spec {spec}")
    t0 = time.time()
    llm = load_llm(spec)
    log(f"model loaded in {time.time() - t0:.0f}s")


@logged
def yesno_step(name: str = "yesno_lists", chunk_size: int = 10000):
    """Yes/No log-probabilities for one record file (one author list per question)."""
    from gts.generate import run_yes_no
    from gts.mc import read_items
    t0 = time.time()
    records = read_items(f"{WORK}/data/mc_items/{name}.jsonl.gz")
    run_yes_no(llm, records, f"{OUT}/{name}", chunk_size=chunk_size, log=log)
    log(f"{name} done in {time.time() - t0:.0f}s for {len(records)} questions")


@logged
def score_step(name: str, chunk_size: int = 20000):
    """Teacher-forced log-probabilities for one scoring record file."""
    from gts.generate import run_score
    from gts.mc import read_items
    t0 = time.time()
    records = read_items(f"{WORK}/data/mc_items/{name}.jsonl.gz")
    run_score(llm, records, f"{OUT}/{name}", chunk_size=chunk_size, log=log)
    log(f"{name} done in {time.time() - t0:.0f}s for {len(records)} records")


@logged
def session3_part1():
    """The three new ladder levels (about 45 minutes)."""
    for name in ("random_field", "similar_title", "collab_swap"):
        mc_step(name)
    log("session3_part1 finished")


@logged
def session3_part2():
    """Catch trials and yes/no questions (about 45 minutes)."""
    mc_step("catch")
    yesno_step()
    log("session3_part2 finished")


@logged
def session3_part3():
    """Two levels added after part 1 (L3t, and L4 with one shared swap position, since the
    per-distractor positions of collab_swap leaked in its controls), then likelihood scores
    (about 1.5 hours), then wrap up."""
    mc_step("similar_topic")
    mc_step("collab_slot")
    score_step("score_lists")
    score_step("score_cands")
    meta_step({"session": 3, "scoring": "teacher-forced prompt log-probabilities (vLLM prompt_logprobs)",
               "gpu_memory_utilization": spec["gpu_memory_utilization"]})
    pack("outputs_s3.tar.gz")
    log("ALL DONE")


log("session3 loaded")
