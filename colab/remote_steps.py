"""Session-1 steps for driving a Colab VM through the Colab CLI.

Send this file once with `colab exec -s gts -f colab/remote_steps.py`; it only
defines things. Kernel state persists between `colab exec` calls, so each step
is then started with a one-liner, e.g.

    echo "load()" | colab exec -s gts --timeout 1200
    echo "samples_step()" | colab exec -s gts --timeout 7000

Keep the `colab exec` client connected for the whole step (a long --timeout):
Colab deletes a runtime about ten minutes after its last client disconnects,
even mid-execution, unless the account has background execution (Pro+).

Every step appends to /content/outputs/progress.log, which `colab download`
can fetch while the kernel is busy. Outputs are chunked and resumable.
"""

import functools
import json
import os
import shutil
import sys
import threading
import time
import traceback
import zipfile

# bf16 weights (61 GiB) do not fit next to vLLM on the 113 GB disk; a large
# RAM disk holds them instead. Must be set before huggingface_hub is imported.
if "HF_HOME" not in os.environ and shutil.disk_usage("/dev/shm").free > 75 * 2**30:
    os.environ["HF_HOME"] = "/dev/shm/hf"

WORK = "/content/gts_bundle"
OUT = "/content/outputs"

if not os.path.exists(WORK):
    zipfile.ZipFile("/content/gts_bundle.zip").extractall(WORK)
if f"{WORK}/src" not in sys.path:
    sys.path.insert(0, f"{WORK}/src")
os.makedirs(OUT, exist_ok=True)


def log(msg) -> None:
    line = f"{time.strftime('%H:%M:%S')} {msg}"
    print(line, flush=True)
    with open(f"{OUT}/progress.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


_beating = threading.Event()


def _heartbeat(stop: threading.Event, interval: int = 60) -> None:
    """Print a line every minute, so the client's stream never goes quiet."""
    while not stop.wait(interval):
        print(f"{time.strftime('%H:%M:%S')} alive", flush=True)


def logged(fn):
    """Write any exception to the progress log (a disconnected client never
    sees the kernel's own traceback), with a heartbeat while the step runs."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        outer = not _beating.is_set()
        stop = threading.Event()
        if outer:
            _beating.set()
            threading.Thread(target=_heartbeat, args=(stop,), daemon=True).start()
        try:
            return fn(*args, **kwargs)
        except Exception:
            log(f"ERROR in {fn.__name__}:\n{traceback.format_exc()}")
            raise
        finally:
            if outer:
                stop.set()
                _beating.clear()
    return wrapper


@logged
def load():
    """Pick bf16 or FP8 from GPU memory and free disk, then start vLLM."""
    global spec, llm
    from gts.generate import load_llm, model_spec
    spec = model_spec()
    log(f"spec {spec}")
    t0 = time.time()
    llm = load_llm(spec)
    log(f"model loaded in {time.time() - t0:.0f}s")


def _papers():
    from gts.data import load_papers
    return load_papers().sort_values("paper_id").to_dict("records")


@logged
def greedy_step():
    """Default answers for all papers, compared with the paper's own bf16 run."""
    from gts.data import load_shipped_greedy
    from gts.generate import read_chunks, run_open
    t0 = time.time()
    run_open(llm, _papers(), f"{OUT}/greedy", mode="greedy", n=1, temperature=0.0,
             chunk_size=2000, log=log)
    ours = {r["paper_id"]: r["texts"][0].strip() for r in read_chunks(f"{OUT}/greedy")}
    shipped = load_shipped_greedy().set_index("paper_id")["generated_text"].fillna("")
    same = sum(ours[p] == shipped[p].strip() for p in ours)
    log(f"greedy done in {time.time() - t0:.0f}s: {len(ours)} answers, "
        f"identical to the paper run {same / len(ours):.1%}, "
        f"with <think> {sum('<think>' in t for t in ours.values())}, "
        f"empty {sum(not t for t in ours.values())}")


@logged
def samples_step(chunk_size: int = 500):
    """50 sampled author lists per paper at T=1, no truncation."""
    from gts.generate import read_chunks, run_open
    t0 = time.time()
    run_open(llm, _papers(), f"{OUT}/samples", mode="samples", n=50, temperature=1.0,
             chunk_size=chunk_size, log=log)
    first = read_chunks(f"{OUT}/samples")[:200]
    distinct = sum(len(set(r["texts"])) for r in first) / max(len(first), 1)
    log(f"samples done in {time.time() - t0:.0f}s; mean distinct lists per paper "
        f"(first 200 papers): {distinct:.1f} of 50")


@logged
def mc_step(name: str = "real_team", chunk_size: int = 5000):
    """Score one item file and log per-condition accuracy."""
    import pandas as pd
    from gts.generate import read_chunks, run_mc
    from gts.mc import read_items
    t0 = time.time()
    items = read_items(f"{WORK}/data/mc_items/{name}.jsonl.gz")
    run_mc(llm, items, f"{OUT}/mc_{name}", chunk_size=chunk_size, log=log)
    res = pd.DataFrame(read_chunks(f"{OUT}/mc_{name}"))
    df = pd.DataFrame(items).merge(res, on="item_id")
    df["chosen"] = df["letter_logprobs"].map(lambda d: max(d, key=d.get) if d else None)
    if "correct_letters" in df:  # candidate MC: any true author, or E when none is
        df["correct"] = [c in letters for c, letters in zip(df["chosen"], df["correct_letters"])]
    else:
        df["correct"] = df["chosen"] == df["correct_label"]
    per_paper = df.groupby(["condition", "paper_id"])["correct"].sum()
    log(f"mc {name} done in {time.time() - t0:.0f}s; per-question accuracy "
        f"{df.groupby('condition')['correct'].mean().round(3).to_dict()}; "
        f">=3 of 4 orders {(per_paper >= 3).groupby('condition').mean().round(3).to_dict()}; "
        f"no letter in top-20: {int(df['chosen'].isna().sum())}")


@logged
def meta_step(extra: dict | None = None):
    from gts.generate import run_metadata
    meta = run_metadata(spec) | (extra or {})
    with open(f"{OUT}/run_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    log(f"meta {meta}")


@logged
def pack(name: str = "outputs.tar.gz"):
    """One archive of everything under /content/outputs, for `colab download`."""
    import tarfile
    with tarfile.open(f"/content/{name}", "w:gz") as tar:
        tar.add(OUT, arcname="outputs")
    log(f"packed /content/{name} ({os.path.getsize(f'/content/{name}') / 1e6:.1f} MB)")


@logged
def samples_then_real_team():
    """The long stretch of session 1, in one call: samples, then real-team MC."""
    samples_step()
    mc_step("real_team")
    log("samples and real_team MC finished")


@logged
def ptrue_step(chunk_size: int = 10000):
    """P(True) for the top candidates of every paper's pool."""
    import importlib
    import gts.generate
    importlib.reload(gts.generate)  # the kernel may hold a copy from before run_ptrue existed
    from gts.generate import run_ptrue
    from gts.mc import read_items
    t0 = time.time()
    records = read_items(f"{WORK}/data/mc_items/ptrue_candidates.jsonl.gz")
    run_ptrue(llm, records, f"{OUT}/ptrue", chunk_size=chunk_size, log=log)
    log(f"ptrue done in {time.time() - t0:.0f}s for {len(records)} candidates")


@logged
def second_half():
    """After the own-sample items and candidates are uploaded: score them, then wrap up."""
    mc_step("own_sample")
    ptrue_step()
    meta_step()
    pack()
    log("ALL DONE")


# ---------------------------------------------------------------- session 2

COT_SAMPLING = {"top_p": 0.95, "top_k": 20}   # Qwen3's recommended thinking-mode settings, with T=0.6
T07_SAMPLING = {"top_p": 0.8, "top_k": 20}    # its recommended non-thinking settings, with T=0.7


@logged
def robust_step():
    """50 samples at Qwen's recommended non-thinking settings, on the control subset."""
    from gts.data import load_papers
    from gts.generate import run_open
    from gts.mc import control_subset
    papers = load_papers()
    ids = set(control_subset(papers, per_stratum=10))
    sub = papers[papers["paper_id"].isin(ids)].sort_values("paper_id").to_dict("records")
    t0 = time.time()
    run_open(llm, sub, f"{OUT}/samples_t07", mode="samples_t07", n=50, temperature=0.7,
             chunk_size=250, log=log, sampling=T07_SAMPLING)
    log(f"samples_t07 done in {time.time() - t0:.0f}s for {len(sub)} papers")


@logged
def cot_step(which: str = "test", chunk_size: int = 500, max_tokens: int = 3500):
    """Thinking mode, one answer per paper of one split (into cot_<split>/); run the
    test split first, so the reported comparison is complete even if time runs out."""
    from gts.generate import run_open
    from gts.splits import load_split
    split = load_split(f"{WORK}/data/splits/split_v1.csv")
    papers = [p for p in _papers() if split[p["paper_id"]] == which]
    t0 = time.time()
    run_open(llm, papers, f"{OUT}/cot_{which}", mode="cot", n=1, temperature=0.6, chunk_size=chunk_size,
             log=log, thinking=True, sampling=COT_SAMPLING, max_tokens=max_tokens)
    log(f"cot {which} done in {time.time() - t0:.0f}s for {len(papers)} papers")


@logged
def session2_mc():
    """First connection of session 2 (about 45 minutes): MC levels and the temperature check."""
    mc_step("real_team_year")
    mc_step("cand_mc")
    robust_step()
    log("session2_mc finished")


@logged
def session2_wrap():
    meta_step({"session": 2, "cot": {"temperature": 0.6, **COT_SAMPLING, "max_tokens": 3500},
               "samples_t07": {"temperature": 0.7, **T07_SAMPLING, "n": 50}})
    pack("outputs_s2.tar.gz")
    log("ALL DONE")


log("remote_steps loaded")
