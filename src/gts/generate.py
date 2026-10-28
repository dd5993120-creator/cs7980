"""vLLM runners for the Colab sessions.

vllm is imported lazily, so the rest of the package works on a laptop. Every
run writes gzipped JSONL chunks and skips chunks that already exist, so an
interrupted session resumes where it stopped.

    open question   greedy (n=1, T=0) and sampling (n=50, T=1, no top-p/top-k
                    truncation), max 96 new tokens as in the paper's greedy run
    multiple choice one token at T=0 with the top-20 log-probabilities, from
                    which the four letter scores are read
"""

from __future__ import annotations

import gzip
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from gts.prompts import OPTION_LABELS, chat_prompt, mc_user_text, open_prompt

MODEL_BF16 = "Qwen/Qwen3-32B"
MODEL_FP8 = "Qwen/Qwen3-32B-FP8"
MAX_NEW_TOKENS = 96
MC_TOP_LOGPROBS = 20


def gpu_memory_gib() -> float:
    """Total memory of GPU 0, read from nvidia-smi.

    Not from torch: initialising CUDA in the notebook process before vLLM
    starts forces the spawn start method, which breaks in notebooks.
    """
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    ).stdout.split("\n")[0]
    return float(out) / 1024


def hf_home() -> Path:
    return Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))


def free_disk_gib() -> float:
    """Free space where the Hugging Face cache lives (HF_HOME may point at /dev/shm)."""
    path = hf_home()
    path.mkdir(parents=True, exist_ok=True)
    return shutil.disk_usage(path).free / 2**30


def is_cached(model_id: str) -> bool:
    """True if the model's weights are already in the Hugging Face cache."""
    repo = hf_home() / "hub" / f"models--{model_id.replace('/', '--')}"
    return repo.exists() and any(repo.rglob("*.safetensors"))


def model_spec() -> dict:
    """bf16 Qwen3-32B when a card of 80 GB or more and room for its 61 GiB of weights exist;
    the official FP8 checkpoint (33 GB) otherwise.

    The paper's run is bf16; FP8 (weight-only on an A100) is a fallback and is
    recorded in the run metadata.
    """
    gib, disk = gpu_memory_gib(), free_disk_gib()
    common = {"gpu_gib": round(gib, 1), "free_disk_gib": round(disk, 1)}
    if gib >= 70 and (is_cached(MODEL_BF16) or disk >= 70):
        # 4096 leaves room for thinking-mode answers; paged KV costs nothing until used
        return {"model": MODEL_BF16, "dtype": "bfloat16", "max_num_seqs": 1024 if gib >= 90 else 512,
                "gpu_memory_utilization": 0.92, "precision": "bf16",
                "max_model_len": 4096 if gib >= 90 else 2048, **common}
    if not is_cached(MODEL_FP8) and disk < 40:
        raise RuntimeError(f"only {disk:.0f} GiB free disk; the FP8 checkpoint needs about 35")
    # On a 40 GB card the FP8 weights leave ~9k tokens of KV cache at 0.94;
    # a smaller prefill batch and 0.96 free room for more concurrent samples.
    return {"model": MODEL_FP8, "dtype": "auto", "max_num_seqs": 512 if gib >= 70 else 256,
            "gpu_memory_utilization": 0.92 if gib >= 70 else 0.96,
            "max_num_batched_tokens": 8192 if gib >= 70 else 4096,
            "precision": "fp8 weights (w8a16)", **common}


def load_llm(spec: dict):
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")  # keep the engine in-process
    from vllm import LLM
    return LLM(
        model=spec["model"], dtype=spec["dtype"], max_model_len=spec.get("max_model_len", 2048),
        gpu_memory_utilization=spec["gpu_memory_utilization"], max_num_seqs=spec["max_num_seqs"],
        max_num_batched_tokens=spec.get("max_num_batched_tokens", 8192),
        enable_prefix_caching=True, seed=0,
    )


def paper_seed(paper_id: str, tag: str) -> int:
    return int.from_bytes(hashlib.sha256(f"{tag}|{paper_id}".encode()).digest()[:4], "big") & 0x7FFFFFFF


def _chunks(rows: list, size: int):
    for i in range(0, len(rows), size):
        yield i // size, rows[i:i + size]


def _write_chunk(records: list[dict], dest: Path) -> None:
    """Write locally, then copy, so a dropped Drive mount never leaves half a file."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    local = Path(tempfile.gettempdir()) / f"{dest.parent.name}_{dest.name}"
    with gzip.open(local, "wt", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    tmp = dest.with_suffix(dest.suffix + ".part")
    shutil.copyfile(local, tmp)
    tmp.replace(dest)
    local.unlink()


def run_open(llm, papers: list[dict], out_dir: Path, *, mode: str, n: int,
             temperature: float, chunk_size: int, log=print, thinking: bool = False,
             sampling: dict | None = None, max_tokens: int = MAX_NEW_TOKENS) -> None:
    """Open-question answers for every paper; one record per paper per line.

    ``sampling`` adds SamplingParams fields such as top_p, top_k or min_p
    (top_p defaults to 1.0; top_k and min_p stay off unless given).
    """
    from vllm import SamplingParams

    tokenizer = llm.get_tokenizer()
    extra = {"top_p": 1.0, **(sampling or {})}
    out_dir = Path(out_dir)
    for idx, chunk in _chunks(papers, chunk_size):
        dest = out_dir / f"chunk_{idx:03d}.jsonl.gz"
        if dest.exists():
            continue
        t0 = time.time()
        prompts = [open_prompt(tokenizer, p["title"], p["year"], thinking=thinking) for p in chunk]
        params = [SamplingParams(n=n, temperature=temperature, max_tokens=max_tokens,
                                 seed=None if temperature == 0 else paper_seed(p["paper_id"], mode), **extra)
                  for p in chunk]
        outputs = llm.generate(prompts, params, use_tqdm=False)
        records = [{
            "paper_id": p["paper_id"],
            "texts": [o.text for o in out.outputs],
            "finish": [o.finish_reason for o in out.outputs],
            "n_tokens": [len(o.token_ids) for o in out.outputs],
        } for p, out in zip(chunk, outputs)]
        _write_chunk(records, dest)
        n_tok = sum(sum(r["n_tokens"]) for r in records)
        log(f"[{mode}] chunk {idx:03d}: {len(chunk)} papers, {n_tok:,} tokens, "
            f"{time.time() - t0:.0f}s ({n_tok / max(time.time() - t0, 1e-9):,.0f} tok/s)")


def _token_logprobs(first_token_logprobs: dict, wanted, normalise=str.strip) -> dict:
    """Log-probability of each wanted token among the top ones, merging variants
    such as "A" and " A" (log-sum-exp)."""
    scores: dict[str, float] = {}
    for lp in first_token_logprobs.values():
        token = normalise(lp.decoded_token or "")
        if token in wanted:
            prev = scores.get(token)
            scores[token] = lp.logprob if prev is None else float(
                max(prev, lp.logprob) + math.log1p(math.exp(-abs(prev - lp.logprob))))
    return scores


def _letter_logprobs(first_token_logprobs: dict) -> dict:
    return _token_logprobs(first_token_logprobs, OPTION_LABELS)


def _yes_no_logprobs(first_token_logprobs: dict) -> dict:
    return _token_logprobs(first_token_logprobs, ("yes", "no"), lambda t: t.strip().lower())


def run_mc(llm, items: list[dict], out_dir: Path, *, chunk_size: int = 5000, log=print) -> None:
    """Score multiple-choice items from the first answer token's log-probabilities."""
    from vllm import SamplingParams

    tokenizer = llm.get_tokenizer()
    params = SamplingParams(n=1, temperature=0.0, max_tokens=1, logprobs=MC_TOP_LOGPROBS)
    out_dir = Path(out_dir)
    for idx, chunk in _chunks(items, chunk_size):
        dest = out_dir / f"chunk_{idx:03d}.jsonl.gz"
        if dest.exists():
            continue
        t0 = time.time()
        # Items may carry their own question ("user_text") and letters ("labels"),
        # as the candidate multiple choice does; otherwise the Section 6 template.
        prompts = [chat_prompt(tokenizer, it.get("user_text")
                               or mc_user_text(it["title"], it["year"], it["options"]))
                   for it in chunk]
        outputs = llm.generate(prompts, params, use_tqdm=False)
        records = []
        for it, out in zip(chunk, outputs):
            first = out.outputs[0]
            wanted = tuple(it.get("labels", OPTION_LABELS))
            scores = _token_logprobs(first.logprobs[0], wanted) if first.logprobs else {}
            records.append({"item_id": it["item_id"], "text": first.text, "letter_logprobs": scores})
        _write_chunk(records, dest)
        log(f"[mc] chunk {idx:03d}: {len(chunk)} items, {time.time() - t0:.0f}s")


def run_ptrue(llm, records: list[dict], out_dir: Path, *, chunk_size: int = 10000, log=print) -> None:
    """P(True) for single candidate authors: the Yes/No log-probabilities of the first token."""
    from vllm import SamplingParams

    from gts.selection import PTRUE_SYSTEM, ptrue_user_text

    tokenizer = llm.get_tokenizer()
    params = SamplingParams(n=1, temperature=0.0, max_tokens=1, logprobs=MC_TOP_LOGPROBS)
    out_dir = Path(out_dir)
    for idx, chunk in _chunks(records, chunk_size):
        dest = out_dir / f"chunk_{idx:03d}.jsonl.gz"
        if dest.exists():
            continue
        t0 = time.time()
        prompts = [chat_prompt(tokenizer, ptrue_user_text(r["name"], r["title"], r["year"]), PTRUE_SYSTEM)
                   for r in chunk]
        outputs = llm.generate(prompts, params, use_tqdm=False)
        rows = []
        for r, out in zip(chunk, outputs):
            first = out.outputs[0]
            rows.append({"record_id": r["record_id"], "text": first.text,
                         "yes_no": _yes_no_logprobs(first.logprobs[0]) if first.logprobs else {}})
        _write_chunk(rows, dest)
        log(f"[ptrue] chunk {idx:03d}: {len(chunk)} candidates, {time.time() - t0:.0f}s")


def read_chunks(out_dir: Path) -> list[dict]:
    rows = []
    for path in sorted(Path(out_dir).glob("chunk_*.jsonl.gz")):
        with gzip.open(path, "rt", encoding="utf-8") as f:
            rows.extend(json.loads(line) for line in f if line.strip())
    return rows


def run_metadata(spec: dict) -> dict:
    import torch
    import vllm
    return {
        "model": spec["model"], "precision": spec["precision"], "gpu_gib": spec["gpu_gib"],
        "gpu_name": subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                   capture_output=True, text=True).stdout.strip(),
        "vllm": vllm.__version__, "torch": torch.__version__,
        "max_new_tokens": MAX_NEW_TOKENS,
        "greedy": {"n": 1, "temperature": 0.0},
        "samples": {"n": 50, "temperature": 1.0, "top_p": 1.0, "top_k": "off", "seed": "per paper"},
        "mc": {"temperature": 0.0, "max_tokens": 1, "top_logprobs": MC_TOP_LOGPROBS},
        "written_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
