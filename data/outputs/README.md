# Outputs

One folder per Colab run. Every file is gzipped JSONL written in chunks (`chunk_000.jsonl.gz`, ...).
Each run folder has `run_meta.json` (model, precision, GPU, vLLM, decoding) and `progress.log`.
The consolidated write-up is `RESULTS.md` at the project root.

## g4_bf16/ (session 1, 2026-10-26)

Qwen3-32B in bf16 on an RTX PRO 6000 Blackwell (Colab "G4", 96 GB), vLLM 0.32.0.

| folder | one line per | fields |
|---|---|---|
| `greedy/` | paper | `paper_id`, `texts` (1 answer), `finish`, `n_tokens` |
| `samples/` | paper | `paper_id`, `texts` (50 answers at T=1, top-p 1, no top-k, max 96 tokens), `finish`, `n_tokens` |
| `mc_real_team/` | MC item | `item_id`, `text` (first token), `letter_logprobs` {A..D: log-prob} |
| `mc_own_sample/` | MC item | same as above |
| `ptrue/` | candidate | `record_id` (`paper_id|rank`), `text`, `yes_no` {yes, no: log-prob} |

`analysis/` holds every table: `summary.md`, `selection_summary.md`, `session2_summary.md`, `paired_tests.md`,
`deep_summary.md`, `paired_deep.md`, CSVs, and `pools.pkl` (a cache of the sample pools, safe to delete).

## s2/ (session 2, 2026-10-26)

Same model and machine type; context 4,096.

| folder | one line per | what |
|---|---|---|
| `mc_real_team_year/` | MC item | real-team distractors within +-2 years |
| `mc_cand_mc/` | MC item | top-4 sampled names + "E: none of these" (letters A..E) |
| `samples_t07/` | paper | 50 samples at T=0.7, top-p 0.8, top-k 20 (control subset) |
| `cot_test/`, `cot_tune/` | paper | thinking mode, 1 answer at T=0.6, top-p 0.95, top-k 20, max 3,500 tokens |
| `samples200/` | paper | 200 samples at T=1 (control subset) |
| `cot5/` | paper | 5 thinking-mode answers (control subset) |

Thinking-mode texts contain the reasoning, then `</think>`, then the answer (`gts.prompts.split_thinking`).
Items are in `data/mc_items/`; the split is `data/splits/split_v1.csv`; the control subset is
`gts.mc.control_subset(papers, per_stratum=10)`.

## s3/ (session 3, 2026-10-28)

Same model and machine type, `gpu_memory_utilization` 0.88. Items and records are built by
`tools/build_session3_items.py` from `data/external/` (Semantic Scholar authors and collaborators,
`tools/fetch_s2.py`).

| folder | one line per | what |
|---|---|---|
| `mc_random_field/` | MC item | L0: random names from the paper's field |
| `mc_similar_title/` | MC item | L3: real teams within +-2 years, matched prominence, most similar titles |
| `mc_similar_topic/` | MC item | L3t: real teams of the most similar titles, any year |
| `mc_collab_slot/` | MC item | L4: one co-author swapped for a real collaborator, one shared slot |
| `mc_collab_swap/` | MC item | L4r: the same with a slot per option (the release's design; leaks) |
| `mc_catch/` | MC item | L1 distractors with "None of the above"; `with_true` and `catch` items |
| `yesno_lists/` | question | `record_id` (`level|paper_id|list`, list 0 = true), `yes_no` {yes, no: log-prob} |
| `score_lists/` | record | `record_id` (`list|level|condition|paper_id|list`), `logprob`, `n_tokens`, `first_logprob` |
| `score_cands/` | record | `record_id` (`author|title|...`, `author|none|...`, `title|author|...`, `title|none|...`), same fields |

Scores are teacher-forced log-probabilities of the continuation only (`gts.prompts.score_texts`).

## fp8_40gb/

An earlier greedy run with the FP8 checkpoint on a 40 GB A100 (that session was lost before sampling).
Aggregate scores match the paper's bf16 run; kept only for that comparison.
