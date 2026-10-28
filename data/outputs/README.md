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

## fp8_40gb/

An earlier greedy run with the FP8 checkpoint on a 40 GB A100 (that session was lost before sampling).
Aggregate scores match the paper's bf16 run; kept only for that comparison.
