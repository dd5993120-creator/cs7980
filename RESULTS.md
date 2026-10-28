# Generate-then-select: results so far

Minyan Xu, CS 7980 "Enhancing Memory of LLMs". Runs of 2026-10-26. All numbers come from
`data/outputs/` and the scripts in `tools/`; nothing here is estimated.

## Setup

- **Model.** Qwen3-32B in bf16 (the paper's precision), thinking off unless stated, vLLM 0.32.0 on a Colab
  RTX PRO 6000 (96 GB).
- **Data.** The 9,108 papers of Zhong et al. (EMNLP 2026), 8 fields x 13 citation bins.
- **Open question.** The paper's own Qwen3-32B prompt. "Default answer" = greedy. Sampling = 50 answers per
  paper at T = 1.0, no top-p or top-k truncation, max 96 tokens.
- **Scoring.** The paper's matcher. It reproduces all 9,108 shipped (M, H, U) scores exactly, with
  nameparser 1.1.3 pinned. HR2 = (H+U)/(M+H+U).
- **Recognition labels.** Built with the team's rules on our data:
  - recalled: default HR2 <= 0.5;
  - otherwise known / notsure / unknown: the true list is chosen in >= 3 / 2 / <= 1 of the 4 cyclic option
    orders of a multiple-choice test.

  All option names are rendered "F. Lastname". The answer is read from the first token's log-probabilities.
- **Split.** For anything tuned: 30% tune (2,734 papers) / 70% test (6,374), stratified by field x citation
  bin. Test is scored once.

## 1. The default answers reproduce the paper

| | ours | paper's run |
|---|---|---|
| true authors named (all 9,108 papers) | 1,211 | 1,202 |
| papers with HR2 <= 0.5 | 222 | 222 |
| mean HR2 | 0.9691 | 0.9691 |

52.4% of answers are identical strings. The rest differ because greedy decoding of invented names is
sensitive to small numerical differences between HF transformers and vLLM.

## 2. Recognition tests leak, and harder distractors cut "known" sharply

Share of papers whose true list is chosen in >= 3 of 4 orders. Independent guessing would give about 5%.

| distractors | standard | title hidden | title swapped |
|---|---|---|---|
| real teams, same field and author count | 0.703 | 0.238 | 0.157 |
| same, published within +-2 years | 0.668 | 0.176 | 0.158 |
| the model's own most frequent wrong lists | 0.412 | 0.252 | 0.155 |

The controls use 1,023 papers (up to 10 per field x bin). Three findings:

- **The false-positive floor is about 16%, not 5%.** The title-swapped control stays at about 0.16 because the
  model prefers some lists consistently across orders. Labels should be read against this control.
- **Most of the title-hidden excess is a year cue.** Matching distractor years (+-2) lowers it from 0.238 to
  0.176, close to the floor, while the standard condition barely moves (0.703 to 0.668).
- **The model's own wrong answers are much harder distractors.** Of the 6,179 papers that are known under real
  teams, 3,344 stay known, 826 become notsure and 2,009 become unknown. Only 82 move the other way.

## 3. Sampling reaches far more true authors than the default answer

"Reachable" means a true author the default answer misses but at least one of the 50 samples names. "Net" means
after subtracting a chance baseline: the same count for the authors of the 3 most similar titles in the field,
measured against this paper's samples. A random same-field, same-bin baseline is smaller still.

| papers | named by default | reachable | net of topic baseline [95% CI] |
|---|---|---|---|
| all 9,108 | 13.3 | 58.5 | 48.4 [46.2, 50.7] |
| known (6,179) | 11.4 | 75.2 | 64.5 [61.8, 67.6] |
| unknown (1,783) | 1.3 | 21.8 | 12.9 [10.1, 15.9] |

All counts are true authors per 100 papers.

- **Reach rises steadily with citations.** Net of the topic baseline it is 11.5 [7.5, 15.4] per 100 papers at
  0-2 citations and 139 at 4097+.
- **Per true author, coverage@50 rises from 0.069 to 0.635**, while the topic baseline stays at 0.02-0.03.
- **Known papers have five times the net reach of unknown ones** (64.5 vs 12.9). So the recognition label does
  carry signal about what sampling can recover.
- **Sampling reaches later authors as well as first authors.** 19.2% of both appear in at least one sample.
  The default answer names first authors more often (4.7% vs 3.2%).
- **The finding does not depend on the sampling settings.** On the 1,023-paper subset, with Qwen's recommended
  settings (T = 0.7, top-p 0.8, top-k 20):

  | settings | coverage@10 | coverage@50 | topic baseline @50 |
  |---|---|---|---|
  | T = 1.0, untruncated | 0.131 | 0.248 | 0.019 |
  | T = 0.7, top-p 0.8, top-k 20 | 0.136 | 0.213 | 0.013 |

  Truncation helps slightly at small k but saturates earlier.

## 4. Selection: frequency is the signal; self-judgement adds little

**P(True), one candidate at a time** ("Is X an author of ...? Yes/No"), on each paper's 10 most frequent sampled
names. It separates true from false candidates worse than the sample count does:

| score | pairwise AUC within paper | top-1 is a true author (papers whose top 10 hold one) |
|---|---|---|
| sample count | 0.740 | 0.427 |
| P(True) | 0.606 | 0.313 |

**Joint multiple choice over the top 4 candidates plus "E: none of these".**

- When a true author is among the four, the model picks one 58.9% of the time, against 55.9% for picking the most
  frequent candidate: +3.0 points [+0.2, +5.8] (1,391 papers, paired by paper).
- When none is, it answers E only 30.2% of the time.

**Frequency rules, tuned on tune and scored on test** (6,374 papers):

| method | tokens / paper | mean HR2 | HR2 <= 0.5 | true / 100 | false names / 100 |
|---|---|---|---|---|---|
| default answer | 33 | 0.9687 | 2.5% | 13.4 | 516 |
| thinking mode (CoT), 1 answer | 646 | 0.9577 | 3.5% | 15.4 | 331 |
| names in >= 3 of the first 20 samples | 645 | 0.9599 | 3.1% | 18.6 | 155 |
| names in >= 7 of 50 samples | 1,610 | 0.9585 | 3.4% | 17.2 | 101 |
| same, minus 0.5 x field background count | 1,610 | 0.9582 | 3.4% | 17.7 | 104 |

For CoT, 10.9% of answers are explicit refusals, counted as no answer, and 0.6% hit the 3,500-token limit.
The background correction, a PMI-like discount for names the model gives for many papers of the field, adds
almost nothing beyond frequency.

**Paired differences on test** (95% bootstrap over papers):

| comparison | mean HR2 | true / 100 | false / 100 |
|---|---|---|---|
| CoT - default | -0.011 [-0.014, -0.008] | +2.1 [+0.9, +3.2] | -185 [-231, -128] |
| 20 samples + frequency - default | -0.009 [-0.011, -0.007] | +5.2 [+4.1, +6.3] | -361 [-373, -351] |
| 20 samples + frequency - CoT (equal tokens) | +0.002 [-0.001, +0.005] | +3.2 [+1.9, +4.4] | -176 [-233, -131] |

At equal compute, selecting by frequency from 20 ordinary samples ties thinking mode on mean HR2. It names about
20% more true authors and half as many invented names.

**Compute curve** (first k samples, threshold tuned per k):

| k | 1 | 3 | 5 | 10 | 20 | 30 | 50 |
|---|---|---|---|---|---|---|---|
| mean HR2 | 0.978 | 0.977 | 0.970 | 0.963 | 0.960 | 0.958 | 0.959 |

Frequency selection beats the default answer (0.969) from about 10 samples (320 tokens) and flattens after 20-30.

## What this means for the track

1. **Measurement.** The team's "known" label is inflated by option cues.
   - Year-matched real teams and the title-swapped floor (about 16%) are the fair reference.
   - With the model's own wrong answers as distractors, a third of "known" papers drop out.
2. **Failure split.** At k = 50, 2,625 of the 6,179 known papers (42%) have a true author that the default
   answer misses but some sample names. Coverage keeps growing with k, so "never generated at k = 50" overstates
   what is unreachable.
3. **Method.** Frequency is the selector to beat.
   - Single-candidate P(True) is worse than frequency.
   - Joint choice among candidates is slightly better (+3 points).
   - A PMI-style background discount adds nothing.
   - Against thinking mode, frequency selection over ordinary samples ties at equal tokens on HR2 and wins on
     true and false names. Self-consistency over thinking answers is the strongest HR2 option at higher budgets.

## Caveats

- Coverage is a ceiling ("in the pool"), not a method result. Selection recovers only part of it.
- The topic-matched baseline removes what knowing who works on a topic predicts. It cannot rule out that some
  "reachable" names come from author-topic association rather than memory of the paper.
- Labels are ours, built with the team's rules. They are not Hana's labels; her files were not available.
- One model and one task. Thinking mode was run once per paper at Qwen's recommended settings.

## 5. Deep-dive on the 1,023-paper subset

Up to 10 papers per field x citation bin. This subset over-weights the rare high-citation bins, so its levels
are better than the full sample's; compare rows within this section only.

**Coverage keeps growing past 50 samples.** Share of true authors named by at least one of k answers:

| answers | k | tokens / paper | true | topic baseline | net |
|---|---|---|---|---|---|
| ordinary samples | 20 | 643 | 0.172 | 0.010 | 0.162 |
| ordinary samples | 50 | 1,606 | 0.241 | 0.019 | 0.222 |
| ordinary samples | 100 | 3,213 | 0.299 | 0.029 | 0.270 |
| ordinary samples | 200 | 6,425 | 0.358 | 0.045 | 0.314 |
| thinking (CoT) answers | 1 | 634 | 0.066 | 0.001 | 0.065 |
| thinking (CoT) answers | 5 | 3,169 | 0.135 | 0.002 | 0.133 |

- **"Unreachable" depends on k.** Each doubling of samples adds about six points of net coverage, so many
  authors unreached at 50 samples are improbable rather than absent. Failure splits must state k.
- **Per token, ordinary samples reach about 2.5x more true authors than thinking answers.** Thinking answers
  are individually more accurate but cost about 20 ordinary samples each.
- The 200-sample run reproduces the 50-sample estimate at k = 50 (0.241 vs 0.248).

**Selection at about 3,200 tokens per paper** (716 subset test papers; thresholds from the subset's tune papers):

| method | mean HR2 | HR2 <= 0.5 | true / 100 | false / 100 |
|---|---|---|---|---|
| 5 CoT answers, names in >= 2 | 0.913 | 7.3% | 30.0 | 124 |
| 100 ordinary samples, names in >= 9 | 0.929 | 5.5% | 35.5 | 212 |
| paired difference [95% CI] | -0.016 [-0.028, -0.005] | | -5.4 [-9.2, -1.8] | -88 [-112, -65] |

**At this budget there is a real trade-off.**

- Self-consistency over thinking answers has the better mean HR2 and invents fewer names.
- Frequency over ordinary samples names more true authors.
- At about 650 tokens (section 4) the two are tied on HR2, and ordinary samples win on true and false names.
