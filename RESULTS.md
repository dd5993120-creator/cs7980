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
   - Session 3 (sections 6-7):
     - AUC over the four orders has no false-positive floor, so report it with the count rule.
     - Single cues cost little: topic 5 points, year 3.5, prominence 2.
     - What is hard is the exact team. Known falls to 36% when one co-author is swapped for a real
       collaborator (L4) and to 41% against the model's own wrong lists (L5).
     - Of the 6,179 papers known at L1, 3,021 also pass the catch trials, and 1,568 are known at every level.
     - The release's collaborator-swap design leaks (title-swapped control 0.41). Use a shared slot.
2. **Failure split.** At k = 50, 2,625 of the 6,179 known papers (42%) have a true author that the default
   answer misses but some sample names. Coverage keeps growing with k, so "never generated at k = 50" overstates
   what is unreachable.
3. **Method.** Frequency is the selector to beat.
   - Single-candidate P(True) is worse than frequency.
   - Joint choice among candidates is slightly better (+3 points).
   - A PMI-style background discount adds nothing.
   - Against thinking mode, frequency selection over ordinary samples ties at equal tokens on HR2 and wins on
     true and false names. Self-consistency over thinking answers is the strongest HR2 option at higher budgets.
   - Session 3: scoring each candidate in the authors-to-title direction is the one addition that helps
     frequency (+2.8 AUC points, +4 points top-1). Likelihood PMI and P(True) do not help it.
   - Against its own wrong lists, the model's likelihood fails (AUC 0.57) while its choice does not (0.71).
     Selection carries information beyond generation.

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

## 6. Session 3: the whole recognition ladder, catch trials and yes/no (2026-10-28)

**Protocol.** Same model and protocol as before: four cyclic orders, every option shown as "F. Lastname", and
the answer read from the first token's log-probabilities. Every level is asked on all 9,108 papers, with both
controls on the 1,023-paper subset.

**New this session.**
- **Pairwise AUC per level.** The true list's letter probability, averaged over the four orders, against each
  wrong list's.
- **Author data from the Semantic Scholar API.** The sample's paper ids are Semantic Scholar ids, so authors
  resolve exactly (tools/fetch_s2.py). Two things were fetched:
  - each author's h-index, used for team prominence;
  - the co-authors of each first author on up to 20 of their other papers, closest in year.

**The ladder.** Each cell gives the share of papers right in >= 3 of 4 orders, then the AUC. Guessing gives
0.05 and 0.5.

| level | wrong options | standard | title hidden | title swapped |
|---|---|---|---|---|
| L0 | random names from the paper's field (the first test's design) | 0.737 / 0.911 | 0.373 / 0.702 | 0.175 / 0.516 |
| L1 | real teams of the field, same size (the team test) | 0.703 / 0.896 | 0.238 / 0.605 | 0.157 / 0.507 |
| L2 | L1, published within +-2 years | 0.668 / 0.877 | 0.176 / 0.537 | 0.158 / 0.525 |
| L3 | L2, teams of matched prominence (best-known member's h-index) | 0.649 / 0.864 | 0.179 / 0.538 | 0.164 / 0.504 |
| L3t | real teams of the 3 most similar titles, any year | 0.652 / 0.876 | 0.242 / 0.617 | 0.160 / 0.537 |
| L4 | the true list with one co-author replaced by a real collaborator of the first author, the same slot in all three | 0.358 / 0.714 | 0.167 / 0.633 | 0.147 / 0.563 |
| L4r | the same, with a slot drawn per option (the release's design) | 0.646 / 0.875 | 0.510 / 0.843 | 0.406 / 0.765 |
| L5 | the model's own most frequent wrong lists | 0.412 / 0.709 | 0.252 / 0.610 | 0.155 / 0.503 |

- **AUC has no 16% floor.**
  - With another paper's title, AUC is 0.50-0.54 at every level except L4 and L4r.
  - Under the same control, >= 3 of 4 stays at 0.15-0.18.
  - The model prefers some options in every order. That preference inflates the count rule; averaging over the
    four orders cancels it.
- **Each surface cue costs little.** These are the drops in >= 3 of 4:

  | cue | levels compared | points |
  |---|---|---|
  | topic | L1 to L3t | 5.1 |
  | year | L1 to L2 | 3.5 |
  | prominence, on top of the year | L2 to L3 | 1.9 |

  Random names leak the most: with the title hidden, 0.373 of papers still pass (AUC 0.70).
- **The team itself is what is hard.**
  - Swapping one co-author for a real collaborator (L4) cuts "known" to 36%.
  - The model's own wrong lists (L5) cut it to 41%.
  - Even at 4097+ citations, L4 is only 0.72.
- **The release's collaborator swap leaks.**
  - With a slot drawn per option (L4r), the true list is the one whose every name recurs in other options. The
    model can find it without knowing the paper.
  - So the title-swapped control reaches 0.406 (AUC 0.765).
  - Putting all three swaps in one slot (L4) removes this: 0.147.
- **L3 matched prominence but not topic.**
  - Prominence matched well: the gap between the true team's and the donor team's best-known member, in
    log(1 + h-index), has a median of 0.16 for L3 donors and 0.72 for L1 donors.
  - Topic barely matched: donor title similarity (TF-IDF) averages 0.016 against 0.006 for L1.
  - Within the sample, the year and team-size constraints leave few topical neighbours. L3t drops those
    constraints and reaches a similarity of 0.164.
- **Where L4's names come from.** 77.6% of the swapped-in names are real collaborators, and 75.6% of papers have
  all three. The rest are random field names, which may explain L4's slightly raised controls.
- Details are in data/outputs/g4_bf16/analysis/session3_summary.md and session3_items.md.

**Hard-known.** These are papers known at every level: L1, L2, L3, L3t, L4 and L5.
- They number 1,568 (17.2%), a quarter of the 6,179 papers known at L1.
- By citation bin, the share is:

  | citations | 0-2 | 65-128 | 513-1024 | 1025 and up |
  |---|---|---|---|---|
  | hard-known | 0.067 | 0.151 | 0.330 | about 0.41 |

  L4 caps it at the top.

**Reliability check.** Recalled papers (HR2 <= 0.5) should almost all be recognised. Of the 222, the number not
recognised is 1 at L1, 21 at L5 and 33 at L4.

**Catch trials.** These use the L1 distractors, with "None of the above" as one of the four options, asked in
all four orders.

| | true list chosen | None chosen | a wrong list chosen |
|---|---|---|---|
| true list present | 0.581 | 0.367 | 0.051 |
| true list removed | | 0.721 | 0.279 |

- **Offering "None" lowers hits** from 0.756 to 0.581 per question. When the true list is present, the model
  rejects it 37% of the time.
- **Per paper (>= 3 of 4 orders), most known papers fail one of the two tests.** Of the 6,179 papers known at L1:
  - 65.4% pick the true list when it is present;
  - 78.9% pick None when it is removed;
  - 48.9% do both.

  For comparison, 78.4% of recalled papers do both, and 0.2% of unknown ones.
- **Doing both rises with citations,** from 0.13 at 0-2 to 0.84 at 4097+.

**Yes/no, one list at a time.** The answer counts as Yes when P(Yes) > P(No).

| lists | hit rate | false-alarm rate | d' | c | AUC |
|---|---|---|---|---|---|
| L1 | 0.388 | 0.107 | 0.96 | 0.76 | 0.793 |
| L4r | 0.388 | 0.303 | 0.23 | 0.40 | 0.632 |
| L5 | 0.389 | 0.270 | 0.33 | 0.45 | 0.617 |

- **The model is conservative.** It says Yes to only 39% of true lists (c > 0).
- **Discrimination is weaker than in joint multiple choice on the same lists:**

  | lists | yes/no AUC | joint multiple choice AUC |
  |---|---|---|
  | L1 | 0.793 | 0.896 |
  | L5 | 0.617 | 0.709 |

  Seeing the options side by side helps.
- **GPT-4o was not replicated.** The proposal's GPT-4o replication of the first test needs an OpenAI API key,
  which this project does not have.

**Likelihood, the third format.** Each option list is scored as the answer to the open question (teacher-forced
log P(list | prompt), the same "F. Lastname" lists). Each AUC compares the true list with the three wrong ones
within a paper. The PMI subtracts the same list's log-probability with the title hidden.

| lists | log P(list \| title) | per token | PMI | joint multiple choice |
|---|---|---|---|---|
| L1 | 0.815 | 0.814 | 0.877 | 0.896 |
| L5 | 0.566 | 0.584 | 0.574 | 0.709 |

- **Against other papers' teams (L1), the PMI nearly matches multiple choice.**
- **Against the model's own wrong lists (L5), likelihood fails.**
  - Those lists were sampled from the model, so they are likely by construction.
  - At 0-2 citations the PMI AUC is 0.41: below chance.
- **Choosing does better than likelihood here** (0.709 vs 0.574). Recognition carries information that generation
  probability lacks, which is the premise of generate-then-select.

## 7. Session 3: likelihood scores for selection

**Candidates.** The same top-10 sampled candidates as P(True): 90,745 candidates on 9,108 papers.

**Three new scores, all teacher-forced:**
- **log P(a | title):** the candidate, in the form the model writes it most often, as the answer to "Name one
  author of the paper titled ...".
- **PMI:** the same, minus the answer's log-probability without the title (the name's prior).
- **Authors to title:** log P(title | a) after a reference-list entry, "Shi, P. et al. (2014).". Subtracting
  log P(title) after "Anonymous (2014)." is constant within a paper, so it changes no ranking there.

**Results on the test split (6,374 papers).** AUC is pooled within papers. Top-1 is the share of papers with a
true candidate whose top-scored candidate is true.

| score | AUC | top-1 true |
|---|---|---|
| sample count (of 50) | 0.744 | 0.433 |
| P(True) | 0.603 | 0.304 |
| log P(a \| title) | 0.690 | 0.382 |
| log P(a), no title | 0.552 | 0.200 |
| PMI | 0.598 | 0.221 |
| authors to title, log P(title \| a) | 0.640 | 0.383 |
| count + PMI (logistic regression fitted on tune) | 0.749 | 0.434 |
| count + P(True) | 0.755 | 0.433 |
| count + authors to title | 0.773 | 0.473 |
| all four | 0.771 | 0.473 |

**Paired bootstrap over test papers, against the count alone:**

| combination | AUC | top-1 (points) |
|---|---|---|
| count + authors to title | +0.028 [+0.022, +0.035] | +4.1 [+2.3, +5.9] |
| count + P(True) | +0.011 [+0.007, +0.015] | 0.0 [-0.8, +0.8] |
| count + PMI | +0.005 [+0.001, +0.010] | +0.1 [-1.0, +1.2] |

**At a fixed budget.** Every score keeps the same 7,328 test names that the ">= 7 of 50" rule keeps, ranked across
all papers' top-10 candidates.

| score | true / 100 papers | false / 100 papers |
|---|---|---|
| sample count | 17.0 | 97.9 |
| count + authors to title | 18.1 | 96.8 |
| all four | 18.4 | 96.6 |
| P(True) alone | 7.2 | 107.7 |
| PMI alone | 6.0 | 109.0 |

- **Frequency is still the best single score.**
- **The name's prior is signal, not noise.**
  - Even without the title, real authors are likelier names (AUC 0.552).
  - So removing the prior (PMI, 0.598) does worse than keeping it (0.690).
- **The authors-to-title direction is the one score that adds to frequency.**
  - It adds 2.8 AUC points and 4 points of top-1.
  - At the same budget it keeps about 1.1 to 1.4 more true names per 100 papers.
  - This fits the proposal's reversal-curse motivation: a reference list stores "authors (year). title", and the
    model checks the fact more easily in that direction.
