# Session 2 analysis

Split: 2734 tune / 6374 test papers (stratified by field x citation bin).

## Selection, tuned on tune, scored on test

Chosen on tune: count>=7, top3, count-0.5*bg>=6 (bg = mean count of the name over the other papers of the field). Mean HR2: lower is better; recalled = HR2 <= 0.5.

| method                                                                     |   tokens_per_paper |   mean_hr2 |   recalled_share |   true_per100 |   false_per100 |
|:---------------------------------------------------------------------------|-------------------:|-----------:|-----------------:|--------------:|---------------:|
| default answer (greedy)                                                    |            32.9198 |     0.9687 |           0.0254 |       13.3511 |       516.144  |
| count>=7 of 50 samples (tuned)                                             |          1609.74   |     0.9585 |           0.0336 |       17.2105 |       101.208  |
| top3 of 50 samples (tuned)                                                 |          1609.74   |     0.9588 |           0.0234 |       18.7324 |       281.032  |
| count-0.5*bg>=6 of 50 samples (tuned)                                      |          1609.74   |     0.9582 |           0.0337 |       17.6655 |       104.173  |
| count>=25 of 50 samples (fixed)                                            |          1609.74   |     0.9787 |           0.021  |        5.3499 |         5.3499 |
| count>=10 of 50 samples (fixed)                                            |          1609.74   |     0.9634 |           0.0348 |       12.802  |        45.9523 |
| count>=5 of 50 samples (fixed)                                             |          1609.74   |     0.9577 |           0.0304 |       22.0897 |       205.821  |
| thinking mode (CoT), 1 answer; refusals scored as names [6374 test papers] |           646      |     0.9577 |           0.0353 |       15.4063 |       353.969  |
| thinking mode (CoT), 1 answer; refusals as no answer [6374 test papers]    |           646      |     0.9577 |           0.0353 |       15.4063 |       331.142  |

CoT: 6374 test papers; median 547 tokens; 0.6% hit the token limit before answering (scored as no names); 10.9% of answers are refusals ("I cannot provide the authors..."), which the paper's scoring splits into one or more false names. Greedy answers with a refusal on the test split: 22.

## Compute curve (first k samples, threshold tuned per k; fast pool scoring on test)

|   k | rule     |   tokens_per_paper |   mean_hr2 |   recalled_share |   true_per100 |   false_per100 |
|----:|:---------|-------------------:|-----------:|-----------------:|--------------:|---------------:|
|   1 | count>=1 |            32.0199 |     0.9781 |           0.0184 |        8.9896 |       504.471  |
|   2 | count>=1 |            64.6015 |     0.9786 |           0.0138 |       13.9316 |      1012.05   |
|   3 | count>=2 |            96.9164 |     0.9768 |           0.0232 |        6.4481 |        24.0979 |
|   5 | count>=2 |           161.745  |     0.9695 |           0.0286 |       10.1192 |        60.4801 |
|  10 | count>=2 |           323.085  |     0.9633 |           0.0276 |       16.9752 |       181.001  |
|  20 | count>=3 |           644.534  |     0.9599 |           0.0309 |       18.5441 |       154.769  |
|  30 | count>=3 |           966.028  |     0.958  |           0.0276 |       24.3019 |       289.238  |
|  50 | count>=7 |          1609.74   |     0.9585 |           0.0336 |       17.1949 |       101.224  |

## Coverage by author position

All papers:

| position      |   authors |   named_by_default |   in_any_sample |
|:--------------|----------:|-------------------:|----------------:|
| first author  |      9108 |             0.047  |          0.1918 |
| other authors |     24684 |             0.0317 |          0.1921 |

Real-team known papers:

| position      |   authors |   named_by_default |   in_any_sample |
|:--------------|----------:|-------------------:|----------------:|
| first author  |      6179 |             0.0337 |          0.2225 |
| other authors |     18484 |             0.0269 |          0.213  |

## Candidate multiple choice (top-4 sampled names + 'none of these')

| items                  |   n_items |   model_correct |   most_frequent_correct |   model_chose_E |
|:-----------------------|----------:|----------------:|------------------------:|----------------:|
| pool has a true author |      5564 |          0.5888 |                  0.5586 |          0.0884 |
| pool has none          |     30768 |          0.3022 |                nan      |          0.3022 |

most_frequent_correct: picking the candidate sampled most often, on the same items.

## Real-team distractors with and without year matching (+-2 years)

| level          | condition     |   papers |   per_question |   >=3 of 4 |
|:---------------|:--------------|---------:|---------------:|-----------:|
| real_team      | standard      |     9108 |         0.7556 |     0.7027 |
| real_team      | title_hidden  |     1023 |         0.3514 |     0.2375 |
| real_team      | title_swapped |     1023 |         0.2556 |     0.1574 |
| real_team_year | standard      |     9108 |         0.7269 |     0.6681 |
| real_team_year | title_hidden  |     1023 |         0.2886 |     0.176  |
| real_team_year | title_swapped |     1023 |         0.2656 |     0.1584 |

## Robustness: sampling settings (1,023-paper control subset)

| sampling                            |   papers |   cov50_true |   cov10_true |   cov50_chance |   cov10_chance |   cov50_chance_topic |   cov10_chance_topic |   distinct_lists |
|:------------------------------------|---------:|-------------:|-------------:|---------------:|---------------:|---------------------:|---------------------:|-----------------:|
| T=1.0, top-p 1, no top-k (main run) |     1023 |       0.2475 |       0.131  |         0.0133 |         0.0041 |               0.0191 |               0.0061 |          47.3597 |
| T=0.7, top-p 0.8, top-k 20          |     1023 |       0.2125 |       0.1363 |         0.0101 |         0.0044 |               0.0132 |               0.0056 |          41.5855 |

