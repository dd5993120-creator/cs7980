# Session 3: the recognition ladder, catch trials, formats and selection scores

## 1. The recognition ladder

Per-question accuracy (chance 0.25), chance-corrected accuracy, the share of papers whose true list is
chosen in >= 3 of 4 orders (guessing: about 0.05), and the pairwise AUC of the true list against the
three wrong ones (chance 0.5). Controls on the 1,023-paper subset.

| level   | distractors    | condition     |   papers |   accuracy |   chance_corrected |   >=3 of 4 |   AUC |
|:--------|:---------------|:--------------|---------:|-----------:|-------------------:|-----------:|------:|
| L0      | random_field   | standard      |     9108 |      0.784 |              0.712 |      0.737 | 0.911 |
| L0      | random_field   | title_hidden  |     1023 |      0.479 |              0.305 |      0.373 | 0.702 |
| L0      | random_field   | title_swapped |     1023 |      0.284 |              0.045 |      0.175 | 0.516 |
| L1      | real_team      | standard      |     9108 |      0.756 |              0.674 |      0.703 | 0.896 |
| L1      | real_team      | title_hidden  |     1023 |      0.351 |              0.135 |      0.238 | 0.605 |
| L1      | real_team      | title_swapped |     1023 |      0.256 |              0.007 |      0.157 | 0.507 |
| L2      | real_team_year | standard      |     9108 |      0.727 |              0.636 |      0.668 | 0.877 |
| L2      | real_team_year | title_hidden  |     1023 |      0.289 |              0.051 |      0.176 | 0.537 |
| L2      | real_team_year | title_swapped |     1023 |      0.266 |              0.021 |      0.158 | 0.525 |
| L3      | similar_title  | standard      |     9108 |      0.711 |              0.614 |      0.649 | 0.864 |
| L3      | similar_title  | title_hidden  |     1023 |      0.286 |              0.049 |      0.179 | 0.538 |
| L3      | similar_title  | title_swapped |     1023 |      0.270 |              0.026 |      0.164 | 0.504 |
| L3t     | similar_topic  | standard      |     9108 |      0.714 |              0.619 |      0.652 | 0.876 |
| L3t     | similar_topic  | title_hidden  |     1023 |      0.357 |              0.143 |      0.242 | 0.617 |
| L3t     | similar_topic  | title_swapped |     1023 |      0.272 |              0.030 |      0.160 | 0.537 |
| L4      | collab_slot    | standard      |     9108 |      0.485 |              0.314 |      0.358 | 0.714 |
| L4      | collab_slot    | title_hidden  |     1023 |      0.355 |              0.140 |      0.167 | 0.633 |
| L4      | collab_slot    | title_swapped |     1023 |      0.306 |              0.074 |      0.147 | 0.563 |
| L4r     | collab_swap    | standard      |     9108 |      0.685 |              0.579 |      0.646 | 0.875 |
| L4r     | collab_swap    | title_hidden  |     1023 |      0.574 |              0.432 |      0.510 | 0.843 |
| L4r     | collab_swap    | title_swapped |     1023 |      0.505 |              0.340 |      0.406 | 0.765 |
| L5      | own_sample     | standard      |     9108 |      0.496 |              0.329 |      0.412 | 0.709 |
| L5      | own_sample     | title_hidden  |     1023 |      0.372 |              0.162 |      0.252 | 0.610 |
| L5      | own_sample     | title_swapped |     1023 |      0.274 |              0.032 |      0.155 | 0.503 |

**>=3 of 4 by citation bin** (standard condition):

| citation_bin   |    L0 |    L1 |    L2 |    L3 |   L3t |    L4 |   L4r |    L5 |
|:---------------|------:|------:|------:|------:|------:|------:|------:|------:|
| 0-2            | 0.541 | 0.465 | 0.457 | 0.413 | 0.456 | 0.256 | 0.496 | 0.175 |
| 3-4            | 0.542 | 0.507 | 0.417 | 0.379 | 0.426 | 0.257 | 0.536 | 0.188 |
| 5-8            | 0.583 | 0.522 | 0.500 | 0.458 | 0.476 | 0.263 | 0.530 | 0.210 |
| 9-16           | 0.637 | 0.586 | 0.537 | 0.512 | 0.554 | 0.266 | 0.557 | 0.242 |
| 17-32          | 0.677 | 0.636 | 0.586 | 0.559 | 0.555 | 0.246 | 0.618 | 0.286 |
| 33-64          | 0.714 | 0.667 | 0.637 | 0.641 | 0.632 | 0.299 | 0.646 | 0.340 |
| 65-128         | 0.767 | 0.751 | 0.706 | 0.689 | 0.661 | 0.329 | 0.681 | 0.375 |
| 129-256        | 0.800 | 0.802 | 0.757 | 0.752 | 0.725 | 0.351 | 0.676 | 0.459 |
| 257-512        | 0.868 | 0.825 | 0.810 | 0.792 | 0.781 | 0.414 | 0.708 | 0.559 |
| 513-1024       | 0.915 | 0.907 | 0.880 | 0.877 | 0.854 | 0.540 | 0.770 | 0.701 |
| 1025-2048      | 0.956 | 0.946 | 0.929 | 0.932 | 0.926 | 0.598 | 0.811 | 0.786 |
| 2049-4096      | 0.982 | 0.966 | 0.969 | 0.960 | 0.951 | 0.607 | 0.813 | 0.874 |
| 4097+          | 1.000 | 1.000 | 1.000 | 1.000 | 0.992 | 0.721 | 0.910 | 0.967 |

**AUC by citation bin** (standard condition):

| citation_bin   |    L0 |    L1 |    L2 |    L3 |   L3t |    L4 |   L4r |    L5 |
|:---------------|------:|------:|------:|------:|------:|------:|------:|------:|
| 0-2            | 0.828 | 0.816 | 0.778 | 0.753 | 0.802 | 0.667 | 0.812 | 0.562 |
| 3-4            | 0.854 | 0.809 | 0.780 | 0.748 | 0.785 | 0.660 | 0.825 | 0.568 |
| 5-8            | 0.859 | 0.828 | 0.792 | 0.777 | 0.793 | 0.666 | 0.813 | 0.566 |
| 9-16           | 0.865 | 0.849 | 0.827 | 0.808 | 0.839 | 0.646 | 0.843 | 0.608 |
| 17-32          | 0.899 | 0.877 | 0.848 | 0.833 | 0.845 | 0.651 | 0.856 | 0.623 |
| 33-64          | 0.896 | 0.887 | 0.872 | 0.858 | 0.875 | 0.672 | 0.867 | 0.679 |
| 65-128         | 0.923 | 0.913 | 0.902 | 0.894 | 0.894 | 0.699 | 0.894 | 0.715 |
| 129-256        | 0.935 | 0.933 | 0.910 | 0.907 | 0.902 | 0.709 | 0.887 | 0.761 |
| 257-512        | 0.957 | 0.945 | 0.937 | 0.931 | 0.927 | 0.749 | 0.907 | 0.811 |
| 513-1024       | 0.980 | 0.976 | 0.971 | 0.968 | 0.953 | 0.830 | 0.945 | 0.878 |
| 1025-2048      | 0.985 | 0.984 | 0.978 | 0.978 | 0.974 | 0.843 | 0.945 | 0.919 |
| 2049-4096      | 0.994 | 0.993 | 0.994 | 0.992 | 0.990 | 0.859 | 0.950 | 0.954 |
| 4097+          | 1.000 | 1.000 | 1.000 | 1.000 | 0.997 | 0.893 | 0.970 | 0.992 |

## 2. Labels at each level

Team rules: recalled = default HR2 <= 0.5; otherwise known / notsure / unknown = the true list chosen
in >= 3 / 2 / <= 1 of the 4 orders.

| level   |   known |   notsure |   recalled |   unknown |
|:--------|--------:|----------:|-----------:|----------:|
| L0      |    6491 |       836 |        222 |      1559 |
| L1      |    6179 |       924 |        222 |      1783 |
| L2      |    5865 |       968 |        222 |      2053 |
| L3      |    5694 |       953 |        222 |      2239 |
| L3t     |    5724 |      1027 |        222 |      2135 |
| L4      |    3070 |      1526 |        222 |      4290 |
| L4r     |    5695 |      1231 |        222 |      1960 |
| L5      |    3552 |      1107 |        222 |      4227 |

**Hard-known** (known at every level L1, L2, L3, L3t, L4, L5): 1568 papers (17.2%); known at L1: 6179.

| citation_bin   |   hard-known papers |   share |
|:---------------|--------------------:|--------:|
| 0-2            |                  53 |   0.067 |
| 3-4            |                  45 |   0.056 |
| 5-8            |                  60 |   0.075 |
| 9-16           |                  67 |   0.084 |
| 17-32          |                  78 |   0.098 |
| 33-64          |                  99 |   0.124 |
| 65-128         |                 121 |   0.151 |
| 129-256        |                 144 |   0.180 |
| 257-512        |                 177 |   0.221 |
| 513-1024       |                 264 |   0.330 |
| 1025-2048      |                 275 |   0.414 |
| 2049-4096      |                 134 |   0.411 |
| 4097+          |                  51 |   0.418 |

**Reliability check** (proposal): recalled papers should almost all be recognised.

| level   |   recalled & recognised |   recalled, not recognised |   not recalled, recognised |   neither |
|:--------|------------------------:|---------------------------:|---------------------------:|----------:|
| L0      |                     220 |                          2 |                       6491 |      2395 |
| L1      |                     221 |                          1 |                       6179 |      2707 |
| L2      |                     220 |                          2 |                       5865 |      3021 |
| L3      |                     218 |                          4 |                       5694 |      3192 |
| L3t     |                     217 |                          5 |                       5724 |      3162 |
| L4      |                     189 |                         33 |                       3070 |      5816 |
| L4r     |                     192 |                         30 |                       5695 |      3191 |
| L5      |                     201 |                         21 |                       3552 |      5334 |

## 3. Catch trials (L1 distractors, with "None of the above")

Per question:

| items                     |   true list chosen |   None chosen |   a wrong list chosen |
|:--------------------------|-------------------:|--------------:|----------------------:|
| true list present         |              0.581 |         0.367 |                 0.051 |
| true list removed (catch) |            nan     |         0.721 |                 0.279 |

Per paper, >= 3 of 4 orders: hit = the true list chosen when present; reject = None chosen when the
true list is removed; both = a recognition that survives the catch trial.

| L1 label   |   hit |   reject |   both |   papers |
|:-----------|------:|---------:|-------:|---------:|
| known      | 0.654 |    0.789 |  0.489 |     6179 |
| notsure    | 0.133 |    0.611 |  0.023 |      924 |
| recalled   | 0.856 |    0.928 |  0.784 |      222 |
| unknown    | 0.050 |    0.551 |  0.002 |     1783 |

| citation_bin   |   hit |   reject |   both |
|:---------------|------:|---------:|-------:|
| 0-2            | 0.220 |    0.751 |  0.132 |
| 3-4            | 0.242 |    0.720 |  0.153 |
| 5-8            | 0.254 |    0.696 |  0.156 |
| 9-16           | 0.316 |    0.685 |  0.219 |
| 17-32          | 0.401 |    0.661 |  0.245 |
| 33-64          | 0.454 |    0.640 |  0.286 |
| 65-128         | 0.522 |    0.685 |  0.356 |
| 129-256        | 0.566 |    0.711 |  0.395 |
| 257-512        | 0.639 |    0.764 |  0.484 |
| 513-1024       | 0.741 |    0.779 |  0.569 |
| 1025-2048      | 0.830 |    0.856 |  0.710 |
| 2049-4096      | 0.890 |    0.862 |  0.761 |
| 4097+          | 0.951 |    0.893 |  0.844 |

## 4. Yes/no recognition (one author list per question)

Answer = Yes when P(Yes) > P(No). AUC uses P(Yes) within each paper.

| level   | distractors   |   papers |   hit rate |   false-alarm rate |    d' |     c |   AUC |   Yes to the true list only |
|:--------|:--------------|---------:|-----------:|-------------------:|------:|------:|------:|----------------------------:|
| L1      | real_team     |     9108 |      0.388 |              0.107 | 0.958 | 0.763 | 0.793 |                       0.240 |
| L4r     | collab_swap   |     9108 |      0.388 |              0.303 | 0.230 | 0.400 | 0.632 |                       0.065 |
| L5      | own_sample    |     9108 |      0.389 |              0.270 | 0.331 | 0.448 | 0.617 |                       0.122 |

**AUC by citation bin:**

| citation_bin   |    L1 |   L4r |    L5 |
|:---------------|------:|------:|------:|
| 0-2            | 0.740 | 0.612 | 0.532 |
| 3-4            | 0.752 | 0.595 | 0.546 |
| 5-8            | 0.761 | 0.619 | 0.586 |
| 9-16           | 0.760 | 0.590 | 0.572 |
| 17-32          | 0.772 | 0.604 | 0.577 |
| 33-64          | 0.770 | 0.608 | 0.582 |
| 65-128         | 0.792 | 0.608 | 0.606 |
| 129-256        | 0.805 | 0.626 | 0.626 |
| 257-512        | 0.814 | 0.638 | 0.665 |
| 513-1024       | 0.847 | 0.688 | 0.693 |
| 1025-2048      | 0.853 | 0.716 | 0.723 |
| 2049-4096      | 0.883 | 0.737 | 0.764 |
| 4097+          | 0.954 | 0.765 | 0.850 |

## 5. Likelihood recognition (each list as the answer to the open question)

| level   | distractors   | score                      |   papers |   AUC |   true list scores highest |
|:--------|:--------------|:---------------------------|---------:|------:|---------------------------:|
| L5      | own_sample    | log P(list | title)        |     9108 | 0.566 |                      0.366 |
| L5      | own_sample    | per token                  |     9108 | 0.584 |                      0.383 |
| L5      | own_sample    | PMI (title shown - hidden) |     9108 | 0.574 |                      0.355 |
| L1      | real_team     | log P(list | title)        |     9108 | 0.815 |                      0.662 |
| L1      | real_team     | per token                  |     9108 | 0.814 |                      0.669 |
| L1      | real_team     | PMI (title shown - hidden) |     9108 | 0.877 |                      0.757 |

**PMI AUC by citation bin:**

| citation_bin   |    L1 |    L5 |
|:---------------|------:|------:|
| 0-2            | 0.755 | 0.411 |
| 3-4            | 0.798 | 0.424 |
| 5-8            | 0.796 | 0.440 |
| 9-16           | 0.829 | 0.469 |
| 17-32          | 0.866 | 0.503 |
| 33-64          | 0.880 | 0.537 |
| 65-128         | 0.905 | 0.585 |
| 129-256        | 0.925 | 0.633 |
| 257-512        | 0.930 | 0.673 |
| 513-1024       | 0.956 | 0.730 |
| 1025-2048      | 0.961 | 0.798 |
| 2049-4096      | 0.978 | 0.840 |
| 4097+          | 0.967 | 0.811 |

## 6. Selection scores for the top-10 sampled candidates

Pooled within-paper AUC (true vs false candidates) and the share of papers with a true candidate whose
top-scored candidate is true. Combinations are fitted on the tune split and scored on test once.

| papers   | score                                             |   AUC |   top-1 true |
|:---------|:--------------------------------------------------|------:|-------------:|
| all      | sample count (of 50)                              | 0.740 |        0.427 |
| all      | P(True)                                           | 0.606 |        0.313 |
| all      | log P(a | title)                                  | 0.686 |        0.380 |
| all      | log P(a), no title                                | 0.546 |        0.193 |
| all      | PMI: log P(a | title) - log P(a)                  | 0.605 |        0.230 |
| all      | log P(title | a), reference list                  | 0.643 |        0.381 |
| all      | log P(title | a) - log P(title)                   | 0.643 |        0.381 |
| test     | sample count (of 50)                              | 0.744 |        0.433 |
| test     | P(True)                                           | 0.603 |        0.304 |
| test     | log P(a | title)                                  | 0.690 |        0.382 |
| test     | log P(a), no title                                | 0.552 |        0.200 |
| test     | PMI: log P(a | title) - log P(a)                  | 0.598 |        0.221 |
| test     | log P(title | a), reference list                  | 0.640 |        0.383 |
| test     | log P(title | a) - log P(title)                   | 0.640 |        0.383 |
| test     | logistic on tune: count + PMI                     | 0.749 |        0.434 |
| test     | logistic on tune: count + reverse                 | 0.773 |        0.473 |
| test     | logistic on tune: count + P(True)                 | 0.755 |        0.433 |
| test     | logistic on tune: count + PMI + reverse + P(True) | 0.771 |        0.473 |

**Paired bootstrap on test papers** (2,000 resamples): each combination minus the count alone.

| combination                     |   AUC gain | AUC 95% CI       |   top-1 gain | top-1 95% CI     |
|:--------------------------------|-----------:|:-----------------|-------------:|:-----------------|
| count + PMI                     |      0.005 | [+0.001, +0.010] |        0.001 | [-0.010, +0.012] |
| count + reverse                 |      0.028 | [+0.022, +0.035] |        0.041 | [+0.023, +0.059] |
| count + P(True)                 |      0.011 | [+0.007, +0.015] |        0.000 | [-0.008, +0.008] |
| count + PMI + reverse + P(True) |      0.027 | [+0.020, +0.034] |        0.040 | [+0.022, +0.059] |

**At a fixed budget** (6374 test papers): each score keeps the same number of names as
">= 7 of 50 samples" (7328), ranked across all test papers' top-10 candidates.

| score                            |   names kept |   true per 100 papers |   false per 100 papers |
|:---------------------------------|-------------:|----------------------:|-----------------------:|
| sample count (of 50)             |         7328 |                17.022 |                 97.945 |
| P(True)                          |         7328 |                 7.233 |                107.735 |
| PMI: log P(a | title) - log P(a) |         7328 |                 5.962 |                109.005 |
| log P(title | a) - log P(title)  |         7328 |                11.374 |                103.593 |
| count + PMI                      |         7328 |                17.211 |                 97.757 |
| count + reverse                  |         7328 |                18.120 |                 96.847 |
| count + P(True)                  |         7328 |                17.022 |                 97.945 |
| count + PMI + reverse + P(True)  |         7328 |                18.403 |                 96.564 |
