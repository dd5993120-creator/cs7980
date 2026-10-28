# Selection on the candidate pools

90745 candidates scored (9108 papers, top 10 per paper); 2818 are true authors.

Pairwise AUC (true vs false candidates of the same paper), by score:

| group          |   papers |   papers_with_true_candidate |   auc_count |   auc_p_true |   top1_count |   top1_p_true |
|:---------------|---------:|-----------------------------:|------------:|-------------:|-------------:|--------------:|
| all            |     9108 |                         1864 |       0.74  |        0.606 |        0.427 |         0.313 |
| label=recalled |      222 |                          218 |       0.964 |        0.645 |        0.922 |         0.528 |
| label=known    |     6179 |                         1520 |       0.713 |        0.603 |        0.374 |         0.295 |
| label=notsure  |      924 |                           52 |       0.586 |        0.573 |        0.173 |         0.212 |
| label=unknown  |     1783 |                           74 |       0.617 |        0.54  |        0.216 |         0.108 |
| bins 0-32      |     3995 |                          303 |       0.662 |        0.589 |        0.261 |         0.221 |
| bins 33-512    |     3200 |                          689 |       0.693 |        0.595 |        0.327 |         0.263 |
| bins 513+      |     1913 |                          872 |       0.795 |        0.618 |        0.563 |         0.384 |

True and false names among the top k candidates, per 100 papers (default answer for reference):

| group          |   default_named_per100 |   top1_count_true_per100 |   top1_p_true_true_per100 |   top3_count_true_per100 |   top3_p_true_true_per100 |   top3_count_false_per100 |   top3_p_true_false_per100 |
|:---------------|-----------------------:|-------------------------:|--------------------------:|-------------------------:|--------------------------:|--------------------------:|---------------------------:|
| all            |                  13.3  |                     8.73 |                      6.4  |                    18.5  |                     13.5  |                    281.25 |                     286.24 |
| label=recalled |                 206.76 |                    90.54 |                     51.8  |                   179.73 |                    102.25 |                    112.16 |                     189.64 |
| label=known    |                  11.41 |                     9.21 |                      7.27 |                    19.79 |                     15.39 |                    280.14 |                     284.54 |
| label=notsure  |                   2.6  |                     0.97 |                      1.19 |                     2.49 |                      2.71 |                    297.51 |                     297.29 |
| label=unknown  |                   1.29 |                     0.9  |                      0.45 |                     2.24 |                      1.51 |                    297.7  |                     298.43 |
| bins 0-32      |                   3.25 |                     1.98 |                      1.68 |                     4.76 |                      3.78 |                    295.19 |                     296.17 |
| bins 33-512    |                  10.16 |                     7.03 |                      5.66 |                    15.62 |                     12.03 |                    284.38 |                     287.97 |
| bins 513+      |                  39.52 |                    25.67 |                     17.51 |                    52.01 |                     36.28 |                    246.89 |                     262.62 |


Frequency thresholds versus the default answer (descriptive; thresholds fixed in advance, not tuned; candidates are the top 10 names of each pool, so a low threshold can miss names outside the top 10):

| group               |   papers | method                       |   true_per100 |   false_per100 |
|:--------------------|---------:|:-----------------------------|--------------:|---------------:|
| all                 |     9108 | default answer               |          13.3 |          514.3 |
| all                 |     9108 | names in >= 25 of 50 samples |           5.3 |            5.1 |
| all                 |     9108 | names in >= 10 of 50 samples |          12.4 |           45.6 |
| all                 |     9108 | names in >= 5 of 50 samples  |          20.8 |          185.2 |
| known (real_team)   |     6179 | default answer               |          11.4 |          525.3 |
| known (real_team)   |     6179 | names in >= 25 of 50 samples |           3.3 |            5.9 |
| known (real_team)   |     6179 | names in >= 10 of 50 samples |          11.3 |           50.6 |
| known (real_team)   |     6179 | names in >= 5 of 50 samples  |          22.3 |          199   |
| known (own_sample)  |     3552 | default answer               |          16.1 |          528   |
| known (own_sample)  |     3552 | names in >= 25 of 50 samples |           5.4 |            5.9 |
| known (own_sample)  |     3552 | names in >= 10 of 50 samples |          17.5 |           46.8 |
| known (own_sample)  |     3552 | names in >= 5 of 50 samples  |          32   |          183.6 |
| unknown (real_team) |     1783 | default answer               |           1.3 |          517.8 |
| unknown (real_team) |     1783 | names in >= 25 of 50 samples |           0.1 |            3.3 |
| unknown (real_team) |     1783 | names in >= 10 of 50 samples |           0.9 |           33.1 |
| unknown (real_team) |     1783 | names in >= 5 of 50 samples  |           2.6 |          151.8 |
