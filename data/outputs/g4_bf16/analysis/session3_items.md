# How the session 3 distractors came out

Donor teams: TF-IDF title similarity to the paper (same field) and the prominence gap
(|log(1 + h-index)| of each team's best-known member).

| level   |   donors |   title similarity, median |   title similarity, mean |   prominence gap, median |   prominence gap <= 0.25 |
|:--------|---------:|---------------------------:|-------------------------:|-------------------------:|-------------------------:|
| L1      |    27324 |                      0.000 |                    0.006 |                    0.716 |                    0.188 |
| L2      |    27324 |                      0.000 |                    0.006 |                    0.693 |                    0.195 |
| L3      |    27324 |                      0.000 |                    0.016 |                    0.164 |                    0.750 |
| L3t     |    27324 |                      0.157 |                    0.164 |                    0.674 |                    0.205 |

L3 donor tiers (most constrained first):

| tier                         |   donors |
|:-----------------------------|---------:|
| same_length_year_prominence  |    19226 |
| same_length_year_prominence2 |     3564 |
| truncated_year_prominence2   |     2279 |
| same_length_year             |     1317 |
| same_length                  |      612 |
| truncated_year               |      257 |
| truncated                    |       41 |
| other_field                  |       28 |

L4: 21199 of 27324 swapped-in names are real collaborators (77.6%); 6886 of 9108 papers (75.6%) have all three; the rest are random field names.
