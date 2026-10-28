# Steering unit on known test papers

| known by   |   papers | method             |   true_per100 |   false_per100 | extra_true_vs_default   |
|:-----------|---------:|:-------------------|--------------:|---------------:|:------------------------|
| real_team  |     4271 | default answer     |          11.3 |          528.5 | +0.0 [+0.0, +0.0]       |
| real_team  |     4271 | CoT, 1 answer      |          16.5 |          344.8 | +5.2 [+3.7, +6.6]       |
| real_team  |     4271 | >= 3 of 20 samples |          19.3 |          167.2 | +8.0 [+6.6, +9.4]       |
| real_team  |     4271 | >= 7 of 50 samples |          17.4 |          110.7 | +6.1 [+4.8, +7.4]       |
| own_sample |     2481 | default answer     |          16.1 |          526   | +0.0 [+0.0, +0.0]       |
| own_sample |     2481 | CoT, 1 answer      |          24.6 |          382.4 | +8.5 [+6.0, +11.0]      |
| own_sample |     2481 | >= 3 of 20 samples |          27.9 |          152.2 | +11.7 [+9.6, +13.9]     |
| own_sample |     2481 | >= 7 of 50 samples |          25.7 |           99.3 | +9.6 [+7.6, +11.6]      |
