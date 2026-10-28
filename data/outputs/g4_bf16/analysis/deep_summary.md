# Deep-dive on the control subset

1023 papers (10 per field x citation bin where available); 307 tune, 716 test.

CoT answers: 634 tokens on average; 0.6% hit the token limit (no answer).

## Coverage per true author vs. compute

| answers                              |   k |   tokens_per_paper |   true |   chance |   chance_topic |   net_topic |
|:-------------------------------------|----:|-------------------:|-------:|---------:|---------------:|------------:|
| non-thinking, 50 samples (session 1) |   1 |               32.2 | 0.0399 |   0.0005 |         0.0008 |      0.0391 |
| non-thinking, 50 samples (session 1) |   5 |              161   | 0.0952 |   0.0023 |         0.0035 |      0.0918 |
| non-thinking, 50 samples (session 1) |  10 |              321.9 | 0.131  |   0.0041 |         0.0061 |      0.1249 |
| non-thinking, 50 samples (session 1) |  20 |              643.8 | 0.175  |   0.007  |         0.0103 |      0.1647 |
| non-thinking, 50 samples (session 1) |  50 |             1609.6 | 0.2475 |   0.0133 |         0.0191 |      0.2284 |
| non-thinking, 200 samples            |   1 |               32.1 | 0.0393 |   0.0005 |         0.0007 |      0.0386 |
| non-thinking, 200 samples            |   5 |              160.6 | 0.0937 |   0.0021 |         0.0032 |      0.0905 |
| non-thinking, 200 samples            |  10 |              321.2 | 0.129  |   0.0038 |         0.0057 |      0.1232 |
| non-thinking, 200 samples            |  20 |              642.5 | 0.1721 |   0.0067 |         0.0097 |      0.1624 |
| non-thinking, 200 samples            |  50 |             1606.2 | 0.2407 |   0.0129 |         0.0185 |      0.2223 |
| non-thinking, 200 samples            | 100 |             3212.5 | 0.2993 |   0.0202 |         0.029  |      0.2703 |
| non-thinking, 200 samples            | 200 |             6424.9 | 0.3584 |   0.0306 |         0.0445 |      0.3138 |
| thinking (CoT), 5 answers            |   1 |              633.8 | 0.0657 |   0.0002 |         0.0006 |      0.0651 |
| thinking (CoT), 5 answers            |   2 |             1267.6 | 0.0921 |   0.0004 |         0.0012 |      0.091  |
| thinking (CoT), 5 answers            |   3 |             1901.5 | 0.1101 |   0.0006 |         0.0016 |      0.1085 |
| thinking (CoT), 5 answers            |   5 |             3169.1 | 0.135  |   0.0009 |         0.0022 |      0.1328 |

## Frequency selection at matched compute (subset test papers)

| answers      |   k | rule      |   tokens_per_paper |   mean_hr2 |   recalled_share |   true_per100 |   false_per100 |
|:-------------|----:|:----------|-------------------:|-----------:|-----------------:|--------------:|---------------:|
| non-thinking |   5 | count>=2  |            161.399 |     0.9383 |           0.0573 |       19.4134 |        66.6201 |
| non-thinking |  10 | count>=2  |            321.7   |     0.9349 |           0.0559 |       28.4916 |       187.709  |
| non-thinking |  20 | count>=3  |            644.036 |     0.9314 |           0.0587 |       30.5866 |       163.966  |
| non-thinking |  50 | count>=7  |           1613.3   |     0.9311 |           0.0628 |       28.0726 |       106.844  |
| non-thinking | 100 | count>=9  |           3232.17  |     0.9291 |           0.0545 |       35.4749 |       212.291  |
| non-thinking | 200 | count>=22 |           6466.91  |     0.9291 |           0.0573 |       31.0056 |       132.402  |
| thinking     |   1 | count>=1  |            631.23  |     0.9303 |           0.0587 |       24.7207 |       276.536  |
| thinking     |   2 | count>=1  |           1241.55  |     0.931  |           0.0489 |       35.3352 |       511.592  |
| thinking     |   3 | count>=2  |           1871.29  |     0.925  |           0.0684 |       22.067  |        56.9832 |
| thinking     |   5 | count>=2  |           3140.06  |     0.9131 |           0.0726 |       30.0279 |       124.441  |

