# Data licence

The **code** in this repository is MIT (`LICENSE`). The **data** is not, and
its terms depend on where each piece came from.

## What is in `data/release/`

| | Origin | Status |
|---|---|---|
| `eval_sample_9108.csv.gz` | derived from OAG v2 and the Semantic Scholar API | see below |
| `responses/*.csv.gz` | our own model queries, plus the ground-truth author lists | outputs are ours; ground truth is derived metadata |
| `probing/`, `prompt_sensitivity/` | our own experimental outputs | CC BY 4.0 |
| `name_frequency/` | aggregate surname counts derived from the corpus | CC BY 4.0 |

Everything we generated ourselves — model responses, evaluation scores, probe
outputs, aggregate counts — is released under **CC BY 4.0**. Cite the paper.

## Bibliographic metadata: unresolved

Titles, author names, years, DOIs, citation counts, and field labels are
derived from:

* **Microsoft Open Academic Graph v2** — <https://www.aminer.cn/oag-2-1>
* **Semantic Scholar Academic Graph API** — <https://api.semanticscholar.org/>

Both carry their own terms, and Semantic Scholar's constrain redistribution of
derived corpora. **We have not yet confirmed that the 9,108-paper sample and
the full 1.92M-paper corpus can be redistributed as-is.** Until that is
resolved:

* Treat the metadata columns as subject to the upstream terms, not CC BY 4.0.
* If redistribution turns out to be restricted, the fallback is to ship the
  `paperId` list plus a rebuild script and have each user re-query the API.

This is tracked as an open item in README.md. Anyone preparing the
public release should settle it first.

## Personal data

The data contains real author names. It is public bibliographic metadata, but
it is still personal data.

* All results are aggregate. Nothing in this repository produces a per-author
  record, and none is shipped.
* A hallucinated name produced by a model is **not a claim about the person
  who bears that name.** The `hallucinated` column records what a model
  generated, nothing more.
* Appendix A6.10 infers a coarse category from surnames using `ethnicseer`.
  Those inferred labels are computed at analysis time and are **never written
  to disk**. The shipped tallies are surname counts only. Surnames are a noisy
  proxy: they do not capture self-identification, they conflate national
  origin with ethnicity, and they misclassify names between categories. The
  resulting effect sizes are descriptive.

If you are an author whose name appears here and you would like it removed,
contact the corresponding author.
