# Generate-then-Select

Minyan Xu's track of the CS 7980 team project "Enhancing Memory of LLMs". It asks whether sampling many
answers and then selecting among them recovers paper authors that Qwen3-32B recognises but fails to name.
The benchmark is the Dreaming Up Authors 9,108-paper sample.

- `RESULTS.md`: results so far (reproduction, recognition tests, failure split, selection, equal compute).
- `docs/slides/`: Update 2 slides (`Generate_then_Select_Update2.pptx` and `.pdf`) and the talk script.
- `src/gts/`: prompts, name matching, multiple-choice items, vLLM runners.
- `tools/`: item builders and analysis scripts; `tools/slides/build_update2.js` builds the slides.
- `colab/`: scripts run on Colab (Qwen3-32B in bf16 through vLLM on a 96 GB GPU).
- `data/`: the benchmark copy (`release/`), multiple-choice items, split v1, and every model output
  (`outputs/`, described in its README).
- `third_party/dua/`: the release's matcher and multiple-choice prompt (MIT); data terms are in `DATA_LICENSE.md`.

Python 3.10+ with the dependencies in `pyproject.toml`. nameparser and Unidecode are pinned so that scores match
the release. Run the tests with `python -m pytest -q`. The slides need Node:
`cd tools/slides && npm install && node build_update2.js`. The analysis scripts rebuild the cache
`data/outputs/g4_bf16/analysis/pools.pkl`, which is not in the repository.
