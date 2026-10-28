"""Write the Colab notebook and the bundle it unpacks.

    python tools/make_colab_bundle.py

Outputs, both to upload to Google Drive under MyDrive/capstone_gts/:
    colab/gts_bundle.zip             src/gts, the release data, the MC items
    colab/01_generate_and_mc.ipynb   session 1: greedy, 50 samples, real-team MC
"""

import zipfile
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
COLAB = ROOT / "colab"

BUNDLE_FILES = [
    *sorted((ROOT / "src" / "gts").glob("*.py")),
    ROOT / "data" / "release" / "eval_sample_9108.csv.gz",
    ROOT / "data" / "release" / "qwen3_32b_greedy_scored.tsv.gz",
    *sorted((ROOT / "data" / "mc_items").glob("*.jsonl.gz")),
    *sorted((ROOT / "data" / "splits").glob("*.csv")),
]

INTRO = """# 第一轮：采样 + 选择题（真实团队干扰项）

这个 notebook 在 Colab 的 A100 上依次做三件事，结果都存到 Google Drive 的 `capstone_gts/outputs/`：

1. **默认答案**：9,108 篇论文各 1 个贪心答案（约 5 分钟），并与原论文的 Qwen3-32B 结果对照；
2. **采样**：每篇 50 组作者名单，共 455,400 组（约 1 小时）；
3. **选择题**：44,616 道题（约 30–45 分钟）。

**运行前**
- `gts_bundle.zip` 和这个 notebook 都放在 Google Drive 的 `capstone_gts` 文件夹里；
- 菜单「代码执行程序 / Runtime」→「更改运行时类型 / Change runtime type」：选 **A100 GPU**，并打开 **高 RAM / High-RAM**（这样会分到 80 GB 的 A100）；
- 菜单「全部运行 / Run all」，弹窗里允许访问 Google Drive。

**中途断开也没关系**：重新「全部运行」，已经完成的部分会自动跳过。运行期间不要关闭这个标签页，电脑不要休眠。
"""

CELL_SETUP = """# Step 1: mount Google Drive and unpack the bundle
import os, sys, zipfile
from google.colab import drive

drive.mount('/content/drive')
DRIVE_DIR = '/content/drive/MyDrive/capstone_gts'
BUNDLE = f'{DRIVE_DIR}/gts_bundle.zip'
WORK = '/content/gts_bundle'
OUT = f'{DRIVE_DIR}/outputs'

assert os.path.exists(BUNDLE), f'Missing {BUNDLE}: upload gts_bundle.zip to MyDrive/capstone_gts'
if not os.path.exists(WORK):
    zipfile.ZipFile(BUNDLE).extractall(WORK)
sys.path.insert(0, f'{WORK}/src')
os.makedirs(OUT, exist_ok=True)
!nvidia-smi --query-gpu=name,memory.total --format=csv"""

CELL_INSTALL = """# Step 2: install vLLM (a few minutes on a fresh runtime)
import importlib.util, subprocess
wanted = [('vllm', 'vllm'), ('nameparser', 'nameparser==1.1.3'), ('unidecode', 'Unidecode==1.3.8')]
missing = [pkg for module, pkg in wanted if importlib.util.find_spec(module) is None]
if missing:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *missing], check=True)
import vllm
print('vllm', vllm.__version__)"""

CELL_MODEL = """# Step 3: load Qwen3-32B (downloads about 65 GB the first time; 10-15 minutes)
from gts.generate import model_spec, load_llm
spec = model_spec()
print(spec)
if spec['precision'] != 'bf16':
    print('NOTE: this GPU has less than 80 GB, so the FP8 checkpoint is used instead of bf16.')
llm = load_llm(spec)"""

CELL_GREEDY = """# Step 4: default (greedy) answers, checked against the paper's own Qwen3-32B run
from gts.data import load_papers, load_shipped_greedy
from gts.generate import run_open, read_chunks

papers = load_papers().sort_values('paper_id').to_dict('records')
run_open(llm, papers, f'{OUT}/greedy', mode='greedy', n=1, temperature=0.0, chunk_size=2000)

greedy = {r['paper_id']: r['texts'][0].strip() for r in read_chunks(f'{OUT}/greedy')}
shipped = load_shipped_greedy().set_index('paper_id')['generated_text'].fillna('')
same = sum(greedy[p] == shipped[p].strip() for p in greedy)
print(f'{len(greedy)} greedy answers; identical to the paper run: {same / len(greedy):.1%}')
print('answers containing <think>:', sum('<think>' in t for t in greedy.values()))
print('empty answers:', sum(not t for t in greedy.values()))
for pid in list(greedy)[:3]:
    print('  ours :', greedy[pid][:100]); print('  paper:', shipped[pid][:100])"""

CELL_SAMPLES = """# Step 5: 50 sampled author lists per paper (about an hour)
run_open(llm, papers, f'{OUT}/samples', mode='samples', n=50, temperature=1.0, chunk_size=500)

first = read_chunks(f'{OUT}/samples')[:200]
distinct = sum(len(set(r['texts'])) for r in first) / len(first)
print(f'{len(first)} papers checked: on average {distinct:.1f} distinct lists out of 50')"""

CELL_MC = """# Step 6: multiple choice with real-team distractors (30-45 minutes)
import pandas as pd
from gts.mc import read_items
from gts.generate import run_mc

items = read_items(f'{WORK}/data/mc_items/real_team.jsonl.gz')
run_mc(llm, items, f'{OUT}/mc_real_team', chunk_size=5000)

res = pd.DataFrame(read_chunks(f'{OUT}/mc_real_team'))
meta = pd.DataFrame(items)[['item_id', 'paper_id', 'condition', 'correct_label']]
df = meta.merge(res, on='item_id')
df['chosen'] = df['letter_logprobs'].map(lambda d: max(d, key=d.get) if d else None)
df['correct'] = df['chosen'] == df['correct_label']
per_paper = df.groupby(['condition', 'paper_id'])['correct'].sum()
print('accuracy per question (chance 25%):'); print(df.groupby('condition')['correct'].mean().round(3))
print('papers right in >= 3 of 4 orders (chance about 5%):'); print((per_paper >= 3).groupby('condition').mean().round(3))"""

CELL_META = """# Step 7: record what produced these outputs
import json
from gts.generate import run_metadata
meta = run_metadata(spec) | {'n_papers': len(papers), 'n_mc_items': len(items)}
with open(f'{OUT}/run_meta.json', 'w') as f:
    json.dump(meta, f, indent=2)
print(json.dumps(meta, indent=2))
print('All done. Download MyDrive/capstone_gts/outputs.')"""

OUTRO = """## 完成之后

在 Google Drive 里右键 `capstone_gts/outputs` →「下载」，把下载的压缩包解压到 `C:\\Users\\asc\\Capstone-GtS\\data\\outputs\\`，然后告诉 Claude。

如果某一步报错，把报错的最后几行截图或复制给 Claude。
"""


def build_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb.cells = [
        nbf.v4.new_markdown_cell(INTRO),
        *(nbf.v4.new_code_cell(src) for src in
          (CELL_SETUP, CELL_INSTALL, CELL_MODEL, CELL_GREEDY, CELL_SAMPLES, CELL_MC, CELL_META)),
        nbf.v4.new_markdown_cell(OUTRO),
    ]
    nb.metadata = {
        "accelerator": "GPU",
        "colab": {"gpuType": "A100", "machine_shape": "hm", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    }
    return nb


def main():
    COLAB.mkdir(exist_ok=True)
    bundle = COLAB / "gts_bundle.zip"
    with zipfile.ZipFile(bundle, "w", zipfile.ZIP_DEFLATED) as z:
        for path in BUNDLE_FILES:
            z.write(path, path.relative_to(ROOT).as_posix())
    nb_path = COLAB / "01_generate_and_mc.ipynb"
    nbf.write(build_notebook(), nb_path)
    print(f"{bundle} ({bundle.stat().st_size / 1e6:.1f} MB, {len(BUNDLE_FILES)} files)")
    print(nb_path)


if __name__ == "__main__":
    main()
