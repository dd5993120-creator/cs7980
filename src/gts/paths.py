"""Project-relative paths. The Colab bundle keeps the same layout under /content."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RELEASE_DIR = DATA_DIR / "release"
MC_ITEMS_DIR = DATA_DIR / "mc_items"
OUTPUTS_DIR = DATA_DIR / "outputs"
