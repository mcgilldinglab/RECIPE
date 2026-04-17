from __future__ import annotations

from pathlib import Path
import sys


DOCS_ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = DOCS_ROOT.parent
SRC_ROOT = PACKAGE_ROOT / "src"

sys.path.insert(0, str(SRC_ROOT))

project = "RECIPE"
author = "McGill Ding Lab"
release = "0.1.0"

extensions = [
    "myst_parser",
]

templates_path: list[str] = []
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
root_doc = "index"

html_theme = "sphinx_rtd_theme"
html_static_path: list[str] = []

myst_heading_anchors = 3
