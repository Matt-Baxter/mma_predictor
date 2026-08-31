#!/usr/bin/env python
"""Inline the model data into web/index.html to make one self-contained page.

`web/index.html` normally fetches `model_data.json` alongside it, which is what
GitHub Pages serves. Some hosts disallow that fetch, so this writes a single
file with the data embedded instead.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

ROOT = Path(__file__).resolve().parent.parent
SOURCE = ROOT / "web" / "index.html"
DATA = ROOT / "web" / "model_data.json"
OUTPUT = ROOT / "artifacts" / "octagon_odds.html"


def main() -> None:
    html = SOURCE.read_text()
    payload = DATA.read_text()
    if "__MODEL_DATA__" not in html:
        raise SystemExit("web/index.html has no __MODEL_DATA__ placeholder")
    # A JSON payload cannot contain "</script>", but guard anyway.
    payload = payload.replace("</", "<\\/")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(html.replace("__MODEL_DATA__", payload))
    print(f"Wrote {OUTPUT} ({OUTPUT.stat().st_size / 1e6:.2f} MB, self-contained)")


if __name__ == "__main__":
    main()
