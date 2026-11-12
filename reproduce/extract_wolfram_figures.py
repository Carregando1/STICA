"""Read rule arrays and cell states off the figures of Wolfram's essay, for the bit-for-bit test of the simulator.

Wolfram, "What's Really Going On in Machine Learning? Some Minimal Models" (2024), shows rule arrays of elementary
rules 4 and 146 as images: each cell's background colour gives its rule (teal = rule 4, pink = rule 146) and a dark
shade marks a cell in state 1. This script downloads three of those figures, samples every cell at its centre and
writes the rule arrays and states to tests/data/wolfram_rule_arrays.json:

- lifetime50_solutions: the 12 evolved 64 x 31 rule arrays whose patterns survive for exactly 50 steps;
- examples: the 8 rule arrays (15 x 31) of the first rule-array figure, four of them random;
- simple_solutions: 4 of the 11 x 15 hand-built arrays with a single rule-146 cell.

The figure parameters (sizes, rules, the single-cell initial condition, the cyclic lattice) come from the Wolfram
Language code attached to each figure. The images are not stored in this repository; they are downloaded to
outputs/wolfram_figures/. The script checks the result with a direct port of Wolfram's RuleArray code, so a
misread cell would show up as a state mismatch.

Usage: python reproduce/extract_wolfram_figures.py
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import numpy as np
from PIL import Image

ESSAY = "https://writings.stephenwolfram.com/2024/08/whats-really-going-on-in-machine-learning-some-minimal-models/"
BASE = "https://content.wolfram.com/sites/43/2024/08/"
FIGURES = {  # name: (image file, rule-array height, width, description)
    "lifetime50_solutions": ("sw08202024arraysBimg5.png", 64, 31,
                             "evolved rule arrays whose pattern survives exactly 50 steps"),
    "examples": ("sw08202024arraysBimg2.png", 15, 31,
                 "all rule 4, all rule 146, checkerboard, sparse lattice, and four random arrays"),
    "simple_solutions": ("sw08202024arraysBimg7.png", 11, 15,
                         "all rule 4 with one rule-146 cell at row 2i, column 7 (1-indexed), i = 1..4"),
}
RULES = (4, 146)
ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "outputs" / "wolfram_figures"
OUT = ROOT / "tests" / "data" / "wolfram_rule_arrays.json"


def _runs(mask):
    runs, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        if not v and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(mask)))
    return runs


def read_panels(path: Path, height: int, width: int):
    """Split the figure into panels (separated by white space) and sample each cell centre. Each panel shows the
    initial condition row (white with one black cell) above `height` rows of the rule array."""
    im = np.asarray(Image.open(path).convert("RGBA")).astype(int)
    rgb, alpha = im[..., :3], im[..., 3]
    ink = (alpha > 0) & (rgb.sum(-1) < 740)
    panels = []
    for y0, y1 in _runs(ink.any(1)):
        for x0, x1 in _runs(ink[y0:y1].any(0)):
            ys = (y0 + (np.arange(height + 1) + 0.5) * (y1 - y0) / (height + 1)).astype(int)
            xs = (x0 + (np.arange(width) + 0.5) * (x1 - x0) / width).astype(int)
            c = rgb[np.ix_(ys, xs)]
            states = (c.sum(-1) < 450).astype(np.int8)              # dark shade = state 1
            rule_index = (c[..., 0] > c[..., 1]).astype(np.int8)    # pink (rule 146) has R > G, teal has G > R
            panels.append((rule_index[1:], states))
    return panels


def wolfram_rule_array(initial, rule_numbers):
    """Line-by-line port of Wolfram's RuleArray (radius 1, orientation "Both", cyclic lattice)."""
    state = [int(v) for v in initial]
    states = [state]
    width = len(state)
    for row in rule_numbers:
        new = []
        for j in range(width):
            left, center, right = state[j - 1], state[j], state[(j + 1) % width]
            code = (left << 2) | (center << 1) | right
            new.append((int(row[j]) >> code) & 1)
        state = new
        states.append(state)
    return np.array(states)


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    data = {"source": ESSAY, "rules": list(RULES), "radius": 1, "boundary": "periodic",
            "rule_array_values": "index into rules: 0 = rule 4, 1 = rule 146",
            "states": "row 0 is the initial condition; row t + 1 is produced by rule-array row t",
            "figures": {}}
    for name, (file, height, width, description) in FIGURES.items():
        path = CACHE / file
        if not path.exists():
            urllib.request.urlretrieve(BASE + file, path)
        panels = read_panels(path, height, width)
        entries = []
        for ra, states in panels:
            sim = wolfram_rule_array(states[0], np.asarray(RULES)[ra])
            mismatches = int((sim != states).sum())
            if mismatches:
                raise SystemExit(f"{name}: {mismatches} cells differ from Wolfram's RuleArray; extraction failed")
            entries.append({"rule_array": ["".join(map(str, r)) for r in ra],
                            "states": ["".join(map(str, r)) for r in states]})
        data["figures"][name] = {"image": BASE + file, "height": height, "width": width,
                                 "description": description, "panels": entries}
        print(f"{name}: {len(entries)} panels, all cells consistent with Wolfram's RuleArray")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, indent=1) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
