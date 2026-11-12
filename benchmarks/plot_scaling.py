"""Plot and tabulate the scaling benchmarks written by benchmarks/scaling.py.

    python benchmarks/plot_scaling.py benchmarks/results/scaling_*.csv

Writes benchmarks/results/scaling.png (median per condition; bars = interquartile range over repetitions) and prints
Markdown tables of the medians.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

STYLE = {  # tag: (label, colour, marker)
    "jax-gpu": ("stica, GPU", "#1f5fa8", "o"),
    "jax-cuda": ("stica, GPU", "#1f5fa8", "o"),
    "jax-cpu": ("stica, CPU (XLA)", "#6aa5e0", "s"),
    "numba": ("numba loop, 1 core", "#c4561d", "^"),
    "numba-par": ("numba loop, all cores", "#e08a3c", "<"),
    "numpy": ("NumPy, vectorised", "#7f7f7f", "D"),
    "cellpylib": ("CellPyLib, rule callable", "#9b59b6", "v"),
}


def load(paths):
    rows = []
    for p in paths:
        with open(p) as f:
            rows += list(csv.DictReader(f))
    meta = {}
    for p in paths:
        j = Path(p).with_suffix(".json")
        if j.exists():
            m = json.loads(j.read_text())
            meta[m["tag"]] = m
    return rows, meta


def series(rows, suite, x, y, extra=None):
    """{(tag, extra value): (xs, median, q25, q75)} for one suite."""
    groups = defaultdict(list)
    for r in rows:
        if r["suite"] != suite or not r.get(y):
            continue
        groups[(r["tag"], r[extra] if extra else None, int(r[x]))].append(float(r[y]))
    out = defaultdict(lambda: ([], [], [], []))
    for (tag, e, xv), vals in sorted(groups.items(), key=lambda kv: (kv[0][0], str(kv[0][1]), kv[0][2])):
        xs, med, lo, hi = out[(tag, e)]
        xs.append(xv)
        med.append(np.median(vals))
        lo.append(np.percentile(vals, 25))
        hi.append(np.percentile(vals, 75))
    return out


def main(paths):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows, meta = load(paths)
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), layout="constrained")
    panels = [
        (axes[0, 0], "sim1d", "size", "cell_updates_per_s", None, "lattice width W (T = W steps)",
         "cell updates / s", "a  1D rule arrays, one trajectory"),
        (axes[0, 1], "sim2d", "size", "cell_updates_per_s", "states", "grid size G (G x G cells, T = G steps)",
         "cell updates / s", "b  2D STICA, one trajectory"),
        (axes[1, 0], "sim1d_batch", "batch", "cell_updates_per_s", None, "independent 64 x 64 rule arrays per call",
         "cell updates / s", "c  1D rule arrays, batched"),
        (axes[1, 1], "evo1d", "batch", "candidates_per_s", None, "runs per call (1-core numba: one after another)",
         "candidate rule arrays evaluated / s", "d  Wolfram's lifetime-50 evolution"),
    ]
    for ax, suite, x, y, extra, xlabel, ylabel, title in panels:
        for (tag, e), (xs, med, lo, hi) in series(rows, suite, x, y, extra).items():
            label, colour, marker = STYLE.get(tag, (tag, None, "o"))
            ls = {None: "-", "2": "-", "3": "--", "4": ":"}[e]
            lab = label if e in (None, "2") else None
            ax.errorbar(xs, med, yerr=[np.subtract(med, lo), np.subtract(hi, med)], color=colour, marker=marker,
                        ms=4, lw=1.2, ls=ls, capsize=2, label=lab)
        ax.set(xscale="log", yscale="log", xlabel=xlabel, ylabel=ylabel)
        ax.grid(alpha=0.25, which="both", lw=0.4)
        ax.set_title(title, loc="left", fontsize=10)
    axes[0, 1].plot([], [], "k-", label="n = 2")
    axes[0, 1].plot([], [], "k--", label="n = 3")
    axes[0, 1].plot([], [], "k:", label="n = 4")
    for ax in axes.flat:
        ax.legend(fontsize=7)
    dev = next((m.get("device") for m in meta.values() if m.get("device") and m.get("platform_jax") == "gpu"), None)
    cpu = next((m.get("cpu") for m in meta.values() if m.get("cpu")), None)
    fig.suptitle(f"GPU: {dev or 'n/a'}   CPU: {cpu or 'n/a'}", fontsize=8)
    out = Path(paths[0]).parent / "scaling.png"
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")

    for suite, x, y, extra in [("sim1d", "size", "cell_updates_per_s", None), ("sim2d", "size", "cell_updates_per_s",
                                                                                "states"),
                               ("sim1d_batch", "batch", "cell_updates_per_s", None),
                               ("evo1d", "batch", "candidates_per_s", None)]:
        s = series(rows, suite, x, y, extra)
        if not s:
            continue
        xs = sorted({v for xs_, *_ in s.values() for v in xs_})
        print(f"\n### {suite}: median {y}\n")
        print("| implementation |" + "".join(f" {x}={v} |" for v in xs))
        print("|---|" + "---:|" * len(xs))
        for (tag, e), (xv, med, _, _) in s.items():
            d = dict(zip(xv, med))
            name = STYLE.get(tag, (tag,))[0] + (f", n={e}" if e else "")
            print(f"| {name} |" + "".join(f" {d[v]:.2e} |" if v in d else " — |" for v in xs))


if __name__ == "__main__":
    main(sys.argv[1:])
