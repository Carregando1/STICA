"""Summarise benchmark CSVs as Markdown tables (median and interquartile range over repetitions).

    python benchmarks/summarize.py benchmarks/results/stica.csv benchmarks/results/legacy.csv
"""
import csv
import sys
from collections import defaultdict

import numpy as np


def load(paths):
    rows = []
    for p in paths:
        with open(p) as f:
            rows += list(csv.DictReader(f))
    return rows


def main(paths):
    groups = defaultdict(list)
    notes = {}
    for r in load(paths):
        key = (r["suite"], int(r["grid"]), int(r["steps"]), int(r["states"]), r["impl"], r["variant"])
        per_unit = float(r["seconds"]) / int(r["units"])
        groups[key].append(per_unit)
        if r.get("note"):
            notes[key] = r["note"]

    for suite, title, unit in (("sim", "Simulation (ms per trajectory)", 1e3),
                               ("ga", "Optimisation (ms per iteration and chain)", 1e3)):
        keys = sorted(k for k in groups if k[0] == suite)
        if not keys:
            continue
        print(f"\n### {title}\n")
        print("| grid | steps | states | implementation | median | IQR | cell updates / s |")
        print("|---:|---:|---:|---|---:|---:|---:|")
        for k in keys:
            _, G, T, n, impl, variant = k
            v = np.array(groups[k]) * unit
            if np.isnan(v).all():
                print(f"| {G} | {T} | {n} | {impl}: {variant} | failed | | {notes.get(k, '')[:60]} |")
                continue
            q1, med, q3 = np.percentile(v, [25, 50, 75])
            cups = G * G * T / (med / unit)
            print(f"| {G} | {T} | {n} | {impl}: {variant} | {med:.3g} | {q1:.3g}-{q3:.3g} | {cups:.2e} |")


if __name__ == "__main__":
    main(sys.argv[1:])
