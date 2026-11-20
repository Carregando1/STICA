"""Summarise an equal-time run directory (benchmarks/equal_time.py, reproduce/wolfram_native.wls, scaling.py) into
headline numbers per implementation.

    python benchmarks/equal_time_summary.py RUN_DIR [--out RUN_DIR/summary.json]

Implementations: Wolfram's code as published (interpreted, one kernel), Wolfram's code with RuleArray compiled (one
run per subkernel), the numba baseline (one thread; one run per thread), STICA (one GPU). Rates come from the
budgeted runs where there is one (reach at scale 1, anytime at scale 10, harvest) and from latency otherwise.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import glob
import json
import math
from pathlib import Path

import numpy as np


def load(run: Path) -> dict:
    d = {}
    for p in sorted(run.rglob("*.json")):                  # the run directory and its per-GPU subdirectories
        if p.name == "summary.json" or ".partial" in p.name:
            continue
        try:
            d[p.stem] = json.loads(p.read_text())
        except json.JSONDecodeError:
            print(f"skipping unreadable {p.name}")
    w = {}
    for k in sorted(k for k in d if k.startswith("wolfram") or k.startswith("w_")):
        w.update(d[k])
    d["wolfram"] = w
    return d


def wl_float(x):
    """Mathematica numbers arrive as numbers or strings like '0.116`5.5' or 'Infinity'."""
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).split("`")[0]
    return math.inf if s == "Infinity" else float(s)


def best_at(trace, t, col=2):
    """Best loss so far at time t from a budget_run trace (None while infinite)."""
    v = None
    for row in trace:
        if row[0] > t:
            break
        v = row[col]
    return v


def wl_best_at(per_kernel, t):
    """Best loss so far over all kernels at time t from adaptLoop event lists."""
    best = math.inf
    for k in per_kernel:
        for when, loss in k["events"]:
            if wl_float(when) <= t:
                best = min(best, wl_float(loss))
        if any(wl_float(s[0]) <= t for s in k["solutions"]):
            best = 0.0
    return None if math.isinf(best) else best


def gpu_energy(run: Path, start: float, end: float) -> float | None:
    """Joules drawn by the GPU between two epoch times, from the nvidia-smi power logs (local time stamps, read in the
    time zone of the cluster, America/New_York)."""
    from zoneinfo import ZoneInfo
    tz = ZoneInfo("America/New_York")
    total = None
    for f in run.rglob("power_gpu_*.csv"):
        rows = list(csv.reader(open(f)))
        pts = []
        for r in rows[1:]:
            try:
                t = dt.datetime.strptime(r[0].strip(), "%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=tz).timestamp()
                pts.append((t, float(r[1].split()[0])))
            except (ValueError, IndexError):
                continue
        pts = [(t, p) for t, p in pts if start <= t <= end]
        if len(pts) > 2:
            ts, ps = np.array(pts).T
            total = (total or 0.0) + float(np.trapezoid(ps, ts))
    return total


def census_counts(run: Path) -> np.ndarray | None:
    """Solved census runs by number of mutations to solution (index = mutations), from the census arrays
    (stica_census.npz, about 300 MB) or from the histogram kept in the repository (stica_census_hist.npz)."""
    full = next(run.rglob("stica_census.npz"), None)
    if full is not None:
        m = np.load(full)["mutations"]
        return np.bincount(m[m > 0])
    hist = next(run.rglob("stica_census_hist.npz"), None)
    return None if hist is None else np.load(hist)["mutation_counts"]


def mutation_stats(values, counts, approximate=0) -> dict:
    """Median and tail of the mutations behind each solution, from (value, count) pairs."""
    v, c = np.asarray(values, float), np.asarray(counts, float)
    if c.sum() == 0:
        return dict(solutions=0)
    o = np.argsort(v)
    v, cdf = v[o], np.cumsum(c[o]) / c.sum()
    at = lambda q: float(v[np.searchsorted(cdf, q)])
    le = lambda k: float(c[o][v <= k].sum() / c.sum())
    return dict(solutions=int(c.sum()), median=at(0.5), q10=at(0.1), q90=at(0.9), min=float(v[0]),
                frac_le_1000=le(1000), frac_le_10000=le(10000), approximate_frac=approximate / c.sum())


def hist_stats(h: dict) -> dict:
    """mutation_stats of a budget_run histogram (bins above 4096 at their lower edge)."""
    pairs = h["exact"] + h["above"]
    return mutation_stats([a for a, _ in pairs], [b for _, b in pairs], h.get("approximate", 0))


CHECKPOINTS = [1, 10, 60, 600, 3600]


def summarise(run: Path) -> dict:
    d = load(run)
    w = d["wolfram"]
    S = {"run": str(run)}

    # ---------------------------------------------------------------- Wolfram's size: rates, latency, per hour
    rate = {}
    if "latency" in w:
        for mode in ("interpreted", "compiled"):
            runs = w["latency"][mode]
            sec = sum(wl_float(r["seconds"]) for r in runs)
            rate[f"wolfram_{mode}_1kernel"] = dict(
                solutions_per_s=sum(r["solved"] for r in runs) / sec,
                median_run_s=float(np.median([wl_float(r["seconds"]) for r in runs])), source="latency")
    if "harvest" in w:
        h = w["harvest"]
        rate["wolfram_compiled_8kernels"] = dict(solutions_per_s=len(h["solutions"]) / wl_float(h["seconds"]),
                                                 source="harvest")
    for impl in ("numba", "stica"):
        if f"{impl}_latency" in d:
            for k, v in d[f"{impl}_latency"]["result"].items():
                rate[f"{impl}_latency_{k}"] = dict(solutions_per_s=1 / v["seconds_per_solution"],
                                                   median_run_s=v["median_seconds_solved"], source="latency")
        if f"{impl}_reach" in d:
            for row in d[f"{impl}_reach"]["result"]["rows"]:
                if row["setup"]["scale"] == 1:
                    rate[f"{impl}_batch"] = dict(solutions_per_s=row["solutions_per_s"], lanes=row["lanes"],
                                                 candidates_per_s=row["candidates_per_s"], source="reach scale 1")
    for v in rate.values():
        v["per_hour"] = v["solutions_per_s"] * 3600
    S["wolfram_size_rates"] = rate

    # ---------------------------------------------------------------- first solution
    first = {}
    if "first" in w:
        secs = [wl_float(r["seconds"]) for r in w["first"]["rounds"] if r["seconds"] is not None]
        first[f"wolfram_compiled_{w['first']['kernels']}kernels"] = float(np.median(secs))
    for impl in ("numba", "stica"):
        if f"{impl}_first" in d:
            for row in d[f"{impl}_first"]["result"]["rows"]:
                first[f"{impl}_{row['lanes']}runs"] = dict(median_s=row["median_interpolated_s"],
                                                           median_detected_s=row["median_detected_s"])
    S["first_solution"] = first

    # ---------------------------------------------------------------- redo his experiments
    redo = {}
    if "replay_lifetime50_seconds" in w:
        redo["wolfram_interpreted_8kernels_100runs"] = wl_float(w["replay_lifetime50_seconds"])
    if "sample" in w:
        s = w["sample"]
        redo[f"wolfram_compiled_{s['kernels']}kernels_{s['runs']}runs"] = dict(
            lifetime50=wl_float(s["lifetime50"]["seconds"]), andxor=wl_float(s["andxor"]["seconds"]))
    for impl in ("numba", "stica"):
        if f"{impl}_redo" in d:
            redo[impl] = {k: (float(np.median(v["seconds"])) if "seconds" in v else v.get("note"))
                          for k, v in d[f"{impl}_redo"]["result"].items()}
    S["redo"] = redo

    # ---------------------------------------------------------------- census, harvest: distinct, precision
    if "stica_census" in d:
        c = d["stica_census"]["result"]
        S["census"] = dict(runs=c["runs"], solved=c["solved"], quantiles=c["quantiles"],
                           distinct_curves=c["distinct_curves"])
        counts = census_counts(run)
        if counts is not None:
            cdf = np.cumsum(counts) / counts.sum()
            quantile = lambda p: float(np.searchsorted(cdf, p))             # inverse CDF of mutations to solution
            prec = {}
            for name, v in rate.items():
                n = int(v["per_hour"])
                if n < 2:
                    continue
                lo, hi = (quantile(0.5 + s * 0.98 / math.sqrt(n)) if 0 < 0.5 + s * 0.98 / math.sqrt(n) < 1
                          else np.nan for s in (-1, 1))
                prec[name] = dict(solutions_per_hour=n, median_ci95_halfwidth=float((hi - lo) / 2),
                                  rarest_detectable_p=3 / n, ks_detectable_D=1.36 * math.sqrt(2 / n))
            S["precision_per_hour"] = prec
            S["census_tail"] = {f"P(mutations<={k})": float(cdf[k]) for k in (1, 2, 5, 10, 100)}
    if "harvest" in w:
        sols = w["harvest"]["solutions"]
        ha = [s[2] for s in sols]
        hp = [s[3] for s in sols]
        S["harvest_distinct"] = dict(solutions=len(sols), distinct_arrays=len(set(ha)), distinct_patterns=len(set(hp)),
                                     median_mutations=float(np.median([s[1] for s in sols])))

    # ---------------------------------------------------------------- anytime and reach
    def stica_like(res):
        out = dict(lanes=res.get("lanes"), solutions=res["solutions"], seconds=res["seconds"],
                   first_solution_s=res["first_solution_s"], candidates_per_s=res["candidates_per_s"],
                   best_loss=res["best_loss"], best_at={t: best_at(res["trace"], t) for t in CHECKPOINTS},
                   solutions_at={t: best_at(res["trace"], t, col=1) for t in CHECKPOINTS})
        if "solution_mutations" in res:                   # runs from before 2026-11-20 did not record them
            out["mutations"] = hist_stats(res["solution_mutations"])
        return out

    def wl_like(res):
        pk = res["per_kernel"]
        sol_t = sorted(wl_float(s[0]) for k in pk for s in k["solutions"])
        steps = [int(s[1]) for k in pk for s in k["solutions"]]
        return dict(kernels=res["kernels"], solutions=res["solutions"], seconds=wl_float(res["seconds"]),
                    first_solution_s=wl_float(res["first_solution_s"]), mutations_total=res["mutations"],
                    candidates_per_s=res["mutations"] / wl_float(res["seconds"]),
                    best_at={t: wl_best_at(pk, t) for t in CHECKPOINTS},
                    solutions_at={t: sum(x <= t for x in sol_t) for t in CHECKPOINTS},
                    mutations=mutation_stats(steps, [1] * len(steps)))

    anytime = {}
    for impl in ("numba", "stica"):
        if f"{impl}_anytime" in d:
            r = d[f"{impl}_anytime"]
            anytime[impl] = stica_like(r["result"])
            if impl == "stica":
                anytime[impl]["gpu_joules"] = gpu_energy(run, r["wall_start"], r["wall_end"])
    for mode in ("interpreted", "compiled"):
        if f"anytime_{mode}" in w:
            anytime[f"wolfram_{mode}"] = wl_like(w[f"anytime_{mode}"])
    S["anytime"] = anytime

    reach = {}
    for impl in ("numba", "stica"):
        if f"{impl}_reach" in d:
            reach[impl] = {row["setup"]["scale"]: stica_like(row) for row in d[f"{impl}_reach"]["result"]["rows"]}
    if "reach_compiled" in w:
        reach["wolfram_compiled"] = {r["setup"]["scale"]: wl_like(r) for r in w["reach_compiled"]}
    for v in reach.values():
        for row in v.values():
            row.pop("best_at", None)
            row.pop("solutions_at", None)
    S["reach"] = reach
    if "ladder_timing" in w:
        S["wolfram_ladder_timing"] = {r["setup"]["scale"]: dict(
            interpreted_s=wl_float(r["interpreted_seconds_per_mutation"]),
            compiled_s=wl_float(r["compiled_seconds_per_mutation"])) for r in w["ladder_timing"]}
    hosts = {k: v["metadata"].get("gpu") or v["metadata"].get("cpu") for k, v in d.items() if "metadata" in v}
    hosts["wolfram"] = {k: w[k].get("host", {}).get("cpu") for k in w if isinstance(w[k], dict) and "host" in w[k]}
    S["hardware"] = hosts
    return S


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    run = Path(args.run)
    S = summarise(run)
    out = Path(args.out) if args.out else run / "summary.json"
    out.write_text(json.dumps(S, indent=1, default=lambda o: None))
    print(json.dumps({k: S[k] for k in ("wolfram_size_rates", "first_solution", "anytime") if k in S}, indent=1,
                     default=str)[:4000])
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
