"""Runtime benchmarks for STICA simulation and optimisation.

Methodology: wall time with time.perf_counter (monotonic); the first call of every condition (JIT compilation) is
run separately and not recorded; device work is synchronised with block_until_ready() before the clock stops; every
repetition is written to CSV (no filtering), together with a JSON file describing software and hardware.

    python benchmarks/bench.py --impl stica
    python benchmarks/bench.py --impl legacy        # original implementation, for comparison
    python benchmarks/summarize.py benchmarks/results/*.csv

Run each implementation in its own process: importing the legacy code enables jax_enable_x64 globally.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SIM_CONDITIONS = [(32, 200, 2), (64, 500, 2), (100, 1000, 2), (64, 200, 3), (64, 200, 4)]   # (grid, steps, states)
GA_GRIDS = [32, 64, 100]   # n = 2, steps = grid - 1, ic_count = grid, srt_count = grid^2 (as in the legacy notebooks)


def timed(fn, reps):
    fn()                                   # warm-up / compile, not recorded
    out = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        out.append(time.perf_counter() - t0)
    return out


def bench_stica(args, rows):
    import jax
    import jax.numpy as jnp
    import stica

    rng = np.random.default_rng(0)
    if args.suite in ("sim", "all"):
        for G, T, n in SIM_CONDITIONS:
            ic = jnp.asarray(rng.integers(0, n, (G, G)), jnp.int8)
            srt = jnp.asarray(rng.integers(0, n, (T, G, G, stica.num_rules(n))), jnp.int8)
            fn = lambda: stica.simulate(ic, srt, n).block_until_ready()
            rows += [dict(suite="sim", variant="simulate", grid=G, steps=T, states=n, batch=1, units=1, seconds=s)
                     for s in timed(fn, args.reps)]
            ca = stica.STICA(srt, num_states=n, check_values=False)
            fn = lambda: ca.run(ic).block_until_ready()
            rows += [dict(suite="sim", variant="STICA.run (CAX)", grid=G, steps=T, states=n, batch=1, units=1,
                          seconds=s) for s in timed(fn, args.reps)]
            del srt, ca
    if args.suite in ("ga", "all"):
        iters = args.ga_iterations
        for G in GA_GRIDS:
            for batch in (1, 16):
                key = jax.random.PRNGKey(G)
                fn = lambda: stica.run_ga_batch(key, batch, grid_size=G, n=2, iterations=iters, ic_count=G,
                                                srt_count=G * G).best.block_until_ready()
                rows += [dict(suite="ga", variant=f"run_ga_batch x{batch}", grid=G, steps=G - 1, states=2,
                              batch=batch, units=iters * batch, seconds=s) for s in timed(fn, max(3, args.reps // 2))]


def bench_legacy(args, rows):
    sys.path.insert(0, str(ROOT / "legacy" / "STICA-Main" / "NSTICA"))
    from algorithm.genetic import ArbitraryRulesetMutator, Optimizer
    from algorithm.genetic.transition import transition
    from algorithm.objectives import surfacecalc

    rng = np.random.default_rng(0)
    if args.suite in ("sim", "all"):
        for G, T, n in SIM_CONDITIONS:
            R = n * math.comb(7 + n, 8)
            # int8 keeps the host copy small; the legacy code itself converts the SRT to float64
            ic, srt = rng.integers(0, n, (G, G)), rng.integers(0, n, (T, G, G, R), dtype=np.int8)
            fn = lambda: transition(ic, srt, T, n, None).block_until_ready()
            try:
                times = timed(fn, args.reps)
            except Exception as e:   # the float64 SRT can exceed device memory
                rows.append(dict(suite="sim", variant="transition", grid=G, steps=T, states=n, batch=1, units=1,
                                 seconds=float("nan"), note=f"{type(e).__name__}: {str(e)[:120]}"))
                continue
            finally:
                del srt
            rows += [dict(suite="sim", variant="transition", grid=G, steps=T, states=n, batch=1, units=1, seconds=s)
                     for s in times]
    if args.suite in ("ga", "all"):
        for G in GA_GRIDS:
            np.random.seed(G)
            opt = Optimizer(ArbitraryRulesetMutator(grid_size=G, states=2, ic_ct=G, srt_ct=G * G), surfacecalc)
            k = args.legacy_ga_iterations

            def fn():
                for _ in range(k):
                    opt.step()
            rows += [dict(suite="ga", variant="Optimizer.step", grid=G, steps=G - 1, states=2, batch=1, units=k,
                          seconds=s) for s in timed(fn, 3)]


def metadata(impl):
    import jax
    meta = dict(impl=impl, date=dt.datetime.now().isoformat(timespec="seconds"), python=platform.python_version(),
                platform=platform.platform(), jax=jax.__version__, devices=[str(d) for d in jax.devices()],
                device_kind=jax.devices()[0].device_kind)
    from importlib.metadata import PackageNotFoundError, version
    for pkg in ("flax", "cax", "numpy", "stica"):
        try:
            meta[pkg] = version(pkg)
        except PackageNotFoundError:
            pass
    try:
        meta["git_commit"] = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                                            capture_output=True, text=True).stdout.strip()
    except OSError:
        pass
    return meta


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--impl", choices=["stica", "legacy"], default="stica")
    p.add_argument("--suite", choices=["sim", "ga", "all"], default="all")
    p.add_argument("--reps", type=int, default=7)
    p.add_argument("--ga-iterations", type=int, default=200, help="iterations per timed call (stica)")
    p.add_argument("--legacy-ga-iterations", type=int, default=10, help="iterations per timed call (legacy)")
    p.add_argument("--out", type=Path, default=None, help="CSV path (default benchmarks/results/<impl>.csv)")
    args = p.parse_args()

    rows: list[dict] = []
    (bench_stica if args.impl == "stica" else bench_legacy)(args, rows)
    out = args.out or ROOT / "benchmarks" / "results" / f"{args.impl}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["impl", "suite", "variant", "grid", "steps", "states", "batch", "units", "seconds", "note"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({"impl": args.impl, "note": "", **r})
    out.with_suffix(".json").write_text(json.dumps(metadata(args.impl), indent=2) + "\n")
    print(f"wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
