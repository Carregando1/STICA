"""Scaling benchmarks: simulation and evolution throughput of the package and of the CPU baselines.

    python benchmarks/scaling.py --impl jax                               # the package on the default JAX device
    JAX_PLATFORMS=cpu python benchmarks/scaling.py --impl jax --tag jax-cpu
    python benchmarks/scaling.py --impl numba                             # also: numpy, cellpylib
    python benchmarks/plot_scaling.py benchmarks/results/scaling_*.csv

Suites
- sim1d:        Wolfram's rule arrays (rules 4 and 146, radius 1, cyclic), T = W steps, one trajectory.
- sim1d_batch:  64 x 64 rule arrays (Wolfram's size), many independent trajectories per call.
- sim2d:        2D STICA (Moore, n = 2, 3, 4 states, full spacetime rule tensors), T = G steps, one trajectory.
- evo1d:        Wolfram's lifetime-50 evolution (64 x 31, 10,000 mutations per run, ties accepted), in several
                variants: unpacked or bit-packed rule arrays; without restart (each of the B parallel runs uses the
                whole budget, as Wolfram's runs stop at zero loss) or with restart (a run that reaches zero loss is
                counted and starts again); for the GPU, the XLA path or the fused Pallas kernel. numba runs chains
                one after another (numba-par: one per thread) and stops each one at zero loss.
- evo1d_large:  the same at 100x the cells (640 x 311, lifetime 500, 200 mutations per run): candidates per second.

Timing. Each condition is called once (compilation, not recorded), then repeatedly for --warmup seconds so the
processor or GPU reaches steady clocks. Each of --reps repetitions is a loop of enough calls to last at least
--min-rep seconds; its per-call time is recorded. All repetitions go to the CSV. In a series of growing conditions,
once a single call takes longer than --max-call seconds the larger conditions are skipped. CPU baselines are
single-threaded except numba-par (all cores). NumPy's batched suite vectorises over the batch. Conditions slower than 2 s per call get three repetitions.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))

SIM1D_WIDTHS = [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
SIM1D_BATCHES = [1, 8, 64, 512, 4096, 32768]
SIM2D = [(G, n) for n in (2, 3, 4) for G in (16, 32, 64, 128, 256)]
SRT_LIMIT_BYTES = 1.6e9
EVO_BATCHES = [1, 64, 1024, 8192, 32768]
EVO = dict(height=64, width=31, target=50, iterations=10_000)
EVO_LARGE = dict(height=640, width=311, target=500, iterations=200)
EVO_LARGE_BATCHES = [1, 64, 512, 2048, 8192, 32768]


def measure(fn, args):
    """Per-call seconds for each repetition (see the module docstring)."""
    t0 = time.perf_counter()
    fn()
    first = time.perf_counter() - t0
    if first > args.max_call:
        return [first], first, "single call (exceeds --max-call): includes compilation"
    end = time.perf_counter() + args.warmup
    while time.perf_counter() < end:
        fn()
    t0 = time.perf_counter()
    fn()
    one = time.perf_counter() - t0
    calls = max(1, int(np.ceil(args.min_rep / max(one, 1e-7))))
    times = []
    for _ in range(args.reps if one < 2.0 else 3):          # slow conditions: three repetitions
        t0 = time.perf_counter()
        for _ in range(calls):
            fn()
        times.append((time.perf_counter() - t0) / calls)
    return times, one, f"{calls} calls per repetition"


# ----------------------------------------------------------------------------------------------- implementations

def make_impl(name):
    """Functions sim1d(ic, ra, tables), sim1d_batch(ics, ras, tables), sim2d(ic, srt, n), evo1d(batch, seed)."""
    import stica
    tables = np.asarray(stica.rule_tables([4, 146]))
    evo_ic, evo_ra0 = np.asarray(stica.single_cell(EVO["width"])), np.zeros((EVO["height"], EVO["width"]), np.int8)
    if name == "jax":
        import jax
        import jax.numpy as jnp

        put = jax.device_put
        sim1d_batch = jax.jit(jax.vmap(lambda ic, ra, tb: stica.rule_array_rollout(ic, ra, tb), (0, 0, None)))

        def jax_evo(variant):
            def fn(batch, seed, setup):
                ic, ra0 = stica.single_cell(setup["width"]), np.zeros((setup["height"], setup["width"]), np.int8)
                kw = dict(objective=stica.LifetimeLoss(setup["target"]), iterations=setup["iterations"], stop_at=0.0)
                key = jax.random.PRNGKey(seed)
                if variant.startswith("unpacked"):
                    res = stica.evolve_rule_array_batch(key, batch, ic, ra0, [4, 146], restart="restart" in variant,
                                                        **kw)
                else:
                    res = stica.evolve_packed_batch(key, batch, ic, ra0, [4, 146], restart="restart" in variant,
                                                    backend="pallas" if "fused" in variant else "xla",
                                                    keep_last_solution=False, **kw)
                return batch * setup["iterations"], int(np.asarray(jax.block_until_ready(res.solves)).sum())
            return fn

        variants = ["unpacked", "unpacked, restart", "packed, restart"]
        if jax.default_backend() == "gpu":
            variants.append("packed, restart, fused kernel")
        evo = {v: jax_evo(v) for v in variants}

        return dict(
            prepare=lambda *a: tuple(put(jnp.asarray(x)) for x in a),
            sim1d=lambda ic, ra, tb: stica.simulate_rule_array(ic, ra, tb).block_until_ready(),
            sim1d_batch=lambda ics, ras, tb: sim1d_batch(ics, ras, tb).block_until_ready(),
            sim2d=lambda ic, srt, n: stica.simulate(ic, srt, n).block_until_ready(),
            evo=evo, tables=tables)
    import baselines as B
    if name in ("numpy", "numba", "numba-par", "cellpylib"):
        sim1d = {"numpy": B.rule_array_numpy, "numba": B.rule_array_numba, "numba-par": B.rule_array_numba_par,
                 "cellpylib": B.rule_array_cellpylib}[name]
        sim2d = {"numpy": B.stica_numpy, "numba": B.stica_numba, "numba-par": B.stica_numba_par,
                 "cellpylib": B.stica_cellpylib}[name]

        def sim1d_batch(ics, ras, tb):
            for ic, ra in zip(ics, ras):
                sim1d(ic, ra, tb)

        if name == "numpy":
            sim1d_batch = B.rule_array_numpy_batch          # vectorised over the batch as well
        if name == "numba-par":
            sim1d_batch = B.rule_array_numba_batch_par      # one trajectory per thread

        evo = {}
        if name in ("numba", "numba-par"):
            def cpu_evo(packed, parallel):
                def fn(batch, seed, setup):
                    ic = np.asarray(stica.single_cell(setup["width"]))
                    ra0 = np.zeros((setup["height"], setup["width"]), np.int8)
                    args = (ic, ra0, tables, setup["target"], setup["iterations"])
                    seeds = seed * 100_003 + np.arange(batch)
                    if parallel:
                        f = B.evolve_lifetime_packed_par if packed else B.evolve_lifetime_numba_par
                        best, used = f(*args, seeds)
                        return int(used.sum()), int((best == 0).sum())
                    used = solved = 0
                    for sd in seeds:
                        f = B.evolve_lifetime_packed if packed else B.evolve_lifetime_numba
                        _, best, it = f(*args, int(sd))
                        used, solved = used + it, solved + (best == 0)
                    return used, solved
                return fn
            par = name == "numba-par"
            evo = {"unpacked": cpu_evo(False, par), "packed": cpu_evo(True, par)}

        return dict(prepare=lambda *a: a, sim1d=sim1d, sim1d_batch=sim1d_batch, sim2d=sim2d, evo=evo,
                    tables=tables)
    raise SystemExit(f"unknown implementation {name!r}")


# --------------------------------------------------------------------------------------------------------- suites

def run_suites(args, impl, rows):
    import stica
    rng = np.random.default_rng(0)
    tables = impl["tables"]

    def record(suite, size, steps, states, batch, work, times, one, note):
        for s in times:
            rows.append(dict(suite=suite, variant="", size=size, steps=steps, states=states, batch=batch, seconds=s,
                             cell_updates=work, cell_updates_per_s=work / s, note=note))
        print(f"  {suite:12s} size={size:5d} steps={steps:5d} n={states} batch={batch:6d}: "
              f"{np.median(times) * 1e3:10.3f} ms/call  {work / np.median(times):9.3e} cell updates/s  ({note})",
              flush=True)
        return one

    if "sim1d" in args.suites and args.impl != "numba-par":   # threads over the cells of one 1D step only add
        for W in SIM1D_WIDTHS:                                 # overhead; one core is the CPU reference there
            ic = (rng.random(W) < 0.5).astype(np.int8)
            ra = rng.integers(0, 2, (W, W)).astype(np.int8)
            a = impl["prepare"](ic, ra, tables)
            times, one, note = measure(lambda: impl["sim1d"](*a), args)
            if record("sim1d", W, W, 2, 1, W * W, times, one, note) > args.max_call:
                break
    if "sim1d_batch" in args.suites:
        for Bn in SIM1D_BATCHES:
            ics = (rng.random((Bn, 64)) < 0.5).astype(np.int8)
            ras = rng.integers(0, 2, (Bn, 64, 64)).astype(np.int8)
            a = impl["prepare"](ics, ras, tables)
            times, one, note = measure(lambda: impl["sim1d_batch"](*a), args)
            if record("sim1d_batch", 64, 64, 2, Bn, Bn * 64 * 64, times, one, note) > args.max_call:
                break
    if "sim2d" in args.suites:
        for n in (2, 3, 4):
            for G in sorted({g for g, m in SIM2D if m == n}):
                R = stica.num_rules(n)
                if G ** 3 * R > SRT_LIMIT_BYTES:
                    continue
                ic = rng.integers(0, n, (G, G)).astype(np.int8)
                srt = rng.integers(0, n, (G, G, G, R), dtype=np.int8)
                a = impl["prepare"](ic, srt)
                times, one, note = measure(lambda: impl["sim2d"](*a, n), args)
                del a, srt
                if record("sim2d", G, G, n, 1, G ** 3, times, one, note) > args.max_call:
                    break
    on_cpu = args.impl != "jax" or args.tag.endswith("cpu")
    limit = args.cpu_evo_runs * (128 if args.impl == "numba-par" else 1)
    for suite, setup, batches in (("evo1d", EVO, EVO_BATCHES), ("evo1d_large", EVO_LARGE, EVO_LARGE_BATCHES)):
        if suite not in args.suites:
            continue
        work = setup["height"] * setup["width"]
        for variant, fn in impl["evo"].items():
            # device memory: unpacked runs keep a (T, W) byte array and trajectory per run
            cap = 2048 if variant.startswith("unpacked") else 8192 if "fused" not in variant else batches[-1]
            for Bn in batches:
                if (on_cpu and Bn > limit) or (suite == "evo1d_large" and args.impl == "jax" and Bn > cap):
                    break
                if args.impl == "numba-par" and Bn < 64:
                    continue
                fn(Bn, 10**6, setup)                                     # compile / warm up
                reps = []
                for r in range(max(3, args.reps // 2)):
                    t0 = time.perf_counter()
                    candidates, solved = fn(Bn, r, setup)
                    sec = time.perf_counter() - t0
                    reps.append(sec)
                    rows.append(dict(suite=suite, variant=variant, size=setup["width"], steps=setup["height"],
                                     states=2, batch=Bn, seconds=sec, cell_updates=candidates * work,
                                     cell_updates_per_s=candidates * work / sec, candidates=candidates,
                                     candidates_per_s=candidates / sec, solved=solved, solved_per_s=solved / sec,
                                     note="one call = batch runs"))
                print(f"  {suite:12s} {variant:30s} runs={Bn:6d}: {np.median(reps):8.3f} s/call, "
                      f"{rows[-1]['candidates_per_s']:.3e} candidates/s, {rows[-1]['solved_per_s']:.1f} solved/s",
                      flush=True)
                if np.median(reps) > args.max_call:
                    break


def metadata(args):
    meta = dict(impl=args.impl, tag=args.tag, date=dt.datetime.now().isoformat(timespec="seconds"),
                python=platform.python_version(), platform=platform.platform(), argv=sys.argv[1:],
                timing=dict(warmup=args.warmup, min_rep=args.min_rep, reps=args.reps, max_call=args.max_call))
    try:
        cpu = [l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")]
        meta.update(cpu=cpu[0], logical_cpus=len(cpu))
    except OSError:
        meta["cpu"] = platform.processor()
    from importlib.metadata import PackageNotFoundError, version
    if args.impl == "numba-par":
        import numba
        meta["numba_threads"] = numba.get_num_threads()
    for pkg in ("jax", "jaxlib", "numpy", "numba", "cellpylib", "stica"):
        try:
            meta[pkg] = version(pkg)
        except PackageNotFoundError:
            pass
    if args.impl == "jax":
        import jax
        meta.update(device=jax.devices()[0].device_kind, platform_jax=jax.devices()[0].platform)
    meta["env"] = {k: os.environ[k] for k in ("JAX_PLATFORMS", "XLA_FLAGS", "OMP_NUM_THREADS") if k in os.environ}
    try:
        meta["git_commit"] = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                                            capture_output=True, text=True).stdout.strip()
    except OSError:
        pass
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--impl", required=True, choices=["jax", "numpy", "numba", "numba-par", "cellpylib"])
    ap.add_argument("--tag", default=None, help="label for the results (default: impl, or jax-<device platform>)")
    ap.add_argument("--suites", default="sim1d,sim1d_batch,sim2d,evo1d,evo1d_large")
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--warmup", type=float, default=1.0)
    ap.add_argument("--min-rep", type=float, default=0.2)
    ap.add_argument("--max-call", type=float, default=20.0)
    ap.add_argument("--cpu-evo-runs", type=int, default=64, help="largest number of sequential runs for CPU evo1d")
    ap.add_argument("--out", default=str(ROOT / "benchmarks" / "results"))
    ap.add_argument("--merge", action="store_true", help="keep earlier results of the suites not run now")
    args = ap.parse_args()
    args.suites = args.suites.split(",")
    if args.impl == "jax" and args.tag is None:
        import jax
        args.tag = f"jax-{jax.devices()[0].platform}"
    args.tag = args.tag or args.impl
    print(f"implementation {args.impl} ({args.tag})", flush=True)
    impl = make_impl(args.impl)
    rows = []
    run_suites(args, impl, rows)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    keys = ["impl", "tag", "suite", "variant", "size", "steps", "states", "batch", "seconds", "cell_updates",
            "cell_updates_per_s", "candidates", "candidates_per_s", "solved", "solved_per_s", "note"]
    path = out / f"scaling_{args.tag}.csv"
    kept = []
    if args.merge and path.exists():                      # keep the earlier results of the suites not rerun
        with open(path) as f:
            kept = [r for r in csv.DictReader(f) if r["suite"] not in args.suites]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys, restval="")
        w.writeheader()
        for r in kept:
            w.writerow({k: r.get(k, "") for k in keys})
        for r in rows:
            w.writerow(dict(impl=args.impl, tag=args.tag, **r))
    (out / f"scaling_{args.tag}.json").write_text(json.dumps(metadata(args), indent=1) + "\n")
    print(f"wrote {out}/scaling_{args.tag}.csv and .json")


if __name__ == "__main__":
    main()
