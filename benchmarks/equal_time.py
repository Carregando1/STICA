"""Equal-time metrics: what implementations of Wolfram's adaptive evolution achieve in the same wall-clock time.

    python benchmarks/equal_time.py METRIC --impl stica|numba [--out DIR] [options]

The task is Wolfram's lifetime evolution with rules 4 and 146 (radius 1, cyclic): a rule array of height T and width W
starts as all rule 4 under a single live cell, each mutation switches one random cell to the other rule and is kept
unless the loss |lifetime - target| rises (infinite while the pattern outlives the array). "stica" is the package on
the default JAX device (the fused kernel on a GPU); "numba" is the bit-packed CPU baseline of benchmarks/baselines.py,
one run per thread (NUMBA_NUM_THREADS). Wolfram's own code is measured by reproduce/wolfram_native.wls.

Metrics (each writes <out>/<impl>_<metric>.json; census also writes a .npz)
  latency   one run at a time at Wolfram's size (64 x 31, lifetime 50): seconds per run, until its solution
  first     B runs at once at Wolfram's size: seconds until the first of them reaches the target
  redo      Wolfram's experiments rerun: his 100 runs, and 10,000 runs of each of his two experiments
  census    many independent runs at Wolfram's size (no restart): mutations to solution of every run and hashes of every
            solution's rule array and pattern (for tails, precision of estimates and distinct solutions)
  anytime   a fixed budget (default one hour) at 640 x 311, lifetime 500, every run restarted after a solution: best
            loss so far and solutions so far against wall-clock time
  reach     a ladder of sizes (scale s: height 64 s, width 31 s + 1, target 50 s), a fixed budget each: solutions, time
            to the first, best loss

Timing excludes compilation, which is reported separately. Runs split into chunks (STICA: calls continued with
`state=`; numba: a parallel loop over persistent chains); a solution's time inside a chunk is interpolated from the
iteration at which it occurred. For anytime and reach STICA's batch size is chosen by a short calibration at each size
(as in the throughput tables); numba runs one chain per thread. Every result records the host, CPU, GPU, allocated
cores, wall-clock start and end (to integrate a power log) and, where readable, RAPL energy counters.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import math
import os
import platform
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "benchmarks"))

RULES = (4, 146)
BUDGET_MUTATIONS = 10_000                                   # Wolfram's per-run budget
LADDER = [1, 2, 4, 6, 8, 10, 14, 20]


def scaled(s: int) -> dict:
    """Wolfram's lifetime-50 task scaled by s in both directions (s = 10: 640 x 311, lifetime 500)."""
    return dict(height=64 * s, width=31 if s == 1 else 31 * s + 1, target=50 * s, scale=s)


WOLFRAM = scaled(1)


def _num(x):
    """JSON-safe float (infinite losses -> None)."""
    x = float(x)
    return None if not math.isfinite(x) else x


# ------------------------------------------------------------------------------------------------- environment

def _cpu_model() -> str:
    try:
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor()


def _gpu() -> str | None:
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,power.limit", "--format=csv,noheader"],
                              capture_output=True, text=True, timeout=20).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        return None


def energy() -> dict:
    """RAPL energy counters (microjoules) that this user can read, keyed by domain name."""
    out = {}
    for path in sorted(glob.glob("/sys/class/powercap/intel-rapl:*/energy_uj")
                       + glob.glob("/sys/class/powercap/intel-rapl:*:*/energy_uj")):
        try:
            name = Path(path).parent.joinpath("name").read_text().strip()
            out[f"{Path(path).parent.name}:{name}"] = int(Path(path).read_text())
        except OSError:
            pass
    return out


def metadata(args) -> dict:
    md = dict(date=dt.datetime.now().isoformat(timespec="seconds"), host=socket.gethostname(),
              python=platform.python_version(), impl=args.impl, cpu=_cpu_model(),
              cpus_allocated=len(os.sched_getaffinity(0)), slurm_job=os.environ.get("SLURM_JOB_ID"),
              commit=subprocess.run(["git", "-C", str(ROOT), "describe", "--always", "--dirty"], capture_output=True,
                                    text=True).stdout.strip() or os.environ.get("STICA_COMMIT"))
    if args.impl == "stica":
        import jax
        md.update(jax=jax.__version__, backend=jax.default_backend(), devices=[str(d) for d in jax.devices()],
                  gpu=_gpu())
    else:
        import numba
        md.update(numba=numba.__version__, threads=numba.get_num_threads())
    return md


# ---------------------------------------------------------------------------------------------------- runners

class SticaRunner:
    """`lanes` runs of the package's packed evolution, advanced `chunk` mutations per call (continued with state=)."""

    def __init__(self, setup: dict, lanes: int, seed: int, chunk: int, restart: bool = True,
                 backend: str = "pallas"):
        import jax
        import stica
        self.jax, self.stica = jax, stica
        self.lanes, self.iters = lanes, chunk
        self.ic = stica.single_cell(setup["width"])
        self.ra0 = np.zeros((setup["height"], setup["width"]), np.int8)
        self.rules = list(setup.get("rules", RULES))
        self.kw = dict(objective=stica.LifetimeLoss(setup["target"]), stop_at=0.0, restart=restart, backend=backend,
                       radius=setup.get("radius", 1), keep_last_solution=False, unpack=False)
        self.key = jax.random.PRNGKey(seed)
        self.state = None

    def chunk(self) -> dict:
        self.key, k = self.jax.random.split(self.key)
        t0 = time.perf_counter()
        res = self.stica.evolve_packed_batch(k, self.lanes, self.ic, self.ra0, self.rules, iterations=self.iters,
                                             state=self.state, **self.kw)
        solves, first, best = (np.asarray(x) for x in (res.solves, res.solved_at, res.best))
        sec = time.perf_counter() - t0
        self.state = res.state
        hit = first[first > 0]
        return dict(seconds=sec, iterations=self.iters, solves=int(solves.sum()),
                    first_iter=int(hit.min()) if hit.size else -1, best=best,
                    lane_solves=solves, lane_solve_steps=np.asarray(res.solve_steps))


def _numba_chunk_fn():
    from numba import njit, prange
    import baselines as B
    loss = B._packed()["lifetime_loss"]
    one = np.uint64(1)

    @njit(parallel=True)
    def run(ic, masks, best, steps, masks0, best0, t0m, t1m, W, target, iters, seeds, restart, solves, first,
            solve_steps):
        L, T = masks.shape[0], masks.shape[1]
        for lane in prange(L):
            np.random.seed(seeds[lane])
            b, st, ns, fs, ss = best[lane], steps[lane], 0, -1, 0
            for it in range(iters):
                if b == 0.0 and not restart:
                    break
                t, x = np.random.randint(T), np.random.randint(W)
                bit = one << np.uint64(x % 64)
                masks[lane, t, x // 64] ^= bit
                v = loss(ic, masks[lane], t0m, t1m, W, target)
                st += 1
                if v <= b:
                    b = v
                else:
                    masks[lane, t, x // 64] ^= bit
                if b == 0.0:
                    if fs < 0:
                        fs = it + 1
                    if restart:
                        ns += 1
                        ss += st
                        masks[lane, :, :] = masks0
                        b, st = best0, 0
            best[lane], steps[lane], solves[lane], first[lane], solve_steps[lane] = b, st, ns, fs, ss
    return run


class NumbaRunner:
    """One chain per lane (lanes = threads by default), advanced `chunk` mutations per parallel call."""
    _fn = None

    def __init__(self, setup: dict, lanes: int, seed: int, chunk: int, restart: bool = True):
        import baselines as B
        if NumbaRunner._fn is None:
            NumbaRunner._fn = _numba_chunk_fn()
        W, T = setup["width"], setup["height"]
        ic = np.zeros(W, np.int8)
        ic[(W - 1) // 2] = 1
        self.ic, masks0, self.t0m, self.t1m, self.W = B._packed_args(ic, np.zeros((T, W), np.int8), _tables())
        self.masks0 = masks0
        self.target = float(setup["target"])
        self.best0 = float(B._packed()["lifetime_loss"](self.ic, masks0, self.t0m, self.t1m, self.W, self.target))
        self.masks = np.repeat(masks0[None], lanes, 0)
        self.best = np.full(lanes, self.best0)
        self.steps = np.zeros(lanes, np.int64)
        self.solves, self.first = np.zeros(lanes, np.int64), np.zeros(lanes, np.int64)
        self.solve_steps = np.zeros(lanes, np.int64)
        self.lanes, self.iters, self.restart = lanes, chunk, restart
        self.rng = np.random.default_rng(seed)

    def chunk(self) -> dict:
        seeds = self.rng.integers(0, 2 ** 32 - 1, self.lanes, dtype=np.int64)
        t0 = time.perf_counter()
        NumbaRunner._fn(self.ic, self.masks, self.best, self.steps, self.masks0, self.best0, self.t0m, self.t1m,
                        self.W, self.target, self.iters, seeds, self.restart, self.solves, self.first,
                        self.solve_steps)
        sec = time.perf_counter() - t0
        hit = self.first[self.first > 0]
        return dict(seconds=sec, iterations=self.iters, solves=int(self.solves.sum()),
                    first_iter=int(hit.min()) if hit.size else -1, best=self.best.copy(),
                    lane_solves=self.solves.copy(), lane_solve_steps=self.solve_steps.copy())


def _tables():
    import stica
    return np.asarray(stica.rule_tables(list(RULES)))


def make_runner(args, setup, lanes, seed, chunk, restart=True):
    if args.impl == "stica":
        return SticaRunner(setup, lanes, seed, chunk, restart)
    return NumbaRunner(setup, lanes, seed, chunk, restart)


def default_lanes(args) -> int:
    if args.impl == "numba":
        import numba
        return numba.get_num_threads()
    return 32768


def sized_chunk(runner_factory, target_s: float, probe: int = 32) -> tuple[int, float]:
    """A chunk length whose call lasts about target_s, from a probe call (after a compile call). Returns the length
    and the probe's compile + first-call seconds."""
    t0 = time.perf_counter()
    r = runner_factory(probe)
    r.chunk()
    r.chunk()                                         # the second call continues from a state (its own trace)
    compile_s = time.perf_counter() - t0
    secs = [r.chunk()["seconds"] for _ in range(3)]
    per_iter = float(np.median(secs)) / probe
    n = max(probe, int(target_s / max(per_iter, 1e-9)))
    return int(2 ** round(math.log2(n))), compile_s


def calibrate_lanes(args, setup, candidates, seconds_each=3.0) -> dict:
    """Best batch size (candidates evaluated per second) at this size, each tried for a few seconds after compiling."""
    rows = []
    T, W = setup["height"], setup["width"]
    for lanes in candidates:
        if T * -(-W // 32) * lanes * 4 > args.max_batch_bytes:
            continue
        chunk, compile_s = sized_chunk(lambda c: make_runner(args, setup, lanes, 1, c), 0.25)
        r = make_runner(args, setup, lanes, 2, chunk)
        r.chunk()
        r.chunk()
        n = sec = 0
        while sec < seconds_each:
            c = r.chunk()
            n, sec = n + c["iterations"] * lanes, sec + c["seconds"]
        rows.append(dict(lanes=lanes, chunk=chunk, candidates_per_s=n / sec, compile_s=compile_s))
        print(f"    calibrate {T}x{W}: {lanes:6d} lanes  {n / sec:.3e} candidates/s", flush=True)
    best = max(rows, key=lambda r: r["candidates_per_s"])
    return dict(best=best, tried=rows)


class MutationHistogram:
    """Mutations each solution took, binned: exactly below 4096, 16 bins per octave above. A lane that solves once in
    a chunk gives its solution's count exactly (a run may span chunks); a lane that solves k > 1 times gives only the
    total, so those k solutions get the mean (all but the first lie within the chunk) and are counted as approximate."""
    EXACT = 4096

    def __init__(self):
        self.exact = np.zeros(self.EXACT, np.int64)
        self.octaves = {}
        self.approximate = 0

    def add(self, lane_solves, lane_steps):
        k, steps = np.asarray(lane_solves), np.asarray(lane_steps, np.float64)
        one = k == 1
        vals = [steps[one], np.repeat(steps[k > 1] / k[k > 1], k[k > 1])]
        self.approximate += int(k[k > 1].sum())
        v = np.rint(np.concatenate(vals)).astype(np.int64)
        small = v[v < self.EXACT]
        self.exact += np.bincount(small, minlength=self.EXACT)[:self.EXACT]
        for b in np.floor(16 * np.log2(v[v >= self.EXACT])).astype(int):
            self.octaves[int(b)] = self.octaves.get(int(b), 0) + 1

    def to_json(self) -> dict:
        nz = np.flatnonzero(self.exact)
        return dict(exact=[[int(i), int(self.exact[i])] for i in nz],
                    above=[[round(2 ** (b / 16)), n] for b, n in sorted(self.octaves.items())],
                    approximate=self.approximate,
                    note="exact: [mutations, solutions]; above: [lower bin edge, solutions], 16 bins per octave")


def budget_run(runner, budget: float) -> dict:
    """Advance the runner for `budget` seconds of evolution (compilation excluded): solutions and best loss vs time,
    and the mutations each solution took."""
    trace, t, solutions, candidates = [], 0.0, 0, 0
    best_ever, first_time = math.inf, None
    hist = MutationHistogram()
    while t < budget:
        c = runner.chunk()
        hist.add(c["lane_solves"], c["lane_solve_steps"])
        if first_time is None and c["first_iter"] > 0:
            first_time = t + c["seconds"] * c["first_iter"] / c["iterations"]
        t += c["seconds"]
        solutions += c["solves"]
        candidates += c["iterations"] * runner.lanes
        cur = c["best"]
        best_ever = min(best_ever, 0.0 if c["solves"] else float(cur.min()))
        trace.append([round(t, 4), solutions, _num(best_ever), _num(cur.min()), _num(np.median(cur))])
    return dict(seconds=t, solutions=solutions, candidates=candidates, candidates_per_s=candidates / t,
                solutions_per_s=solutions / t, first_solution_s=first_time, best_loss=_num(best_ever),
                trace_columns=["seconds", "solutions", "best_loss_so_far", "lane_min_loss", "lane_median_loss"],
                trace=trace, solution_mutations=hist.to_json())


# ---------------------------------------------------------------------------------------------------- metrics

def metric_latency(args) -> dict:
    """One run at a time at Wolfram's size, each until its solution (or the 10,000-mutation budget)."""
    out = {}
    if args.impl == "stica":
        for backend, chunk in (("pallas", args.latency_chunk), ("xla", args.latency_chunk)):
            def one_run(seed):
                r = SticaRunner(WOLFRAM, 1, seed, chunk, restart=False, backend=backend)
                done = sec = 0
                while done < BUDGET_MUTATIONS:
                    c = r.chunk()
                    sec += c["seconds"]
                    if c["first_iter"] > 0:
                        return dict(seconds=sec, mutations=done + c["first_iter"], solved=True)
                    done += c["iterations"]
                return dict(seconds=sec, mutations=done, solved=False)
            t0 = time.perf_counter()
            one_run(10 ** 6)
            compile_s = time.perf_counter() - t0
            runs = [one_run(s) for s in range(args.runs)]
            out[backend] = _latency_summary(runs, dict(chunk=chunk, compile_and_first_run_s=compile_s))
    else:
        import baselines as B
        ic = np.zeros(31, np.int8)
        ic[15] = 1
        ra0, tables = np.zeros((64, 31), np.int8), _tables()
        B.evolve_lifetime_packed(ic, ra0, tables, 50, 100, 0)                 # compile
        runs = []
        for s in range(args.runs):
            t0 = time.perf_counter()
            _, best, used = B.evolve_lifetime_packed(ic, ra0, tables, 50, BUDGET_MUTATIONS, s)
            runs.append(dict(seconds=time.perf_counter() - t0, mutations=int(used), solved=bool(best == 0)))
        out["numba, 1 thread"] = _latency_summary(runs, {})
    return out


def _latency_summary(runs, extra) -> dict:
    sec = np.array([r["seconds"] for r in runs])
    solved = np.array([r["solved"] for r in runs])
    muts = np.array([r["mutations"] for r in runs])
    print(f"    latency: median {np.median(sec[solved]) * 1e3:.2f} ms per solved run, "
          f"{solved.mean():.0%} solved, {sec.sum() / max(solved.sum(), 1) * 1e3:.2f} ms per solution", flush=True)
    return dict(extra, runs=len(runs), solved=int(solved.sum()), median_seconds_solved=float(np.median(sec[solved])),
                mean_seconds=float(sec.mean()), seconds_per_solution=float(sec.sum() / max(solved.sum(), 1)),
                median_mutations_solved=float(np.median(muts[solved])), seconds=sec.tolist(), mutations=muts.tolist(),
                solved_flags=solved.tolist())


def metric_first(args) -> dict:
    """B runs started together: seconds until the first reaches the target (interpolated inside the chunk) and until
    the host sees it (end of that chunk)."""
    out = []
    lane_list = [int(x) for x in args.lanes.split(",")] if args.lanes else (
        [128, 1024, 8192, 32768] if args.impl == "stica" else [default_lanes(args)])
    for lanes in lane_list:
        chunk = args.first_chunk if args.impl == "stica" else 256
        make = lambda seed: make_runner(args, WOLFRAM, lanes, seed, chunk)
        t0 = time.perf_counter()
        warm = make(10 ** 6)
        warm.chunk()
        warm.chunk()
        compile_s = time.perf_counter() - t0
        reps = []
        for rep in range(args.runs):
            r = make(rep)
            sec = done = 0
            while done < BUDGET_MUTATIONS:
                c = r.chunk()
                if c["first_iter"] > 0:
                    reps.append(dict(interpolated=sec + c["seconds"] * c["first_iter"] / c["iterations"],
                                     detected=sec + c["seconds"], mutations=done + c["first_iter"]))
                    break
                sec, done = sec + c["seconds"], done + c["iterations"]
        it = np.array([r["interpolated"] for r in reps])
        print(f"    first of {lanes} runs: median {np.median(it) * 1e3:.2f} ms "
              f"({np.median([r['mutations'] for r in reps]):.0f} mutations)", flush=True)
        out.append(dict(lanes=lanes, chunk=chunk, compile_s=compile_s, repeats=len(reps),
                        median_interpolated_s=float(np.median(it)),
                        median_detected_s=float(np.median([r["detected"] for r in reps])), runs=reps))
    return dict(rows=out)


def metric_redo(args) -> dict:
    """Wolfram's experiments rerun with his budget (10,000 mutations, stop at the target)."""
    out = {}
    jobs = [("lifetime50_his_100_runs", WOLFRAM, 100), ("lifetime50_10000_runs", WOLFRAM, 10_000),
            ("andxor_10000_runs", dict(height=40, width=21, target=30, rules=(8, 6), radius=0.5), 10_000)]
    for name, setup, runs in jobs:
        if args.impl == "stica":
            import jax
            import stica
            kw = dict(objective=stica.LifetimeLoss(setup["target"]), iterations=BUDGET_MUTATIONS, stop_at=0.0,
                      radius=setup.get("radius", 1), backend="pallas", unpack=False)
            call = lambda seed: stica.evolve_packed_batch(
                jax.random.PRNGKey(seed), runs, stica.single_cell(setup["width"]),
                np.zeros((setup["height"], setup["width"]), np.int8), list(setup.get("rules", RULES)), **kw)
            t0 = time.perf_counter()
            np.asarray(call(10 ** 6).solves)
            compile_s = time.perf_counter() - t0
            secs = []
            for rep in range(3):
                t0 = time.perf_counter()
                res = call(rep)
                first = np.asarray(res.solved_at)
                secs.append(time.perf_counter() - t0)
            out[name] = dict(runs=runs, seconds=secs, compile_and_first_s=compile_s, solved=int((first > 0).sum()),
                             median_mutations=float(np.median(first[first > 0])))
        else:
            if setup.get("radius", 1) != 1:
                out[name] = dict(note="not available: the packed CPU baseline handles radius 1 only")
                continue
            import baselines as B
            ic = np.zeros(setup["width"], np.int8)
            ic[(setup["width"] - 1) // 2] = 1
            ra0 = np.zeros((setup["height"], setup["width"]), np.int8)
            B.evolve_lifetime_packed_par(ic, ra0, _tables(), setup["target"], 100, np.arange(8))
            secs = []
            for rep in range(3):
                t0 = time.perf_counter()
                best, used = B.evolve_lifetime_packed_par(ic, ra0, _tables(), setup["target"], BUDGET_MUTATIONS,
                                                          rep * 100_003 + np.arange(runs))
                secs.append(time.perf_counter() - t0)
            out[name] = dict(runs=runs, seconds=secs, solved=int((best == 0).sum()),
                             median_mutations=float(np.median(used[best == 0])))
        print(f"    {name}: {np.median(out[name]['seconds']):.3f} s", flush=True)
    return out


def _hash_fn():
    import jax
    import jax.numpy as jnp
    U = jnp.uint32
    rotl = lambda x, r: (x << U(r)) | (x >> U(32 - r))

    @jax.jit
    def h32(words, seed):
        """murmur3-style hash of each column of words (n, lanes) uint32 -> (lanes,) uint32."""
        def body(i, h):
            k = rotl(words[i] * U(0xCC9E2D51), 15) * U(0x1B873593)
            return rotl(h ^ k, 13) * U(5) + U(0xE6546B64)
        h = jax.lax.fori_loop(0, words.shape[0], body, jnp.full(words.shape[1], seed, U))
        h = (h ^ U(4 * words.shape[0]))
        h = (h ^ (h >> U(16))) * U(0x85EBCA6B)
        h = (h ^ (h >> U(13))) * U(0xC2B2AE35)
        return h ^ (h >> U(16))

    def h64(words):
        a, b = (np.asarray(h32(words, U(s))).astype(np.uint64) for s in (0x9747B28C, 0x5BD1E995))
        return (a << np.uint64(32)) | b
    return h64


def metric_census(args) -> dict:
    """Independent runs at Wolfram's size with his budget, no restart: per run the mutations to solution, and for
    each solution 64-bit hashes of its rule array and of its pattern (the spacetime states), and its rule-146 count."""
    if args.impl != "stica":
        raise SystemExit("census: STICA only (the distribution is the same algorithm's; see the reproduction)")
    import jax
    import jax.numpy as jnp
    import stica
    from stica.packed import fused_rollout
    h64 = _hash_fn()
    W, T, L = WOLFRAM["width"], WOLFRAM["height"], args.census_batch
    ic_b = jnp.broadcast_to(stica.pack_cells(stica.single_cell(W))[:, None], (1, L))
    kw = dict(objective=stica.LifetimeLoss(WOLFRAM["target"]), iterations=BUDGET_MUTATIONS, stop_at=0.0,
              backend="pallas", unpack=False)
    batches = -(-args.census_runs // L)
    cols = {k: [] for k in ("mutations", "hash_array", "hash_pattern", "rule146_cells")}
    secs, t_all = [], time.perf_counter()
    for b in range(batches):
        t0 = time.perf_counter()
        res = stica.evolve_packed_batch(jax.random.PRNGKey(10 ** 7 + b), L, stica.single_cell(W),
                                        np.zeros((T, W), np.int8), list(RULES), **kw)
        first = np.asarray(res.solved_at)
        secs.append(time.perf_counter() - t0)
        M = res.state.arrays                                              # (T, nw, L) uint32, the final arrays
        states = fused_rollout(ic_b, M, rules=RULES, width=W, rows_only=False)
        cols["mutations"].append(np.where(first > 0, first, -1).astype(np.int32))
        cols["hash_array"].append(h64(M.reshape(-1, L)))
        cols["hash_pattern"].append(h64(states.reshape(-1, L)))
        cols["rule146_cells"].append(np.asarray(jax.lax.population_count(M).sum((0, 1))).astype(np.int32))
        if b % 16 == 0:
            print(f"    census batch {b + 1}/{batches}: {secs[-1]:.2f} s (evolution), "
                  f"{time.perf_counter() - t_all:.0f} s so far", flush=True)
    data = {k: np.concatenate(v) for k, v in cols.items()}
    np.savez_compressed(Path(args.out) / f"{args.impl}_census.npz", **data)
    solved = data["mutations"] > 0
    curves = {}
    for key in ("hash_array", "hash_pattern"):
        h = data[key][solved]
        ns = sorted({int(2 ** k) for k in range(6, int(math.log2(max(h.size, 2))) + 1)} | {h.size})
        curves[key] = [[n, int(np.unique(h[:n]).size)] for n in ns]
    m = data["mutations"][solved]
    return dict(runs=int(solved.size), solved=int(solved.sum()), batch=L, evolution_seconds=secs,
                first_batch_includes_compile=True, median_mutations=float(np.median(m)),
                quantiles={str(q): float(np.quantile(m, q)) for q in (1e-4, 1e-3, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999)},
                distinct_curves=curves, wall_seconds=time.perf_counter() - t_all)


def _budgeted(args, setup, budget) -> dict:
    """Calibrate the batch size (STICA) and chunk length, then run for `budget` seconds."""
    if args.impl == "stica":
        lanes_list = [int(x) for x in args.lanes.split(",")] if args.lanes else [2048, 8192, 32768]
        cal = calibrate_lanes(args, setup, lanes_list)
        lanes = cal["best"]["lanes"]
    else:
        cal, lanes = None, default_lanes(args)
    chunk, compile_s = sized_chunk(lambda c: make_runner(args, setup, lanes, 3, c), args.chunk_seconds)
    runner = make_runner(args, setup, lanes, args.seed, chunk)
    t0 = time.perf_counter()
    runner.chunk()                                    # compile the first and the continuation call at this length
    runner.chunk()
    warm_s = time.perf_counter() - t0
    runner = make_runner(args, setup, lanes, args.seed, chunk)
    res = budget_run(runner, budget)
    res.update(setup=setup, lanes=lanes, chunk=chunk, calibration=cal, compile_s=compile_s + warm_s, budget_s=budget)
    print(f"    {setup['height']}x{setup['width']} lifetime {setup['target']}: {res['solutions']} solutions in "
          f"{res['seconds']:.0f} s, first at {res['first_solution_s']}, best loss {res['best_loss']}, "
          f"{res['candidates_per_s']:.3e} candidates/s", flush=True)
    return res


def metric_anytime(args) -> dict:
    return _budgeted(args, scaled(args.scale), args.budget)


def metric_reach(args) -> dict:
    """Each size is saved as it completes (<impl>_reach.partial.json), so a preempted job resumes at the next size."""
    partial = Path(args.out) / f"{args.impl}_reach.partial.json"
    rows = json.loads(partial.read_text()) if partial.exists() else []
    done = {r["setup"]["scale"] for r in rows}
    for s in (int(s) for s in args.scales.split(",")):
        if s in done:
            print(f"    scale {s}: done earlier", flush=True)
            continue
        rows.append(_budgeted(args, scaled(s), args.reach_budget))
        partial.write_text(json.dumps(rows))
    return dict(rows=sorted(rows, key=lambda r: r["setup"]["scale"]))


METRICS = dict(latency=metric_latency, first=metric_first, redo=metric_redo, census=metric_census,
               anytime=metric_anytime, reach=metric_reach)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("metric", choices=list(METRICS))
    ap.add_argument("--impl", required=True, choices=["stica", "numba"])
    ap.add_argument("--out", default=str(ROOT / "benchmarks" / "results" / "equal_time"))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--runs", type=int, default=50, help="latency: runs; first: repeats")
    ap.add_argument("--lanes", default=None, help="comma-separated batch sizes (first, anytime, reach)")
    ap.add_argument("--latency-chunk", type=int, default=100)
    ap.add_argument("--first-chunk", type=int, default=16)
    ap.add_argument("--census-runs", type=int, default=2 ** 23)
    ap.add_argument("--census-batch", type=int, default=2 ** 16)
    ap.add_argument("--scale", type=int, default=10, help="anytime: size scale (10 = 640 x 311, lifetime 500)")
    ap.add_argument("--budget", type=float, default=3600.0, help="anytime: seconds")
    ap.add_argument("--scales", default=",".join(map(str, LADDER)), help="reach: size scales")
    ap.add_argument("--reach-budget", type=float, default=600.0, help="reach: seconds per size")
    ap.add_argument("--chunk-seconds", type=float, default=1.0, help="anytime/reach: target seconds per chunk")
    ap.add_argument("--max-batch-bytes", type=float, default=8e9, help="largest packed batch tried in calibration")
    args = ap.parse_args()
    if args.impl == "numba":
        os.environ.setdefault("JAX_PLATFORMS", "cpu")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    md = metadata(args)
    print(f"{args.metric} / {args.impl} on {md['host']} ({md.get('gpu') or md['cpu']})", flush=True)
    start, e0 = time.time(), energy()
    result = METRICS[args.metric](args)
    end, e1 = time.time(), energy()
    rapl = {k: e1[k] - e0[k] for k in e0 if k in e1} or None
    record = dict(metric=args.metric, metadata=md, settings=vars(args), wall_start=start, wall_end=end,
                  rapl_microjoules=rapl, result=result)
    path = Path(args.out) / f"{args.impl}_{args.metric}.json"
    path.write_text(json.dumps(record, indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
