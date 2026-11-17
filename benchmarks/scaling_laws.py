"""Scaling laws of STICA on one device: time against grid size and against the size of the rule space.

    python benchmarks/scaling_laws.py [suite ...] [--out DIR] [--tag TAG] [--quick] [--max-bytes 2e10]

Suites (default: all)
  grid1d    one W x W rule array (two rules, radius 1), W = 32 .. 65,536: unpacked and bit-packed (XLA)
  batch1d   B arrays of Wolfram's size (64 steps x 31 cells) at once, B = 1 .. 2**20: bit-packed XLA (vmap) and the
            fused kernel (rows only, as in evolution)
  rules1d   the rule space of 1D arrays at 1024 x 1024: K = 2 .. 256 different rules per array (bit-packed planes;
            unpacked tables), and radius r = 1/2, 1, 2, 3 (tables of 4 .. 128 entries) with K = 2
  grid2d    2D STICA, a G x G grid for G steps, n = 2 .. 6 states (R = 18 .. 7,722 rules per cell), at the four sharing
            levels (spacetime (G, G, G, R), time only (G, 1, 1, R), space only (1, G, G, R), homogeneous (1, 1, 1, R)),
            as far as the rule tensor fits in --max-bytes
  evo1d     evolution (candidates per second): against array size (scale s: 64 s x (31 s + 1), lifetime 50 s) on the
            fused kernel and the XLA path, and against K = 2 .. 16 rules (XLA path) at Wolfram's size

Timing: a compile call, a warm-up of --warmup seconds, then 5 repetitions, each a loop of calls lasting at least
--min-rep seconds; the median per-call time is kept. A condition slower than --max-call seconds ends its series.
Rows: <out>/scaling_laws_<tag>.csv. Fits: for every series, t = t0 + N / rate (weighted least squares on relative
error; t0 = fixed cost per call, rate = cell updates per second at large N) and the exponent of t against N over the
three largest conditions, in <out>/scaling_laws_<tag>_fits.json.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import platform
import socket
import subprocess
import time
from functools import partial
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import stica
from stica.packed import _Model, _simulate_words, _tmask, fused_rollout, pack_cells, pack_rule_array

ROOT = Path(__file__).resolve().parents[1]
FIELDS = ["suite", "impl", "series", "param", "size", "batch", "cell_updates", "rule_bytes", "seconds", "rate",
          "note"]


def measure(fn, args):
    t0 = time.perf_counter()
    jax.block_until_ready(fn())
    first = time.perf_counter() - t0
    if first > args.max_call:
        return first, "single call, includes compilation"
    end = time.perf_counter() + args.warmup
    while time.perf_counter() < end:
        jax.block_until_ready(fn())
    t0 = time.perf_counter()
    jax.block_until_ready(fn())
    one = time.perf_counter() - t0
    calls = max(1, int(np.ceil(args.min_rep / max(one, 1e-7))))
    reps = []
    for _ in range(5 if one < 2 else 3):
        t0 = time.perf_counter()
        for _ in range(calls):
            out = fn()
        jax.block_until_ready(out)
        reps.append((time.perf_counter() - t0) / calls)
    return float(np.median(reps)), f"{calls} calls per repetition"


class Recorder:
    def __init__(self, args):
        self.args, self.rows = args, []

    def __call__(self, suite, impl, series, param, size, batch, work, rule_bytes, fn):
        try:
            sec, note = measure(fn, self.args)
        except Exception as e:                                       # out of device memory: this series ends here
            if "RESOURCE_EXHAUSTED" not in str(e):
                raise
            print(f"  {suite:8s} {impl:18s} {series:22s} param={param:5} size={size:7d}: out of memory, series ends",
                  flush=True)
            return float("inf")
        self.rows.append(dict(suite=suite, impl=impl, series=series, param=param, size=size, batch=batch,
                              cell_updates=work, rule_bytes=rule_bytes, seconds=sec, rate=work / sec, note=note))
        print(f"  {suite:8s} {impl:18s} {series:22s} param={param:5} size={size:7d} batch={batch:8d}: "
              f"{sec * 1e3:11.3f} ms, "
              f"{work / sec:.3e} cell updates/s", flush=True)
        return sec


def suite_grid1d(rec, args):
    rng = np.random.default_rng(0)
    for impl in ("unpacked", "packed"):
        for W in [32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536][: 7 if args.quick else None]:
            if W * W * (9 if impl == "unpacked" else 1) > args.max_bytes:
                break
            ic = jnp.asarray((rng.random(W) < 0.5).astype(np.int8))
            if impl == "unpacked":
                ra = jnp.asarray(rng.integers(0, 2, (W, W)).astype(np.int8))
                tables = stica.rule_tables([4, 146])
                fn = jax.jit(lambda: stica.rule_array_rollout(ic, ra, tables))
            else:
                # random packed words drawn directly (packing a W x W int8 array needs a 4x larger temporary)
                ra = jax.random.bits(jax.random.PRNGKey(W), (1, W, -(-W // 32)), jnp.uint32) & jnp.asarray(
                    stica.packed._valid_mask(W))
                model = _Model(W, 2, 1, "alternating", "periodic")
                icw, planes, tm = pack_cells(ic), ra, _tmask([4, 146], 1)
                fn = lambda: _simulate_words(icw, planes, tm, model, W)
            if rec("grid1d", impl, "W x W, K=2, r=1", 2, W, 1, W * W, W * W, fn) > args.max_call:
                break
            del ra


def suite_batch1d(rec, args):
    rng = np.random.default_rng(1)
    T, W = 64, 31
    tm = _tmask([4, 146], 1)
    model = _Model(W, 2, 1, "alternating", "periodic")
    sim = jax.jit(jax.vmap(lambda ic, pl: _simulate_words(ic, pl, tm, model, T)))
    for impl in ("packed XLA", "fused kernel"):
        for B in [1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576][: 7 if args.quick else None]:
            if impl == "fused kernel" and jax.default_backend() != "gpu":
                break
            if impl == "fused kernel" and B < 128:                       # its batches are multiples of 128 lanes
                continue
            Bk = B
            masks = jnp.asarray(rng.integers(0, 2 ** 31, (T, 1, Bk), dtype=np.int64).astype(np.uint32) & 0x7FFFFFFF)
            ic = jnp.full((1, Bk), 1 << 15, jnp.uint32)
            if impl == "fused kernel":
                fn = lambda: fused_rollout(ic, masks, rules=(4, 146), width=W, rows_only=True)
            else:
                pl = jnp.moveaxis(masks, -1, 0)[:, None]                                 # (B, 1, T, 1)
                icb = ic.T                                                               # (B, 1)
                fn = lambda: sim(icb, pl)
            if rec("batch1d", impl, "64 x 31 arrays", 2, W, Bk, Bk * T * W, Bk * T * 4, fn) > args.max_call:
                break


def suite_rules1d(rec, args):
    rng = np.random.default_rng(2)
    N = 256 if args.quick else 1024
    ic = jnp.asarray((rng.random(N) < 0.5).astype(np.int8))
    for impl in ("unpacked", "packed"):
        for K in [2, 4, 8, 16, 32, 64, 128, 256]:
            rules = sorted(rng.choice(256, K, replace=False).tolist())
            ra = jnp.asarray(rng.integers(0, K, (N, N)).astype(np.int32))
            if impl == "unpacked":
                tables = stica.rule_tables(rules)
                fn = jax.jit(lambda: stica.rule_array_rollout(ic, ra.astype(jnp.int8 if K <= 127 else jnp.int16),
                                                              tables))
            else:
                model = _Model(N, K, 1, "alternating", "periodic")
                icw, planes, tm = pack_cells(ic), pack_rule_array(ra, K), _tmask(rules, 1)
                fn = lambda: _simulate_words(icw, planes, tm, model, N)
            rec("rules1d", impl, f"K rules, {N} x {N}", K, N, 1, N * N, N * N * max(1, int(np.ceil(np.log2(K)))) // 8,
                fn)
        for r in (0.5, 1, 2, 3):
            m = stica.num_inputs(r)
            rules = [int(rng.integers(0, 2 ** 62)) % (2 ** (2 ** m)) for _ in range(2)]
            ra = jnp.asarray(rng.integers(0, 2, (N, N)).astype(np.int8))
            if impl == "unpacked":
                tables = stica.rule_tables(rules, 2, r)
                fn = jax.jit(lambda: stica.rule_array_rollout(ic, ra, tables, radius=r))
            else:
                model = _Model(N, 2, r, "alternating", "periodic")
                icw, planes, tm = pack_cells(ic), pack_rule_array(ra, 2), _tmask(rules, r)
                fn = lambda: _simulate_words(icw, planes, tm, model, N)
            rec("rules1d", impl, f"radius, {N} x {N}", 2 ** m, N, 1, N * N, N * N // 8, fn)


@partial(jax.jit, static_argnames=("shape",))
def _repeat(block, shape):
    """block (tb, hb, wb, R) repeated to (Ts, Hs, Ws, R) in one gather (a single output buffer)."""
    Ts, Hs, Ws = shape
    t, x, y = (jnp.arange(m) % b for m, b in zip(shape, block.shape[:3]))
    return block[t[:, None, None], x[None, :, None], y[None, None, :]]


SHARING = {"spacetime": lambda G: (G, G, G), "time only": lambda G: (G, 1, 1), "space only": lambda G: (1, G, G),
           "homogeneous": lambda G: (1, 1, 1)}


def suite_grid2d(rec, args):
    for sharing, shape in SHARING.items():
        for n in (2, 3, 4, 5, 6):
            R = stica.num_rules(n)
            for G in [16, 32, 64, 128, 256, 512, 1024][: 5 if args.quick else None]:
                Ts, Hs, Ws = shape(G)
                nbytes = Ts * Hs * Ws * R
                if nbytes > args.max_bytes or 4 * (G + 1) * G * G > args.max_bytes:
                    break
                key = jax.random.PRNGKey(G * 10 + n)
                # a random block of at most 64 x 64 cells repeated over time and space, written straight into the
                # tensor (the generator works in 32-bit words, 4x the memory; jnp.tile makes a further copy); the
                # values do not change the work per step
                hb, wb = min(Hs, 64), min(Ws, 64)
                tb = max(1, min(Ts, int(2 ** 26 // max(1, hb * wb * R))))
                try:
                    block = (jax.random.bits(key, (tb, hb, wb, R), jnp.uint8) % n).astype(jnp.int8)
                    srt = _repeat(block, (Ts, Hs, Ws))
                    del block
                except Exception as e:
                    if "RESOURCE_EXHAUSTED" not in str(e):
                        raise
                    print(f"  grid2d   {sharing:18s} n={n} G={G}: rule tensor does not fit, series ends", flush=True)
                    break
                ic = (jax.random.bits(jax.random.fold_in(key, 1), (G, G), jnp.uint8) % n).astype(jnp.int32)
                fn = lambda: stica.simulate(ic, srt, n, num_steps=G)
                sec = rec("grid2d", sharing, f"n={n} (R={R})", n, G, 1, G ** 3, nbytes, fn)
                del srt
                if sec > args.max_call:
                    break


def suite_evo1d(rec, args):
    for backend in (["pallas", "xla"] if jax.default_backend() == "gpu" else ["xla"]):
        for s in [1, 2, 4, 8, 16][: 3 if args.quick else None]:
            H, W, target = 64 * s, (31 if s == 1 else 31 * s + 1), 50 * s
            lanes = 8192 if backend == "pallas" else 1024
            its = 64
            fn = lambda: stica.evolve_packed_batch(
                jax.random.PRNGKey(s), lanes, stica.single_cell(W), np.zeros((H, W), np.int8), [4, 146],
                objective=stica.LifetimeLoss(target), iterations=its, stop_at=0.0, restart=True, backend=backend,
                keep_last_solution=False, unpack=False).best
            rec("evo1d", backend, "array size", s, H * W, lanes, lanes * its * H * W, H * W // 8, fn)
    rng = np.random.default_rng(3)
    for K in (2, 4, 8, 16):
        rules = sorted(rng.choice(256, K, replace=False).tolist()) if K > 2 else [4, 146]
        fn = lambda: stica.evolve_packed_batch(
            jax.random.PRNGKey(K), 1024, stica.single_cell(31), np.zeros((64, 31), np.int8), rules,
            objective=stica.LifetimeLoss(50), iterations=64, stop_at=0.0, restart=True, backend="xla",
            keep_last_solution=False, unpack=False).best
        rec("evo1d", "xla", "K rules", K, 64 * 31, 1024, 1024 * 64 * 64 * 31, 64 * 31 * K.bit_length() // 8, fn)


SUITES = dict(grid1d=suite_grid1d, batch1d=suite_batch1d, rules1d=suite_rules1d, grid2d=suite_grid2d,
              evo1d=suite_evo1d)


def fit(rows) -> list:
    """Per series: t = t0 + N / rate by least squares on relative error, and the large-N exponent."""
    out = []
    keys = sorted({(r["suite"], r["impl"], r["series"], r["param"]) for r in rows}, key=str)
    for k in keys:
        pts = sorted((r["cell_updates"], r["seconds"]) for r in rows if (r["suite"], r["impl"], r["series"],
                                                                          r["param"]) == k)
        if len(pts) < 3 or k[0] in ("rules1d",):
            continue
        N, t = (np.array(x, float) for x in zip(*pts))
        A = np.stack([1 / t, N / t], 1)                                    # (t0 + b N) / t = 1
        t0, b = np.linalg.lstsq(A, np.ones_like(t), rcond=None)[0]
        slope = float(np.polyfit(np.log(N[-3:]), np.log(t[-3:]), 1)[0])
        out.append(dict(suite=k[0], impl=k[1], series=k[2], param=k[3], t0_seconds=float(t0),
                        rate_cell_updates_per_s=float(1 / b) if b > 0 else None, large_n_exponent=slope,
                        points=len(pts), N_range=[float(N[0]), float(N[-1])]))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("suites", nargs="*", default=list(SUITES))
    ap.add_argument("--out", default=str(ROOT / "benchmarks" / "results"))
    ap.add_argument("--tag", default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--max-bytes", type=float, default=2e10, help="largest rule tensor or trajectory to allocate")
    ap.add_argument("--warmup", type=float, default=1.0)
    ap.add_argument("--min-rep", type=float, default=0.2)
    ap.add_argument("--max-call", type=float, default=30.0)
    args = ap.parse_args()
    tag = args.tag or f"jax-{jax.default_backend()}"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rec = Recorder(args)
    print(f"scaling laws on {socket.gethostname()}: {jax.devices()}", flush=True)
    for s in args.suites:
        SUITES[s](rec, args)
        with open(out / f"scaling_laws_{tag}.csv", "w", newline="") as f:          # after every suite
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rec.rows)
    gpu = None
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20).stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        pass
    meta = dict(date=dt.datetime.now().isoformat(timespec="seconds"), host=socket.gethostname(), gpu=gpu,
                jax=jax.__version__, python=platform.python_version(), settings=vars(args),
                commit=os.environ.get("STICA_COMMIT") or subprocess.run(
                    ["git", "-C", str(ROOT), "describe", "--always", "--dirty"], capture_output=True,
                    text=True).stdout.strip())
    (out / f"scaling_laws_{tag}_fits.json").write_text(json.dumps(dict(metadata=meta, fits=fit(rec.rows)), indent=1))
    print(f"wrote {out}/scaling_laws_{tag}.csv and _fits.json")


if __name__ == "__main__":
    main()
