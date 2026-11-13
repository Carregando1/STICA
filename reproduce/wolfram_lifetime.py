"""Reproduce Wolfram's lifetime experiments with rule arrays, then scale one up.

    python reproduce/wolfram_lifetime.py                  # all parts
    python reproduce/wolfram_lifetime.py --runs 4096 --scale-runs 8192

Parts
1. Published solutions. The 12 lifetime-50 rule arrays shown in the essay (read off its figure by
   extract_wolfram_figures.py) are simulated and compared with the figure, cell for cell.
2. Wolfram's experiments, rerun many times. The settings are those of the code attached to his figures:
   - rules 4 and 146 (radius 1), a 64 x 31 rule array, target lifetime 50;
   - And and Xor (rules 8 and 6, radius 1/2, alternating neighbourhood), 40 x 21, target lifetime 30.
   Every run starts from a single cell and a uniform array of the first rule, switches one cell's rule per
   mutation, keeps mutations that do not increase the loss (|lifetime - target|, infinite if the pattern outlives
   the array), and stops at zero loss or after 10,000 mutations. Wolfram's seeds drive Mathematica's generator, so
   single runs cannot be matched; the comparison is with the distribution of outcomes.
3. Scale-up: the rules-4/146 task on a rule array 10x taller and 10x wider (100x the cells), as a demonstration,
   run on bit-packed rule arrays (stica.evolve_packed_batch).

Writes reproduce/results/wolfram_lifetime.json (statistics, timings, hardware) and wolfram_lifetime.png.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import platform
import subprocess
import time
from pathlib import Path

import jax
import numpy as np

import stica
from stica import LifetimeLoss, lifetime, simulate_rule_array, single_cell

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reproduce" / "results"
SETUPS = {
    "rules_4_146": dict(rules=[4, 146], radius=1, height=64, width=31, target=50, iterations=10_000,
                        label="rules 4/146, 64x31, lifetime 50"),
    "and_xor": dict(rules=[8, 6], radius=0.5, height=40, width=21, target=30, iterations=10_000,
                    label="And/Xor, 40x21, lifetime 30"),
}
WOLFRAM_REPORTED = {"and_xor": "one run shown: 1705 mutations, 'only 780 of them actually contribute to generating "
                               "the final rule array; all the others are discarded along the way' (the counting "
                               "is not specified)"}


def check_published_solutions() -> dict:
    data = json.loads((ROOT / "tests" / "data" / "wolfram_rule_arrays.json").read_text())
    fig = data["figures"]["lifetime50_solutions"]
    cells = mismatched = 0
    lifetimes = []
    for panel in fig["panels"]:
        ra = np.array([[int(c) for c in r] for r in panel["rule_array"]], np.int8)
        states = np.array([[int(c) for c in r] for r in panel["states"]], np.int8)
        traj = np.asarray(simulate_rule_array(states[0], ra, data["rules"]))
        cells += states.size
        mismatched += int((traj != states).sum())
        lifetimes.append(int(lifetime(traj)))
    return dict(arrays=len(fig["panels"]), cells_compared=cells, cells_mismatched=mismatched, lifetimes=lifetimes,
                source=fig["image"])


def evolve(setup: dict, runs: int, seed: int, chunk: int):
    """Run `runs` evolutions, `chunk` of them in parallel per call. The program is compiled ahead of time, so the
    timings separate compilation from execution. Returns the stacked results and the two timings."""
    from stica.optimize import hill_climb_batch
    from stica.rulearray import _evolve_args, _result

    ic, ra, tables, static = _evolve_args(single_cell(setup["width"]),
                                          np.zeros((setup["height"], setup["width"]), np.int8), setup["rules"],
                                          objective=LifetimeLoss(setup["target"]), k=2, radius=setup["radius"],
                                          neighborhood="alternating", boundary="periodic", num_steps=None,
                                          mutations=1)
    static.update(iterations=setup["iterations"], record=False, accept_ties=True, stop_at=0.0)
    key = jax.random.PRNGKey(seed)
    compiled, compile_s = {}, 0.0
    parts, run_s = [], 0.0
    for i, start in enumerate(range(0, runs, chunk)):
        n = min(chunk, runs - start)
        args = (jax.random.split(jax.random.fold_in(key, i), n), np.broadcast_to(ic, (n,) + ic.shape),
                np.broadcast_to(ra, (n,) + ra.shape), tables)
        if n not in compiled:
            t0 = time.perf_counter()
            compiled[n] = hill_climb_batch.lower(*args, **static).compile()
            compile_s += time.perf_counter() - t0
        t0 = time.perf_counter()
        out = jax.block_until_ready(compiled[n](*args))
        run_s += time.perf_counter() - t0
        parts.append(jax.tree.map(np.asarray, _result(out, 0.0)._replace(mutations=None)))
    res = jax.tree.map(lambda *xs: np.concatenate(xs), *parts)
    return res, compile_s, run_s


def evolve_packed(setup: dict, runs: int, seed: int):
    """The same evolution on bit-packed rule arrays (stica.evolve_packed_batch; on a GPU, one fused kernel per
    candidate evaluation for all runs). No per-iteration trace is kept, so there are no kept-mutation statistics.
    The single timed call includes compilation (a few seconds)."""
    from stica import evolve_packed_batch
    t0 = time.perf_counter()
    res = evolve_packed_batch(jax.random.PRNGKey(seed), runs, single_cell(setup["width"]),
                              np.zeros((setup["height"], setup["width"]), np.int8), setup["rules"],
                              objective=LifetimeLoss(setup["target"]), iterations=setup["iterations"],
                              radius=setup["radius"], stop_at=0.0)
    res = jax.tree.map(np.asarray, jax.block_until_ready(res))
    return res, 0.0, time.perf_counter() - t0


def summarize(name: str, setup: dict, res, compile_s: float, run_s: float) -> dict:
    solved = res.solved_at > 0
    s = res.solved_at[solved]
    accepted = getattr(res, "accepted", None)
    kept = np.array([accepted[i, :res.solved_at[i]].sum() for i in np.flatnonzero(solved)]) \
        if accepted is not None else np.array([])
    runs, iters = len(res.best), setup["iterations"]
    q = lambda a, p: float(np.percentile(a, p)) if len(a) else None
    return dict(
        setup={k: v for k, v in setup.items() if k != "label"}, label=setup["label"], runs=runs,
        solved_fraction=float(solved.mean()),
        mutations_to_solution=dict(median=q(s, 50), q25=q(s, 25), q75=q(s, 75), min=int(s.min()) if len(s) else None,
                                   max=int(s.max()) if len(s) else None, mean=float(s.mean()) if len(s) else None),
        kept_mutations_to_solution=dict(median=q(kept, 50), q25=q(kept, 25), q75=q(kept, 75)),
        kept_fraction_median=q(kept / s, 50) if len(kept) else None,
        cells_changed_from_start=dict(median=q((res.ra[solved] != 0).sum((1, 2)), 50)),
        timing=dict(compile_seconds=compile_s, run_seconds=run_s, runs_per_second=runs / run_s,
                    candidate_evaluations_per_second=runs * iters / run_s,
                    cell_updates_per_second=runs * iters * setup["height"] * setup["width"] / run_s,
                    note="every run executes the full iteration budget; runs that reach zero loss stop changing"),
        wolfram_reported=WOLFRAM_REPORTED.get(name),
    )


# ------------------------------------------------------------------------------------------------------- figures

TEAL, PINK = np.array([136, 226, 226]), np.array([230, 140, 210])
TEAL_DARK, PINK_DARK = np.array([64, 112, 112]), np.array([112, 64, 112])


def render(ic, ra, rules, radius) -> np.ndarray:
    """Wolfram-style picture: background colour = rule (first rule teal, second pink), dark = state 1."""
    traj = np.asarray(simulate_rule_array(ic, ra, rules, radius=radius))
    light, dark = np.stack([TEAL, PINK]), np.stack([TEAL_DARK, PINK_DARK])
    img = np.where(traj[1:, :, None] == 1, dark[ra], light[ra])
    top = np.where(traj[0][:, None] == 1, 0, 255) * np.ones(3, int)
    return np.concatenate([top[None], img]).astype(np.uint8)


def make_figure(results: dict, examples: dict, path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(11, 7.2), layout="constrained")
    grid = fig.add_gridspec(2, 3, height_ratios=[1, 1.25])
    ax = fig.add_subplot(grid[0, 0])
    for name, r in results["rerun"].items():
        s = np.sort(examples[name]["solved_at"][examples[name]["solved_at"] > 0])
        ax.step(s, np.arange(1, len(s) + 1) / r["runs"], where="post", label=r["label"])
    ax.axvline(1705, color="0.5", ls=":", lw=1)
    ax.text(1705, 0.04, " Wolfram's And/Xor run", fontsize=7, color="0.4")
    ax.set(xscale="log", xlabel="mutations to reach the target lifetime", ylabel="fraction of runs solved",
           title="a  Time to solution", ylim=(0, 1.02))
    ax.legend(fontsize=7, loc="upper left")

    ax = fig.add_subplot(grid[0, 1])
    tr = examples["rules_4_146"]["best_trace"][:25]
    for t in tr:
        finite = np.isfinite(t)
        ax.plot(np.flatnonzero(finite) + 1, t[finite], lw=0.8, alpha=0.7)
    ax.set(xscale="log", xlabel="mutation", ylabel="loss |lifetime - 50|", title="b  Loss curves, 25 runs (4/146)")

    ax = fig.add_subplot(grid[0, 2])
    names = list(results["rerun"])
    rates = [results["rerun"][n]["timing"]["candidate_evaluations_per_second"] for n in names]
    ax.barh([results["rerun"][n]["label"] for n in names], rates, color="0.35")
    if "scale" in results:
        ax.barh([results["scale"]["label"]], [results["scale"]["timing"]["candidate_evaluations_per_second"]],
                color="0.6")
    ax.set(xscale="log", xlabel="candidates evaluated / s", title="c  Throughput (one GPU)")
    ax.tick_params(axis="y", labelsize=7)

    sub = grid[1, :].subgridspec(1, 7, width_ratios=[1] * 6 + [3.2])
    setup = SETUPS["rules_4_146"]
    for j, ra in enumerate(examples["rules_4_146"]["solutions"][:6]):
        ax = fig.add_subplot(sub[0, j])
        ax.imshow(render(single_cell(setup["width"]), ra, setup["rules"], setup["radius"]), interpolation="nearest")
        ax.axhline(setup["target"] + 0.5, color="0.3", lw=0.6)
        ax.set_axis_off()
        if j == 0:
            ax.set_title("d  Evolved lifetime-50 arrays (this run)", loc="left", fontsize=9)
    if "scale" in examples:
        ax = fig.add_subplot(sub[0, 6])
        sc = results["scale"]["setup"]
        ax.imshow(render(single_cell(sc["width"]), examples["scale"]["solution"], sc["rules"], sc["radius"]),
                  interpolation="nearest")
        ax.axhline(sc["target"] + 0.5, color="0.3", lw=0.6)
        ax.set_axis_off()
        ax.set_title(f"e  {sc['height']}x{sc['width']}, lifetime {sc['target']}", loc="left", fontsize=9)
    fig.savefig(path, dpi=200)
    plt.close(fig)


def metadata() -> dict:
    meta = dict(date=dt.datetime.now().isoformat(timespec="seconds"), python=platform.python_version(),
                platform=platform.platform(), jax=jax.__version__, stica=stica.__version__,
                device=jax.devices()[0].device_kind)
    try:
        meta["git_commit"] = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"],
                                            capture_output=True, text=True).stdout.strip()
    except OSError:
        pass
    return meta


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", type=int, default=1024, help="independent runs per Wolfram setup")
    ap.add_argument("--chunk", type=int, default=1024, help="runs evolved in parallel per call")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--scale-runs", type=int, default=4096)
    ap.add_argument("--scale-height", type=int, default=640)
    ap.add_argument("--scale-width", type=int, default=311)
    ap.add_argument("--scale-target", type=int, default=500)
    ap.add_argument("--scale-iterations", type=int, default=100_000)
    ap.add_argument("--skip-scale", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    results = dict(metadata=metadata(), published=check_published_solutions(), rerun={})
    p = results["published"]
    print(f"1. published solutions: {p['arrays']} arrays, {p['cells_compared']} cells compared, "
          f"{p['cells_mismatched']} mismatched, lifetimes {sorted(set(p['lifetimes']))}", flush=True)

    examples = {}
    for name, setup in SETUPS.items():
        res, c, r = evolve(setup, args.runs, args.seed, args.chunk)
        results["rerun"][name] = summarize(name, setup, res, c, r)
        solved = np.flatnonzero(res.solved_at > 0)
        examples[name] = dict(solved_at=res.solved_at, best_trace=res.best_trace,
                              solutions=[res.ra[i] for i in solved[:12]])
        s = results["rerun"][name]
        print(f"2. {s['label']}: {s['solved_fraction']:.1%} of {s['runs']} runs solved; median "
              f"{s['mutations_to_solution']['median']:.0f} mutations (IQR {s['mutations_to_solution']['q25']:.0f}-"
              f"{s['mutations_to_solution']['q75']:.0f}), median {s['kept_mutations_to_solution']['median']:.0f} kept; "
              f"{r:.1f} s for all runs (+{c:.1f} s compile)", flush=True)

    if not args.skip_scale:
        setup = dict(SETUPS["rules_4_146"], height=args.scale_height, width=args.scale_width,
                     target=args.scale_target, iterations=args.scale_iterations,
                     label=f"rules 4/146, {args.scale_height}x{args.scale_width}, lifetime {args.scale_target}")
        res, c, r = evolve_packed(setup, args.scale_runs, args.seed + 1)
        results["scale"] = summarize("scale", setup, res, c, r)
        results["scale"]["engine"] = "stica.evolve_packed_batch (bit-packed; fused GPU kernel when on a GPU)"
        results["scale"]["timing"]["note"] = "one call for all runs, including compilation"
        results["scale"]["final_loss"] = dict(median=float(np.median(res.best)), min=float(res.best.min()))
        best = int(np.argmin(np.where(np.isfinite(res.best), res.best, np.inf)))
        examples["scale"] = dict(solution=res.ra[best])
        s = results["scale"]
        print(f"3. {s['label']}: {s['solved_fraction']:.1%} of {s['runs']} runs solved within "
              f"{setup['iterations']} mutations; median final loss {s['final_loss']['median']}; {r:.1f} s "
              f"including compilation", flush=True)

    (OUT / "wolfram_lifetime.json").write_text(json.dumps(results, indent=1) + "\n")
    # per-run data for figures: mutations to solution of every run, and the first evolved solutions
    np.savez_compressed(OUT / "wolfram_lifetime_runs.npz",
                        **{f"{k}_solved_at": v["solved_at"] for k, v in examples.items() if "solved_at" in v},
                        **{f"{k}_solutions": np.stack(v["solutions"]) for k, v in examples.items()
                           if v.get("solutions")})
    make_figure(results, examples, OUT / "wolfram_lifetime.png")
    print(f"wrote {OUT.relative_to(ROOT)}/wolfram_lifetime.json and .png")


if __name__ == "__main__":
    main()
