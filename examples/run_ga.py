"""Run a batch of independent (1+1)-EA chains and save the results.

    python examples/run_ga.py --grid 32 --states 2 --iterations 2000 --runs 16 --out outputs/demo

Writes <out>.npz (best IC/SRT per run, objective traces, acceptance flags), <out>.json (configuration and software
versions) and, with --stl, the best run's trajectory as a voxel mesh.
"""
import argparse
import json
import time
from pathlib import Path

import jax
import numpy as np

import stica


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grid", type=int, default=32)
    p.add_argument("--states", type=int, default=2)
    p.add_argument("--iterations", type=int, default=1000)
    p.add_argument("--runs", type=int, default=8)
    p.add_argument("--ic-count", type=int, default=None, help="IC cells mutated per iteration (default: grid)")
    p.add_argument("--srt-count", type=int, default=None, help="SRT entries mutated per iteration (default: grid^2)")
    p.add_argument("--objective", default="surface", choices=sorted(stica.OBJECTIVES))
    p.add_argument("--inhomogeneity", default="spacetime", choices=["spacetime", "time", "space", "none"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=Path("outputs/run"))
    p.add_argument("--stl", action="store_true", help="also export the best trajectory as an STL mesh")
    a = p.parse_args()

    G, n = a.grid, a.states
    ic_count = G if a.ic_count is None else a.ic_count
    srt_count = min(G * G, int(np.prod(stica.srt_shape(G, n, a.inhomogeneity)))) if a.srt_count is None else a.srt_count
    t0 = time.perf_counter()
    res = stica.run_ga_batch(jax.random.PRNGKey(a.seed), a.runs, grid_size=G, n=n, iterations=a.iterations,
                             ic_count=ic_count, srt_count=srt_count, objective=a.objective,
                             inhomogeneity=a.inhomogeneity)
    res.best.block_until_ready()
    elapsed = time.perf_counter() - t0

    best = np.asarray(res.best)
    i = int(best.argmin())
    print(f"{a.runs} runs x {a.iterations} iterations in {elapsed:.1f} s (including compilation)")
    print(f"best objective per run: {np.round(best, 3).tolist()}  ->  best run {i}: {best[i]:.3f}")

    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out.with_suffix(".npz"), ic=np.asarray(res.ic), srt=np.asarray(res.srt), best=best,
                        best_trace=np.asarray(res.best_trace), accepted=np.asarray(res.accepted))
    config = {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()}
    config.update(ic_count=ic_count, srt_count=srt_count, seconds=elapsed, stica=stica.__version__,
                  jax=jax.__version__, device=jax.devices()[0].device_kind)
    a.out.with_suffix(".json").write_text(json.dumps(config, indent=2) + "\n")
    if a.stl:
        traj = np.asarray(stica.simulate(res.ic[i], res.srt[i], n, num_steps=G - 1))
        tris = stica.lattice_to_stl(traj, a.out.with_suffix(".stl"))
        print(f"wrote {a.out.with_suffix('.stl')} ({tris} triangles)")
    print(f"wrote {a.out.with_suffix('.npz')} and {a.out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
