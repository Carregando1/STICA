"""Compare Wolfram's own code (reproduce/wolfram_native.wls, run in Mathematica) with STICA (reproduce/wolfram_lifetime.py).

    JAX_PLATFORMS=cpu python reproduce/compare_wolfram.py

1. Published arrays: both implementations reproduce every digitised panel of the essay's figures.
2. Cross-check: STICA's lifetime loss on every final array of Wolfram's replayed runs equals the loss his code reports.
3. Replay: his figure-5 panels come back from his seeds, run by run, and the runs shown step by step in his figures 3
   and 20 pass through the same lifetimes.
4. Distribution: mutations to reach the target lifetime, his code (seeded runs in Mathematica) against STICA's reruns
   (JAX's random numbers), with runs that never reach it within 10,000 mutations counted as censored at 10,001.
5. Speed: candidates evaluated per second by his code on one kernel, as published and with his RuleArray compiled.

Writes reproduce/results/wolfram_compare.json.
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

import stica

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "reproduce" / "results"
SETUPS = {"rules_4_146": dict(native="lifetime50", rules=[4, 146], radius=1, target=50, width=31),
          "and_xor": dict(native="andxor", rules=[8, 6], radius=0.5, target=30, width=21)}
CENSOR = 10_001
# lifetimes printed under the panels of the essay's figures that show one run as its loss improves, read off
# sw08202024arraysBimg3.png (rules 4/146, SeedRandom[234241]) and sw08202024arraysBimg20.png (And/Xor, SeedRandom[234234])
FIGURE_LIFETIMES = {"replay_lifetime50_seed234241": (50, ["inf", 6, 13, 16, 25, 27, 30, 35, 49, 50]),
                    "replay_andxor_seed234234": (30, [1, 15, 19, 20, 21, 23, 24, 28, 30])}


def stica_loss(rows, setup):
    """Wolfram's LifetimeLoss with STICA, on an array given as rows of 0 (first rule) and 1: |lifetime - target|, or
    inf if the pattern outlives the array."""
    ra = np.array([[int(c) for c in row] for row in rows], np.int8)
    traj = stica.simulate_rule_array(stica.single_cell(setup["width"]), ra, setup["rules"], radius=setup["radius"])
    life = int(stica.lifetime(traj))
    return float("inf") if life == ra.shape[0] + 1 else float(abs(life - setup["target"]))


def native_loss(s):
    return float("inf") if s == "Infinity" else float(s)


def censored(mutations, solved):
    m = np.asarray(mutations, float)
    return np.where(np.asarray(solved, bool), m, CENSOR)


def describe(t):
    s = t[t < CENSOR]
    q = lambda p: float(np.percentile(s, p)) if len(s) else None
    return dict(runs=int(len(t)), solved=int(len(s)), solved_fraction=float(len(s) / len(t)), median=q(50), q25=q(25),
                q75=q(75), mean=float(s.mean()) if len(s) else None)


def main():
    nat = json.loads((RES / "wolfram_native.json").read_text())
    rep = json.loads((RES / "wolfram_lifetime.json").read_text())
    runs = np.load(RES / "wolfram_lifetime_runs.npz")
    out = dict(mathematica=nat.get("version"), code=nat.get("code"))

    # 1. published arrays
    out["published"] = dict(wolfram_code=nat.get("verify"), stica=rep["published"])

    # 2. STICA's loss on the final arrays of his replayed runs
    finals = [(r["array"], r["final_loss"], "rules_4_146") for r in nat.get("replay_lifetime50", [])]
    for key, setup in (("replay_lifetime50_seed234241", "rules_4_146"), ("replay_andxor_seed234234", "and_xor")):
        if key in nat:
            finals.append((nat[key]["array"], nat[key]["final_loss"], setup))
    agree = [stica_loss(a, SETUPS[s]) == native_loss(fl) for a, fl, s in finals]
    out["cross_check"] = dict(arrays=len(finals), losses_equal=int(sum(agree)),
                              unsolved_arrays=int(sum(native_loss(fl) > 0 for _, fl, _ in finals)))

    # 3. replay of his figures
    if "replay_lifetime50" in nat:
        ev = nat["replay_lifetime50"]
        t = censored([r["mutations"] for r in ev], [r["solved"] for r in ev])
        figures = {}
        for key, (target, shown) in FIGURE_LIFETIMES.items():
            replayed = ["inf" if l == "Infinity" else target - int(l) for l in nat[key]["loss_trajectory"]]
            figures[key] = dict(essay=shown, replayed=replayed, equal=replayed == shown)
        out["replay"] = dict(
            figure5=nat["replay_figure5"],
            figure5_reproduced=int(sum(nat["replay_figure5"]["matches_published_in_order"])),
            figure_lifetimes=figures,
            lifetime50_100_seeds=describe(t),
            seed234241={k: v for k, v in nat["replay_lifetime50_seed234241"].items() if k != "array"},
            andxor_seed234234={k: v for k, v in nat["replay_andxor_seed234234"].items() if k != "array"},
            andxor_detail=nat.get("detail_andxor_seed234234"))

    # 4. distributions
    if "sample" in nat:
        out["distribution"] = {}
        for key, setup in SETUPS.items():
            w = nat["sample"][setup["native"]]
            tw = censored(w["mutations"], w["solved"])
            sa = runs[f"{key}_solved_at"]
            ts = np.where(sa > 0, sa, CENSOR).astype(float)
            ks = stats.ks_2samp(tw, ts)
            mw = stats.mannwhitneyu(tw, ts)
            table = [[int((tw < CENSOR).sum()), int((tw == CENSOR).sum())],
                     [int((ts < CENSOR).sum()), int((ts == CENSOR).sum())]]
            out["distribution"][key] = dict(
                wolfram=describe(tw), stica=describe(ts), ks_statistic=float(ks.statistic), ks_pvalue=float(ks.pvalue),
                mannwhitney_pvalue=float(mw.pvalue), solved_fisher_pvalue=float(stats.fisher_exact(table).pvalue),
                wolfram_seconds=w["seconds"], wolfram_kernels=nat["sample"]["kernels"],
                stica_seconds=rep["rerun"][key]["timing"]["run_seconds"])

    # 5. speed of his code
    out["timing"] = dict(interpreted=nat.get("timing_interpreted"), compiled=nat.get("timing_compiled"),
                         cpu=nat.get("timing_cpu"))

    (RES / "wolfram_compare.json").write_text(json.dumps(out, indent=1))
    c = out["cross_check"]
    print(f"cross-check: STICA's loss equals his on {c['losses_equal']}/{c['arrays']} final arrays "
          f"({c['unsolved_arrays']} unsolved)")
    if "replay" in out:
        r = out["replay"]
        print(f"figure 5: {r['figure5_reproduced']}/12 published arrays reproduced from his seeds; "
              f"100 seeds: {r['lifetime50_100_seeds']}")
        for key, f in r["figure_lifetimes"].items():
            print(f"{key}: lifetimes along the run {f['replayed']}, as in the essay's figure: {f['equal']}")
        print(f"And/Xor seed 234234: {r['andxor_seed234234']['mutations']} mutations, "
              f"{r['andxor_seed234234']['accepted']} accepted, {r['andxor_seed234234']['loss_changes']} loss changes")
    for key, d in out.get("distribution", {}).items():
        print(f"{key}: Wolfram {d['wolfram']}\n{' ' * len(key)}  STICA   {d['stica']}\n"
              f"{' ' * len(key)}  KS D = {d['ks_statistic']:.4f} (p = {d['ks_pvalue']:.3g}), Mann-Whitney p = "
              f"{d['mannwhitney_pvalue']:.3g}, solved-fraction Fisher p = {d['solved_fisher_pvalue']:.3g}")
    print("wrote reproduce/results/wolfram_compare.json")


if __name__ == "__main__":
    main()
