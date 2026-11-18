# Cluster results (MIT Engaging, 2026-11-17)

Recorded outputs of the runs on the Engaging cluster. Every run's JSON holds its metadata (host, CPU or GPU, commit,
Slurm job) and settings.

| Directory | Produced by | Hardware |
|---|---|---|
| `equal_time/` | `benchmarks/equal_time.py` (numba), `reproduce/wolfram_native.wls` (Wolfram's code, Mathematica 14.2.1, 8 kernels) | 10 cores each of one Intel Xeon Platinum 8580 node, run side by side |
| `equal_time/a100/` | `benchmarks/equal_time.py` (STICA), `benchmarks/scaling.py` | one NVIDIA A100-SXM4-80GB; `power_gpu_*.csv` are `nvidia-smi` power logs (cluster local time, America/New_York) |
| `scaling_laws/` | `benchmarks/scaling_laws.py` (rule tensor capped at 30 GB) | one NVIDIA A100-SXM4-80GB |
| `stochastic/` | `benchmarks/stochastic.py` (cost, robustness, evolve) | one NVIDIA A100-SXM4-80GB |

`equal_time/summary.json` is written by

    python benchmarks/equal_time_summary.py benchmarks/results/engaging/equal_time

The census's per-run arrays (2^24 runs: mutations, hashes; about 300 MB) are not kept; `stica_census_hist.npz` holds
the number of solved runs per number of mutations (`mutation_counts`), the unsolved count and the histogram of rule-146
cells, which is all the summary needs. Absolute cluster paths in the settings were shortened to `~/`.
