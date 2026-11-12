#!/usr/bin/env bash
# Every benchmark behind the README tables and the paper figures, one implementation at a time (nothing else should
# run on the machine meanwhile). Set STICA_MEMCAP (e.g. 14G) to cap memory (and forbid swap) with systemd-run
# where it is available.
set -euo pipefail
cd "$(dirname "$0")/.."
run() {
  if [ -n "${STICA_MEMCAP:-}" ] && command -v systemd-run >/dev/null; then
    systemd-run --user --scope -q -p MemoryMax="$STICA_MEMCAP" -p MemorySwapMax=0 "$@"
  else
    "$@"
  fi
}
run python benchmarks/scaling.py --impl jax
JAX_PLATFORMS=cpu run python benchmarks/scaling.py --impl jax --tag jax-cpu
for impl in numba numba-par numpy cellpylib; do
  run python benchmarks/scaling.py --impl "$impl"
done
run python benchmarks/bench.py --impl stica
run python benchmarks/bench.py --impl legacy
run python reproduce/wolfram_lifetime.py --runs 4096 --chunk 4096
python benchmarks/plot_scaling.py benchmarks/results/scaling_*.csv > benchmarks/results/scaling_tables.md
python benchmarks/summarize.py benchmarks/results/stica.csv benchmarks/results/legacy.csv > benchmarks/results/legacy_tables.md
