# Legacy implementation (superseded)

This folder keeps the original STICA code (NSTICA for N states, STICA2S for 2 states) exactly as it was on
`main` before the November 2026 rebuild. It is kept for traceability of the preliminary runtime results on the
`A_result_prelim` branch. **Do not use it for new results**; use the `stica` package in `src/stica/` instead.

The legacy NSTICA code is still runnable from `legacy/STICA-Main/NSTICA` (`from algorithm.genetic import ...`) and
`benchmarks/bench.py --impl legacy` uses it for speed comparisons. Importing it turns on `jax_enable_x64` for the whole
process, so run it in a separate process from the new package.

## Known problems (all verified on GPU, jax 0.6.2 / flax 0.10.7 / cax 0.2.1)

| Where | Problem | Effect |
|---|---|---|
| `NSTICA/.../nstica.py` (`NSTICAUpdate.__call__`) | The rule index uses `factorial(7 - S)`; when all 8 neighbours are non-zero, `S = 8` and jax returns `factorial(-1) = 0`, so the index is `inf`. | Every cell whose 8 neighbours are all non-zero reads the wrong rule (a state-0 cell reads the last rule, other cells read rule 0). About 0.4 % of cells at 2 states and 23 % at 6 states for uniformly random states. All N-state results are affected. |
| `NSTICA/.../mutation.py` (`mutate`) | `new_ic` / `new_srt` are only assigned when `ic_enable` / `srt_enable` is true. | `UnboundLocalError` whenever IC or SRT mutation is disabled. |
| `*/genetic/optimize.py` (`step_muts`) | Uses `self.states` (does not exist), slices Python lists, reads SRT values from `ic_muts`, indexes the 4-D SRT with 3 coordinates, and edits the current state in place. | Unusable; only called from the stale transformer notebooks. |
| `*/objectives/surfacevol.py` | `-surface / vol` with no guard. | An empty lattice gives NaN; `NaN >= x` is false, so the optimiser accepts it and every later mutation. |
| `NSTICA/.../nstica.py` | `self.time` is a plain array attribute mutated inside `nnx.jit`/`nnx.scan`. | Works with flax 0.10.7 only by implicit behaviour; versions were never pinned. |
| `STICA2S/.../objectives/__init__.py`, `genetic/test_transition.py` | Import a `cax_sica` package / use `from transition import`. | ImportError; the STICA2S tests never ran. |
| `STICA2S` vs `NSTICA` | 2-state rule order is ascending neighbour count in STICA2S and descending in NSTICA. | A 2-state SRT from one is wrong in the other (`stica.rules.reorder_two_state_rules` converts). |
| `*/transformer/*.ipynb` | Copied from an earlier project (`lamm_automata`); imports classes that do not exist here. | Saved with NameErrors; not part of STICA. |

## Performance notes

Per GA step the legacy code builds a new CA module, casts the full SRT to float64 on the host and copies it to the GPU
(at grid 100 and 1000 steps: 350 ms cast + 154 ms copy + 88 ms simulation). A 4-state 64x64x200 SRT needs 4.3 GB as
float64 and does not fit on a 20 GB GPU together with its copies.

## Timing data on `A_result_prelim`

The timing notebooks used `time.time()` (wall clock). Clock corrections during the runs shifted individual
measurements by about 0.5-1.8 s; the analysis notebook removes only the negative ones, so shifted positive values remain
in the reported means. Those numbers should not be published; rerun with `benchmarks/bench.py`.
