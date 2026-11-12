# STICA: spacetime-inhomogeneous cellular automata in JAX

STICA is a cellular automaton in which **every cell at every time step can use its own update rule**. Stephen Wolfram
introduced this model as *rule arrays*, "'spacetime-inhomogeneous' generalizations of cellular automata", in his
minimal models of machine learning [1]. Wolfram's rule arrays are one-dimensional and binary, and each cell chooses
between two fixed rules.

This package implements Wolfram's rule arrays exactly, generalises them, and runs both on accelerators:

- **Wolfram's rule arrays (1D)**: rule numbers as in `CellularAutomaton`, k colours, radius r or 1/2 ("brick"
  lattices), a finite rule set or a full rule table per cell. The simulator reproduces, cell for cell, all 12 evolved
  rule arrays published in [1].
- **2D** lattices with the Moore neighbourhood and **N states**, with a **full outer-totalistic lookup table per cell
  and time step**: the rule sees the cell's own state and how many neighbours are in each state.
- **Rule sharing**: in both models the rules can be shared across time, across space, or both. Sharing across both
  gives an ordinary homogeneous CA.
- A **CAX model** [2] in which time is carried in a state channel, so STICA is an ordinary autonomous CA.
- **Adaptive evolution** of rule arrays, initial conditions and rule tensors (random mutations, kept unless the loss
  increases), running entirely on the GPU and batched over independent runs. Wolfram's lifetime experiment runs 4,096
  times in about two seconds on one desktop GPU.
- A **bit-packed engine** for Wolfram's binary rule arrays: 32 cells per machine word, and a fused GPU kernel that
  evaluates a whole batch of candidate arrays in one launch. It finds about 200,000 solutions of Wolfram's lifetime
  task per second, 11× more than the same algorithm bit-packed on all 20 threads of a workstation CPU.

> The original prototype is kept in [`legacy/`](legacy/README.md), together with a list of the problems it had.

## Installation

```bash
pip install -e ".[dev]"            # CPU
pip install -e ".[dev,cuda12]"     # NVIDIA GPU (CUDA 12)
pip install -e ".[dev,bench]"      # also the CPU baselines and plotting used by benchmarks/ and reproduce/
```

Tested with Python 3.11–3.13, jax 0.6.2, flax 0.10.7 and cax 0.2.1. These versions are pinned in `pyproject.toml`.

## Quick start

```python
import jax
import numpy as np
import stica

# Wolfram's rule arrays: rules 4 and 146 on a 64 x 31 array; evolve 1,024 arrays whose pattern lives exactly 50 steps
ic, ra0 = stica.single_cell(31), np.zeros((64, 31), np.int8)          # ra0: every cell uses rules[0] = rule 4
res = stica.evolve_rule_array_batch(jax.random.PRNGKey(0), 1024, ic, ra0, [4, 146],
                                    objective=stica.LifetimeLoss(50), iterations=10_000, stop_at=0.0)
print((res.solved_at > 0).mean())                 # fraction of runs that reached lifetime 50
traj = stica.simulate_rule_array(ic, res.ra[0], [4, 146])              # (65, 31): the IC and every later step

# the same on bit-packed arrays, restarting every run that succeeds: count solutions found in a fixed budget
res = stica.evolve_packed_batch(jax.random.PRNGKey(0), 4096, ic, ra0, [4, 146], objective=stica.LifetimeLoss(50),
                                iterations=2000, stop_at=0.0, restart=True)       # use ~32768 runs on a GPU
print(res.solves.sum())                           # lifetime-50 rule arrays found

# 2D STICA: random IC and rule tensor for 3 states on a 32x32 grid, 50 steps
n, G, T = 3, 32, 50
key_ic, key_srt = jax.random.split(jax.random.PRNGKey(1))
ic = jax.random.randint(key_ic, (G, G), 0, n)
srt = jax.random.randint(key_srt, (T, G, G, stica.num_rules(n)), 0, n)
traj = stica.simulate(ic, srt, n)                 # (T + 1, G, G): the IC and every later step

# the same model as a CAX cellular automaton
ca = stica.STICA(srt, num_states=n)
traj = ca.run(ic)
state, cells = ca.step(stica.initial_state(ic))  # one step at a time; state[..., 1] is the time

# adaptive evolution in 2D: 16 independent runs in parallel
res = stica.run_ga_batch(jax.random.PRNGKey(2), 16, grid_size=32, n=2, iterations=1000,
                         ic_count=32, srt_count=32 * 32, objective="surface")
print(res.best)                                   # best objective value of each run
```

`python examples/run_ga.py --help` runs a 2D batch from the command line and saves the results and their metadata.

## Wolfram's rule arrays (1D)

A rule array `ra` (T, W) holds, for every step and cell, an index into a list of rules. Step t computes

```
cell'(x) = rules[ra[t, x]][code_t(x)]
```

where code_t(x) is the neighbourhood of x read as a base-k number. The conventions are those of the `RuleArray`
code that Wolfram published with the figures of [1]:

- Rule numbers follow `CellularAutomaton`. The neighbourhood (left, ..., right) is read with the leftmost cell most
  significant, and digit `code` of the rule number (base k) is the new state. Rules 4 and 146 are `[4, 146]`; Wolfram's
  two-input And and Xor are `[8, 6]` with `radius=0.5`.
- The lattice is cyclic (`boundary="fixed"` treats cells outside as 0 instead).
- Radius-1/2 rules alternate between reading (centre, right) on even steps and (left, centre) on odd steps
  (`neighborhood="alternating"`, Wolfram's default), or always read one pair.
- `single_cell(W)` puts the single black cell at (W - 1) // 2, like `CenterArray[1, W]`.
- `ra` may have shape (T, W), (T, 1) (Wolfram's "layered" array), (1, W) ("vertically layered") or (1, 1).
- `simulate_table_array` is the generalisation in which every cell and step owns a full lookup table,
  tables[t, x, code].

`LifetimeLoss(target)` is Wolfram's loss: |lifetime - target|, or infinity if the pattern outlives the array.
`evolve_rule_array` and `evolve_rule_array_batch` run his adaptive evolution (see Optimisation).

**Checked against Wolfram's own results** (`tests/test_rulearray.py`):
- **Published figures.** [1] shows 12 evolved lifetime-50 rule arrays (64 × 31), 8 example arrays and 4 hand-built
  solutions. `reproduce/extract_wolfram_figures.py` reads every cell's rule and state off the published images
  (`tests/data/wolfram_rule_arrays.json`). The simulator reproduces all of them cell for cell: 24,180 cells of the
  lifetime-50 arrays alone, with no mismatch.
- **His code.** A line-by-line port of Wolfram's `RuleArray` function, for radius 1 and all three radius-1/2
  neighbourhoods.
- **Known patterns.** Rule 90 grown from one cell is Pascal's triangle mod 2, and the centre column of rule 30 is
  OEIS A051023.

**Bit-packed engine** (`stica.packed`), for binary (k = 2) rule arrays:
- **Representation.** 32 cells are stored per uint32 word. A rule is applied to all 32 at once as a sum of minterms
  of the shifted words, and the rule array is stored as bit-planes.
- **Interface.** `simulate_packed` takes the same arguments as `simulate_rule_array` and returns the same trajectory.
  `evolve_packed_batch` runs Wolfram's evolution, one point mutation per iteration.
- **Backends.**
  - `backend="pallas"` uses one fused GPU kernel per candidate evaluation for the whole batch: each block of 128 runs
    is advanced through all T steps in registers, with the two rules compiled into the logic. It handles two rules,
    radius 1 or 1/2.
  - `backend="xla"` runs anywhere and supports any number of rules and any radius.
  - The default picks the kernel on a GPU when it applies.
- **Lifetime objectives.** An objective that only needs to know which rows are non-empty can declare
  `rows_only = True`, as `LifetimeLoss` does; the kernel then skips writing the cells.
- **Tests.** Everything here is tested bit for bit against the unpacked simulator, and every solution it reports is
  re-checked with it.

## The 2D model

**State and neighbourhood.**
- Cells take states 0..n-1, and each cell sees its 8 Moore neighbours.
- Cells outside the grid count as state 0.

**Rule index.** A cell in state `s` whose neighbours contain `a_k` cells in state `k`, for k = 1..n-1, uses rule

```
index = s * C(7 + n, 8) + rank(a_1, ..., a_{n-1})
```

- `rank` is the position of the tuple when all tuples with `a_1 + ... + a_{n-1} <= 8` are sorted in descending
  lexicographic order.
- For n = 3 the order starts (8,0), (7,1), (7,0), (6,2), ... and has C(10, 8) = 45 entries, so each cell and step has
  R = n * C(7 + n, 8) = 135 rules.
- The rank is computed exactly with integer binomial coefficients.
- `stica.neighbor_tuples(n)` lists the order.

**Spacetime rule tensor (SRT).**
- Its shape is (T, H, W, R), with entries in 0..n-1, and `srt[t]` takes the lattice from step t to step t + 1.
- An axis of size 1 shares rules along that axis:

| SRT shape | Rules vary over | Equivalent to |
|---|---|---|
| (T, H, W, R) | time and space | Wolfram's general rule array |
| (T, 1, 1, R) | time only | Wolfram's "layered" rule array |
| (1, H, W, R) | space only | a non-uniform CA ("vertically layered") |
| (1, 1, 1, R) | nothing | an ordinary homogeneous CA |

**Time as a state channel.**
- `stica.STICA` is a CAX `CA` whose state is (H, W, 2): channel 0 holds the cells and channel 1 holds the time step.
- Every update reads `srt[t, x, y, index]` and then sets t to t + 1, whatever the cells do.
- The SRT is a constant parameter of the update module (an `nnx.Variable`), so `set_srt` swaps it without recompiling.
- For t >= T the cells keep their state. `run` refuses to step past the end of the SRT.

**Converting legacy 2-state rule tables.** The legacy STICA2S code indexed 2-state rules as `9 * s + count`.
`stica.reorder_two_state_rules` converts such tables to the order used here, and converts them back.

## Optimisation

All optimisers share one compiled (1+1) evolutionary loop (`stica.hill_climb`), i.e. a stochastic hill climber. Each
iteration:

1. picks exactly the requested number of distinct entries (cells of the IC, entries of the rule tensor, or cells of
   the rule array);
2. moves each one to a uniformly chosen different value (another state, or another rule from the set);
3. simulates the candidate and evaluates the loss;
4. keeps the candidate if the loss did not increase.

Notes:
- Keeping ties (`accept_ties=True`, the default) follows Wolfram's adaptive evolution [1].
- `accept_ties=False` keeps only strict improvements, as the legacy code did.
- `stop_at` freezes a run once its loss reaches that value; Wolfram's runs stop at zero loss.
- `restart=True` (with `stop_at`) counts the solution and starts that run again from the initial array instead of
  leaving it idle. A batch then keeps producing solutions for its whole budget; `solves` counts them.
  - Runs still in progress when the budget ends are not counted. Solution-time statistics from a restart run are
    therefore biased toward short runs. Measure them without restart, and use restart for throughput.
- The whole loop is one compiled `lax.scan`. Only the mutated entries are written, and they are reverted if the
  candidate is rejected.
- The batch versions run independent chains in parallel with `vmap`; results are reproducible from the PRNG key.
- `record_mutations=True` returns every mutation (position, old value, new value).

2D: `run_ga` and `run_ga_batch` evolve an IC and an SRT. Their objectives are minimised and treat the trajectory as a
3D voxel lattice, with time as the first axis:
- Built-in: `"surface"` (negative exposed surface area) and `"surface_to_volume"`.
- Any JAX function from the trajectory to a scalar also works.
- `stica.lattice_to_stl` exports a trajectory as a voxel mesh.

1D: `evolve_rule_array` and `evolve_rule_array_batch` evolve a rule array from a fixed initial condition. The loss is
any hashable function of the trajectory, such as `LifetimeLoss(target)`.

## Reproducing Wolfram's lifetime experiments

```bash
python reproduce/wolfram_lifetime.py      # writes reproduce/results/wolfram_lifetime.{json,png}
```

The script reruns the two lifetime experiments of [1] with the settings in the code attached to his figures. Each run:
- starts from a single cell and a uniform array of the first rule;
- switches one cell's rule per mutation and keeps mutations that do not increase the loss;
- stops at zero loss or after 10,000 mutations.

Wolfram's random seeds drive Mathematica's generator, so single runs cannot be matched; the comparison is with the
distribution over many runs.

| Experiment | Runs | Solved | Mutations to solution, median (IQR) | Time, one GPU |
|---|---:|---:|---:|---:|
| Rules 4/146, 64 × 31, lifetime 50 | 4,096 | 99.9% | 1,992 (1,108–2,921) | 2.3 s |
| And/Xor (radius 1/2), 40 × 21, lifetime 30 | 4,096 | 100% | 1,830 (1,418–2,400) | 1.6 s |
| Rules 4/146, 640 × 311, lifetime 500 (100× the cells) | 4,096 | 6% within 100,000 | — | 189 s |

How these compare with Wolfram's runs:
- His one And/Xor run took 1,705 mutations, inside the interquartile range of ours.
- He writes that only 780 of those mutations "contribute" to the final array. He does not say how they were
  counted, so we do not compare against that number.
- Compilation (about 3 s per setup) is not included in the times, except for the scale-up, which runs on the
  bit-packed engine and is timed in a single call.
- The GPU is an NVIDIA RTX 4000 Ada.

## Tests

```bash
JAX_PLATFORMS=cpu pytest -q
```

The suite has 150 tests; 10 of them test the fused GPU kernel and run only on a GPU. They check:
- the 1D rule arrays against Wolfram's published figures, a port of his code, known elementary-CA patterns and a
  generic k-colour reference, for every neighbourhood, boundary and rule-sharing level;
- the 2D simulation, bit for bit, against a loop-based NumPy reference: 2–6 states, dense neighbourhoods, and every
  rule-sharing level;
- the 2D rule index exhaustively, up to 8 states;
- the CAX model: equal to `simulate`, step-by-step use, pause and resume, swapping the SRT;
- Conway's Game of Life as a homogeneous special case;
- the optimisers: the exact mutation counts, both acceptance rules, stopping at a target, determinism, replay of the
  recorded mutations, and batched runs equal to single runs;
- the bit-packed engine against the unpacked simulator on lattices that do and do not fill whole words, and its
  evolution (with and without restart, both backends) against the unpacked one;
- the CPU baselines used in the benchmarks: each computes exactly what the package computes.

CI runs the suite on every push.

## Benchmarks

```bash
STICA_MEMCAP=14G benchmarks/run_all.sh      # every benchmark below, one implementation at a time
```

**Implementations compared** (all checked equal in the tests; `benchmarks/baselines.py`):
- **NumPy**, vectorised over the lattice (and over the batch where there is one).
- **CellPyLib** [3], the generic way to run an inhomogeneous CA in an existing Python CA library: a rule callable that
  receives the cell index and time step.
- **numba** [4], compiled loops: the closest analogue to Wolfram's own `FunctionCompile` code. It runs on one core,
  and also on all 20 threads, parallelised over runs or over the rows of a step.
- **numba, bit-packed** (64 cells per word): the CPU counterpart of the packed engine, for the evolution
  benchmarks. Each run stops simulating a candidate as soon as its pattern dies.
- **stica** on the CPU (XLA) and on the GPU.

**Method** (`benchmarks/scaling.py`):
- Each condition is compiled in a separate call that is not recorded.
- It is then run for a second to let clocks settle.
- Seven repetitions follow, each a loop of calls lasting at least 0.2 s.
- Every repetition is saved to CSV, and a JSON file records the software and hardware.

**Results.** NVIDIA RTX 4000 Ada (a desktop GPU that also drives the display, under WSL2) and Intel Xeon w5-2445
(10 cores, 20 threads). Medians, in cell updates per second unless stated. Full tables: `benchmarks/results/
scaling_tables.md`; plot: `benchmarks/results/scaling.png`.

| Workload | NumPy | CellPyLib | numba, 1 core | numba, 20 threads | stica, CPU | stica, GPU |
|---|---:|---:|---:|---:|---:|---:|
| 1D, one 64 × 64 rule array | 5.0e6 | 1.7e6 | **7.2e8** | — | 2.4e7 | 7.0e6 |
| 1D, one 8192 × 8192 rule array | 1.2e8 | 2.0e6 | 7.2e8 | — | 4.6e8 | **4.0e9** |
| 1D, 32,768 rule arrays of 64 × 64 | 8.0e7 | 1.7e6 | 7.0e8 | 4.8e9 | 4.0e8 | **2.9e10** |
| 2D, 2 states, 256 × 256, 256 steps | 3.9e7 | 1.8e5 | 2.2e8 | 6.0e8 | 2.0e8 | **4.4e9** |
| 2D, 4 states, 128 × 128, 128 steps | 2.5e7 | 1.8e5 | 2.6e7 | 1.1e8 | 3.9e7 | **1.3e9** |

**Evolution.** Wolfram's lifetime task, at his size (64 × 31, lifetime 50) and at 100× the cells (640 × 311,
lifetime 500). Best batch size for each entry.

| Workload | numba, 1 core | numba, 20 threads | stica, GPU, XLA | stica, GPU, fused kernel |
|---|---:|---:|---:|---:|
| 64 × 31, unpacked: solved runs / s | 273 | 2,900 | 2,440 (7,240 with restart) | — |
| 64 × 31, bit-packed: solved runs / s | 1,270 | 17,400 | 32,900 | **198,000** |
| 64 × 31, bit-packed: candidates / s | 3.0e6 | 3.8e7 | 7.6e7 | **4.6e8** |
| 640 × 311, bit-packed: candidates / s | 5.1e4 | 1.0e6 | 4.5e5 | **5.5e6** |

**How to read these numbers.**
- **One small trajectory.** A compiled CPU loop is fastest. On the GPU every step costs a few microseconds
  (3–15 µs here) whatever its size, so a single small run is limited by latency, not arithmetic.
- **Large batches and 2D lattices.** The GPU is ahead. At the largest sizes measured it is 6–12× faster than numba
  on 20 threads and 20–50× faster than one core.
- **One large 1D lattice.** The GPU is 5.5× faster than one core.
- **Wolfram's evolution.** Three changes took the GPU from 2,440 to 198,000 solved runs per second:
  - restarting runs that succeed, instead of letting them idle for the rest of the budget;
  - bit-packing the arrays;
  - the fused kernel, which removes a fixed launch cost of 7–8 µs per time step.
  
  The CPU baseline gets the same bit-packing and still stops early, and the fused kernel is 11× faster than it at
  Wolfram's size and 5× faster at 100× the cells. Without the fused kernel the GPU is only 1.9× ahead at Wolfram's
  size, and at 100× the cells the bit-packed CPU on 20 threads is 2× faster than it.
- **Provisional numbers.** The GPU also drives the display. Final numbers need a dedicated GPU.

**Comparison with the legacy prototype** (`benchmarks/bench.py`; `benchmarks/results/legacy_tables.md`). This is the
project's own earlier code, not an independent baseline. Most of its time went into converting the whole rule tensor
to float64 and copying it to the GPU on every call.

| Task | Legacy prototype | stica |
|---|---:|---:|
| Simulate 100 × 100, 1000 steps, 2 states | 471 ms | 3.7 ms |
| Simulate 64 × 64, 200 steps, 4 states | 1,210 ms | 1.5 ms |
| Optimisation step, 100 × 100 × 99, 1 run | 319 ms | 0.73 ms |
| Optimisation step, 100 × 100 × 99, 16 runs (per run) | — | 0.24 ms |

## Implementation notes

These choices account for most of the speed and are easy to get wrong in JAX:
- **Unrolled time loop.** On a GPU every iteration of a compiled loop costs tens of microseconds, more than a small
  CA step. The time loop therefore runs 16 steps per iteration, which is 2–8× faster.
- **A barrier between steps.** Each step ends in an optimisation barrier. Without it, XLA fuses consecutive unrolled
  stencil steps into each other. When a whole run fits in one unrolled block, that fusion grows exponentially and
  compilation stalls.
- **Keeping the rule tensor out of the loop carry.** CAX's default loop (`nnx.scan`) carries the whole module, rule
  tensor included, from step to step, and `nnx.jit` returns the module state from every call. With a large rule
  tensor both cost more than the simulation. `STICA` keeps the module as a loop constant, and `STICA.run` uses Flax's
  split/merge with `jax.jit`.
- **Small writes.** Rule tensors are int8. A mutation writes, and if rejected reverts, only the entries it touched.
- **Batching.** Independent runs are batched with `vmap`, which is where the throughput comes from.
- **A fused kernel for small sequential problems.** A 31-cell lattice gives a GPU almost nothing to do per step, so
  the time goes on launching kernels: about 8 µs per step here, under WSL2, whatever the batch size. The Pallas
  kernel (`stica.packed.fused_rollout`) keeps each run's state in registers through all T steps, so a candidate
  evaluation costs one launch for the whole batch.
- **No accidental copies.** A mutation is written into the batch in place and reverted in place. Referring to the
  pre-mutation array afterwards would make XLA copy the whole batch on every iteration.
- **Restarts sized to the batch.**
  - A restart rewrites every run's array. For large batches it is skipped on iterations where no run finished.
  - For small batches it is always done, because on a GPU the conditional costs a device-host round trip, more than
    the rewrite.

## Repository layout

```
src/stica/       rules.py (2D indexing), ca.py (CAX model, simulate), rulearray.py (Wolfram's 1D rule arrays),
                 packed.py (bit-packed binary rule arrays, fused GPU kernel), optimize.py (the shared evolution
                 loop), objectives.py, export.py
tests/           pytest suite, NumPy references, data read off Wolfram's figures
reproduce/       Wolfram's lifetime experiments; extraction of his published rule arrays
benchmarks/      CPU baselines, scaling and legacy benchmarks, recorded results
examples/        command-line example
legacy/          the original implementation (superseded) and its known issues
```

## References

1. S. Wolfram, "What's Really Going On in Machine Learning? Some Minimal Models," *Stephen Wolfram Writings*, 2024.
   https://writings.stephenwolfram.com/2024/08/whats-really-going-on-in-machine-learning-some-minimal-models/
2. M. Faldor and A. Cully, "CAX: Cellular Automata Accelerated in JAX," ICLR 2025. arXiv:2410.02651.
3. L. M. Antunes, "CellPyLib: A Python Library for working with Cellular Automata," *Journal of Open Source
   Software* 6(67), 3608 (2021).
4. S. K. Lam, A. Pitrou and S. Seibert, "Numba: a LLVM-based Python JIT compiler," *Proc. Second Workshop on the
   LLVM Compiler Infrastructure in HPC* (2015).

## License

Apache License 2.0; see [LICENSE](LICENSE).
