# STICA: spacetime-inhomogeneous cellular automata in JAX

STICA is a cellular automaton in which **every cell at every time step can use its own update rule**. Stephen Wolfram
introduced this model as *rule arrays*, "'spacetime-inhomogeneous' generalizations of cellular automata", in his
minimal models of machine learning [1]. Wolfram's rule arrays are one-dimensional and binary, and each cell chooses
between two fixed rules.

This package generalises them and runs them on accelerators:

- **2D** lattices with the Moore neighbourhood and **N states**.
- A **full outer-totalistic lookup table per cell and time step**: the rule sees the cell's own state and how many
  neighbours are in each state.
- **Rule sharing**: the rule tensor can be shared across time, across space, or both. Sharing across both gives an
  ordinary homogeneous CA.
- A **CAX model** [2] in which time is carried in a state channel, so STICA is an ordinary autonomous CA.
- **Adaptive evolution** of initial conditions and rules (random mutations, kept unless the objective gets worse),
  running entirely on the GPU and batched over independent runs.

> The original prototype is kept in [`legacy/`](legacy/README.md), together with a list of the problems it had.

## Installation

```bash
pip install -e ".[dev]"            # CPU
pip install -e ".[dev,cuda12]"     # NVIDIA GPU (CUDA 12)
```

Tested with Python 3.11–3.13, jax 0.6.2, flax 0.10.7 and cax 0.2.1. These versions are pinned in `pyproject.toml`.

## Quick start

```python
import jax
import stica

# simulate: random IC and rule tensor for 3 states on a 32x32 grid, 50 steps
n, G, T = 3, 32, 50
key_ic, key_srt = jax.random.split(jax.random.PRNGKey(0))
ic = jax.random.randint(key_ic, (G, G), 0, n)
srt = jax.random.randint(key_srt, (T, G, G, stica.num_rules(n)), 0, n)
traj = stica.simulate(ic, srt, n)                 # (T + 1, G, G): the IC and every later step

# the same model as a CAX cellular automaton
ca = stica.STICA(srt, num_states=n)
traj = ca.run(ic)
state, cells = ca.step(stica.initial_state(ic))  # one step at a time; state[..., 1] is the time

# adaptive evolution: 16 independent runs in parallel
res = stica.run_ga_batch(jax.random.PRNGKey(1), 16, grid_size=32, n=2, iterations=1000,
                         ic_count=32, srt_count=32 * 32, objective="surface")
print(res.best)                                   # best objective value of each run
```

`python examples/run_ga.py --help` runs a batch from the command line and saves the results and their metadata.

## The model

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

`run_ga` and `run_ga_batch` implement a (1+1) evolutionary strategy, i.e. a stochastic hill climber. Each iteration:

1. picks exactly `ic_count` distinct IC cells and `srt_count` distinct SRT entries;
2. moves each one to a uniformly chosen different state;
3. simulates the candidate and evaluates the objective;
4. keeps the candidate if the objective did not increase.

Notes:
- Keeping ties (`accept_ties=True`, the default) follows Wolfram's adaptive evolution [1].
- `accept_ties=False` keeps only strict improvements, as the legacy code did.
- The whole loop is one compiled `lax.scan`. Only the mutated entries are written, and they are reverted if the
  candidate is rejected.
- `run_ga_batch` runs independent chains in parallel with `vmap`.
- Results are reproducible from the PRNG key.
- `record_mutations=True` returns every mutation (position, old value, new value).

**Objectives.** These are minimised, and they treat the trajectory as a 3D voxel lattice, with time as the first axis.
- Built-in: `"surface"` (negative exposed surface area) and `"surface_to_volume"`.
- Any JAX function from the trajectory to a scalar also works.
- `stica.lattice_to_stl` exports a trajectory as a voxel mesh.

## Tests

```bash
JAX_PLATFORMS=cpu pytest -q
```

The suite has 60 tests. They check:
- simulation, bit for bit, against a loop-based NumPy reference: 2–6 states, dense neighbourhoods, and every
  rule-sharing level;
- the rule index exhaustively, up to 8 states;
- the CAX model: equal to `simulate`, step-by-step use, pause and resume, swapping the SRT;
- Conway's Game of Life as a homogeneous special case;
- the optimiser: the exact mutation counts, both acceptance rules, determinism, and batched runs equal to single runs.

CI runs the suite on every push.

## Benchmarks

```bash
python benchmarks/bench.py --impl stica      # this package
python benchmarks/bench.py --impl legacy     # the original prototype, for reference
python benchmarks/summarize.py benchmarks/results/*.csv
```

**Method.**
- Timing uses `time.perf_counter`. Each condition is compiled in a separate warm-up call that is not recorded.
- The device is synchronised before the clock stops.
- Every repetition is saved to CSV, and a JSON file records the software and hardware.

**Results** (`benchmarks/results/`): NVIDIA RTX 4000 Ada (a desktop GPU that also drives the display, under WSL2),
jax 0.6.2, medians over 7 repetitions.

| Task | Legacy prototype | stica | stica, cell updates/s |
|---|---:|---:|---:|
| Simulate 100×100, 1000 steps, 2 states | 474 ms | 28.6 ms | 3.5×10⁸ |
| Simulate 64×64, 200 steps, 3 states | 290 ms | 2.2 ms | 3.7×10⁸ |
| Simulate 64×64, 200 steps, 4 states | out of GPU memory | 2.2 ms | 3.7×10⁸ |
| Optimisation step, 100×100×99, 1 run | 357 ms | 3.5 ms | 2.8×10⁸ |
| Optimisation step, 100×100×99, 16 runs (per run) | — | 0.32 ms | 3.1×10⁹ |

**How to read these numbers.**
- The legacy column is the project's own earlier prototype, not an independent baseline.
- Most of its time went into converting the whole rule tensor to float64 and copying it to the GPU on every call.
- A single trajectory on the GPU is limited by per-step overhead (roughly 10–50 µs per step), not by arithmetic.
  Throughput comes from batching independent runs.

## Repository layout

```
src/stica/       rules.py (indexing), ca.py (CAX model, simulate), optimize.py, objectives.py, export.py
tests/           pytest suite and the NumPy reference implementation
benchmarks/      timing harness, summaries, recorded results
examples/        command-line example
legacy/          the original implementation (superseded) and its known issues
```

## References

1. S. Wolfram, "What's Really Going On in Machine Learning? Some Minimal Models," *Stephen Wolfram Writings*, 2024.
   https://writings.stephenwolfram.com/2024/08/whats-really-going-on-in-machine-learning-some-minimal-models/
2. M. Faldor and A. Cully, "CAX: Cellular Automata Accelerated in JAX," ICLR 2025. arXiv:2410.02651.

## License

Apache License 2.0; see [LICENSE](LICENSE).
