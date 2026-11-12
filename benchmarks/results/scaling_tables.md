wrote benchmarks/results/scaling.png

### sim1d: median cell_updates_per_s

| implementation | size=32 | size=64 | size=128 | size=256 | size=512 | size=1024 | size=2048 | size=4096 | size=8192 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CellPyLib, rule callable | 1.43e+06 | 1.67e+06 | 1.84e+06 | 1.95e+06 | 1.95e+06 | 1.94e+06 | 1.97e+06 | 1.97e+06 | 1.97e+06 |
| stica, CPU (XLA) | 7.37e+06 | 2.42e+07 | 8.2e+07 | 1.36e+08 | 1.64e+08 | 1.32e+08 | 2.26e+08 | 3.52e+08 | 4.64e+08 |
| stica, GPU | 3.25e+06 | 6.98e+06 | 1.05e+07 | 4.15e+07 | 1.52e+08 | 4.93e+08 | 1.08e+09 | 2.09e+09 | 3.98e+09 |
| numba loop, 1 core | 4.56e+08 | 7.21e+08 | 8.75e+08 | 9.28e+08 | 9.43e+08 | 9.33e+08 | 9.51e+08 | 9.35e+08 | 7.22e+08 |
| NumPy, vectorised | 2.56e+06 | 5.04e+06 | 9.56e+06 | 1.83e+07 | 3.16e+07 | 5.41e+07 | 7.61e+07 | 1e+08 | 1.2e+08 |

### sim2d: median cell_updates_per_s

| implementation | size=16 | size=32 | size=64 | size=128 | size=256 |
|---|---:|---:|---:|---:|---:|
| CellPyLib, rule callable, n=2 | 1.32e+05 | 1.6e+05 | 1.74e+05 | 1.8e+05 | 1.84e+05 |
| CellPyLib, rule callable, n=3 | 1.31e+05 | 1.56e+05 | 1.75e+05 | 1.81e+05 | — |
| CellPyLib, rule callable, n=4 | 1.3e+05 | 1.58e+05 | 1.71e+05 | 1.77e+05 | — |
| stica, CPU (XLA), n=2 | 4.25e+07 | 7.9e+07 | 1.49e+08 | 1.77e+08 | 2.02e+08 |
| stica, CPU (XLA), n=3 | 1.67e+07 | 4.65e+07 | 5.45e+07 | 7.55e+07 | — |
| stica, CPU (XLA), n=4 | 1.24e+07 | 3.46e+07 | 3.12e+07 | 3.94e+07 | — |
| stica, GPU, n=2 | 1.17e+07 | 2.26e+07 | 1.67e+08 | 1.41e+09 | 4.36e+09 |
| stica, GPU, n=3 | 7.19e+06 | 2.17e+07 | 1.71e+08 | 1.42e+09 | — |
| stica, GPU, n=4 | 7.49e+06 | 2.14e+07 | 1.75e+08 | 1.26e+09 | — |
| numba loop, 1 core, n=2 | 1.55e+08 | 2.07e+08 | 2.16e+08 | 2.18e+08 | 2.22e+08 |
| numba loop, 1 core, n=3 | 8.38e+07 | 1.24e+08 | 7.09e+07 | 5.29e+07 | — |
| numba loop, 1 core, n=4 | 2.47e+07 | 6.11e+07 | 5.21e+07 | 2.6e+07 | — |
| numba loop, all cores, n=2 | 8.84e+06 | 3.29e+07 | 1.08e+08 | 2.91e+08 | 5.98e+08 |
| numba loop, all cores, n=3 | 7.93e+06 | 2.88e+07 | 8.47e+07 | 1.63e+08 | — |
| numba loop, all cores, n=4 | 6.13e+06 | 2.38e+07 | 6.55e+07 | 1.07e+08 | — |
| NumPy, vectorised, n=2 | 7.93e+06 | 1.99e+07 | 3.2e+07 | 3.8e+07 | 3.89e+07 |
| NumPy, vectorised, n=3 | 7.4e+06 | 1.89e+07 | 2.47e+07 | 2.72e+07 | — |
| NumPy, vectorised, n=4 | 6.05e+06 | 1.75e+07 | 2.33e+07 | 2.45e+07 | — |

### sim1d_batch: median cell_updates_per_s

| implementation | batch=1 | batch=8 | batch=64 | batch=512 | batch=4096 | batch=32768 |
|---|---:|---:|---:|---:|---:|---:|
| CellPyLib, rule callable | 1.67e+06 | 1.66e+06 | 1.67e+06 | 1.68e+06 | 1.68e+06 | 1.68e+06 |
| stica, CPU (XLA) | 3.48e+07 | 6.62e+07 | 1.06e+08 | 2.61e+08 | 4.16e+08 | 4.01e+08 |
| stica, GPU | 3.28e+06 | 2.02e+07 | 1.8e+08 | 1.4e+09 | 1.16e+10 | 2.87e+10 |
| numba loop, 1 core | 6.39e+08 | 6.96e+08 | 7.1e+08 | 7.03e+08 | 7.07e+08 | 7.01e+08 |
| numba loop, all cores | 1.18e+08 | 6.45e+08 | 2.2e+09 | 4.56e+09 | 6.03e+09 | 4.78e+09 |
| NumPy, vectorised | 5.01e+06 | 3.05e+07 | 1.07e+08 | 1.32e+08 | 1.07e+08 | 8.03e+07 |

### evo1d: median candidates_per_s

| implementation | batch=1 | batch=64 | batch=1024 | batch=8192 | batch=32768 |
|---|---:|---:|---:|---:|---:|
| stica, CPU (XLA): packed, restart | 6.49e+04 | 1.44e+06 | — | — | — |
| stica, CPU (XLA): unpacked | 1.99e+04 | 5.52e+04 | — | — | — |
| stica, CPU (XLA): unpacked, restart | 1.95e+04 | 5.44e+04 | — | — | — |
| stica, GPU: packed, restart | 5e+03 | 3.32e+05 | 5.01e+06 | 3.24e+07 | 7.6e+07 |
| stica, GPU: packed, restart, fused kernel | 2.85e+04 | 1.8e+06 | 3.14e+07 | 1.8e+08 | 4.59e+08 |
| stica, GPU: unpacked | 6.27e+03 | 4.42e+05 | 6.31e+06 | 2.45e+07 | 1.86e+07 |
| stica, GPU: unpacked, restart | 8.07e+03 | 4.61e+05 | 6.36e+06 | 1.67e+07 | 1.38e+07 |
| numba loop, 1 core: packed | 7.59e+05 | 3.02e+06 | — | — | — |
| numba loop, 1 core: unpacked | 3.44e+05 | 6.5e+05 | — | — | — |
| numba loop, all cores: packed | — | 1.91e+07 | 3.43e+07 | 3.8e+07 | — |
| numba loop, all cores: unpacked | — | 3.95e+06 | 5.78e+06 | 6.32e+06 | — |

### evo1d: median solved_per_s

| implementation | batch=1 | batch=64 | batch=1024 | batch=8192 | batch=32768 |
|---|---:|---:|---:|---:|---:|
| stica, CPU (XLA): packed, restart | 26.3 | 629 | — | — | — |
| stica, CPU (XLA): unpacked | 1.99 | 5.52 | — | — | — |
| stica, CPU (XLA): unpacked, restart | 9.71 | 24 | — | — | — |
| stica, GPU: packed, restart | 2 | 144 | 2.18e+03 | 1.4e+04 | 3.29e+04 |
| stica, GPU: packed, restart, fused kernel | 9.55 | 752 | 1.34e+04 | 7.79e+04 | 1.98e+05 |
| stica, GPU: unpacked | 0.627 | 44.2 | 631 | 2.44e+03 | 1.86e+03 |
| stica, GPU: unpacked, restart | 3.99 | 211 | 2.79e+03 | 7.24e+03 | 5.95e+03 |
| numba loop, 1 core: packed | 409 | 1.27e+03 | — | — | — |
| numba loop, 1 core: unpacked | 214 | 273 | — | — | — |
| numba loop, all cores: packed | — | 8e+03 | 1.6e+04 | 1.74e+04 | — |
| numba loop, all cores: unpacked | — | 1.89e+03 | 2.69e+03 | 2.9e+03 | — |

### evo1d_large: median candidates_per_s

| implementation | batch=1 | batch=64 | batch=512 | batch=2048 | batch=8192 | batch=32768 |
|---|---:|---:|---:|---:|---:|---:|
| stica, CPU (XLA): packed, restart | 4.23e+03 | 9.41e+03 | — | — | — | — |
| stica, CPU (XLA): unpacked | 2.16e+03 | 1.48e+03 | — | — | — | — |
| stica, CPU (XLA): unpacked, restart | 2.08e+03 | 1.42e+03 | — | — | — | — |
| stica, GPU: packed, restart | 518 | 3.51e+04 | 2.48e+05 | 4.2e+05 | 4.45e+05 | — |
| stica, GPU: packed, restart, fused kernel | 3.14e+03 | 3.11e+05 | 2.11e+06 | 1.72e+06 | 2.04e+06 | 5.49e+06 |
| stica, GPU: unpacked | 928 | 4.9e+04 | 1.78e+05 | 1.52e+05 | — | — |
| stica, GPU: unpacked, restart | 958 | 4.81e+04 | 1.33e+05 | 1.39e+05 | — | — |
| numba loop, 1 core: packed | 2.89e+04 | 5.07e+04 | — | — | — | — |
| numba loop, 1 core: unpacked | 4.82e+03 | 6.43e+03 | — | — | — | — |
| numba loop, all cores: packed | — | 6.47e+05 | 9.47e+05 | 1.01e+06 | 1.03e+06 | — |
| numba loop, all cores: unpacked | — | 4.9e+04 | 5.54e+04 | 5.69e+04 | 5.75e+04 | — |
