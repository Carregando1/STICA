
### sim1d: median cell_updates_per_s

| implementation | size=32 | size=64 | size=128 | size=256 | size=512 | size=1024 | size=2048 | size=4096 | size=8192 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CellPyLib, rule callable | 1.43e+06 | 1.67e+06 | 1.84e+06 | 1.95e+06 | 1.95e+06 | 1.94e+06 | 1.97e+06 | 1.97e+06 | 1.97e+06 |
| stica, CPU (XLA) | 7.37e+06 | 2.42e+07 | 8.20e+07 | 1.36e+08 | 1.64e+08 | 1.32e+08 | 2.26e+08 | 3.52e+08 | 4.64e+08 |
| stica, GPU | 3.25e+06 | 6.98e+06 | 1.05e+07 | 4.15e+07 | 1.52e+08 | 4.93e+08 | 1.08e+09 | 2.09e+09 | 3.98e+09 |
| numba loop, 1 core | 4.56e+08 | 7.21e+08 | 8.75e+08 | 9.28e+08 | 9.43e+08 | 9.33e+08 | 9.51e+08 | 9.35e+08 | 7.22e+08 |
| NumPy, vectorised | 2.56e+06 | 5.04e+06 | 9.56e+06 | 1.83e+07 | 3.16e+07 | 5.41e+07 | 7.61e+07 | 1.00e+08 | 1.20e+08 |

### sim2d: median cell_updates_per_s

| implementation | size=16 | size=32 | size=64 | size=128 | size=256 |
|---|---:|---:|---:|---:|---:|
| CellPyLib, rule callable, n=2 | 1.32e+05 | 1.60e+05 | 1.74e+05 | 1.80e+05 | 1.84e+05 |
| CellPyLib, rule callable, n=3 | 1.31e+05 | 1.56e+05 | 1.75e+05 | 1.81e+05 | — |
| CellPyLib, rule callable, n=4 | 1.30e+05 | 1.58e+05 | 1.71e+05 | 1.77e+05 | — |
| stica, CPU (XLA), n=2 | 4.25e+07 | 7.90e+07 | 1.49e+08 | 1.77e+08 | 2.02e+08 |
| stica, CPU (XLA), n=3 | 1.67e+07 | 4.65e+07 | 5.45e+07 | 7.55e+07 | — |
| stica, CPU (XLA), n=4 | 1.24e+07 | 3.46e+07 | 3.12e+07 | 3.94e+07 | — |
| stica, GPU, n=2 | 1.17e+07 | 2.26e+07 | 1.67e+08 | 1.41e+09 | 4.36e+09 |
| stica, GPU, n=3 | 7.19e+06 | 2.17e+07 | 1.71e+08 | 1.42e+09 | — |
| stica, GPU, n=4 | 7.49e+06 | 2.14e+07 | 1.75e+08 | 1.26e+09 | — |
| numba loop, 1 core, n=2 | 1.55e+08 | 2.07e+08 | 2.16e+08 | 2.18e+08 | 2.22e+08 |
| numba loop, 1 core, n=3 | 8.38e+07 | 1.24e+08 | 7.09e+07 | 5.29e+07 | — |
| numba loop, 1 core, n=4 | 2.47e+07 | 6.11e+07 | 5.21e+07 | 2.60e+07 | — |
| numba loop, all cores, n=2 | 8.84e+06 | 3.29e+07 | 1.08e+08 | 2.91e+08 | 5.98e+08 |
| numba loop, all cores, n=3 | 7.93e+06 | 2.88e+07 | 8.47e+07 | 1.63e+08 | — |
| numba loop, all cores, n=4 | 6.13e+06 | 2.38e+07 | 6.55e+07 | 1.07e+08 | — |
| NumPy, vectorised, n=2 | 7.93e+06 | 1.99e+07 | 3.20e+07 | 3.80e+07 | 3.89e+07 |
| NumPy, vectorised, n=3 | 7.40e+06 | 1.89e+07 | 2.47e+07 | 2.72e+07 | — |
| NumPy, vectorised, n=4 | 6.05e+06 | 1.75e+07 | 2.33e+07 | 2.45e+07 | — |

### sim1d_batch: median cell_updates_per_s

| implementation | batch=1 | batch=8 | batch=64 | batch=512 | batch=4096 | batch=32768 |
|---|---:|---:|---:|---:|---:|---:|
| CellPyLib, rule callable | 1.67e+06 | 1.66e+06 | 1.67e+06 | 1.68e+06 | 1.68e+06 | 1.68e+06 |
| stica, CPU (XLA) | 3.48e+07 | 6.62e+07 | 1.06e+08 | 2.61e+08 | 4.16e+08 | 4.01e+08 |
| stica, GPU | 3.28e+06 | 2.02e+07 | 1.80e+08 | 1.40e+09 | 1.16e+10 | 2.87e+10 |
| numba loop, 1 core | 6.39e+08 | 6.96e+08 | 7.10e+08 | 7.03e+08 | 7.07e+08 | 7.01e+08 |
| numba loop, all cores | 1.18e+08 | 6.45e+08 | 2.20e+09 | 4.56e+09 | 6.03e+09 | 4.78e+09 |
| NumPy, vectorised | 5.01e+06 | 3.05e+07 | 1.07e+08 | 1.32e+08 | 1.07e+08 | 8.03e+07 |

### evo1d: median candidates_per_s

| implementation | batch=1 | batch=64 | batch=1024 | batch=8192 |
|---|---:|---:|---:|---:|
| stica, CPU (XLA) | 1.90e+04 | 5.49e+04 | — | — |
| stica, GPU | 6.96e+03 | 4.55e+05 | 6.40e+06 | 2.41e+07 |
| numba loop, 1 core | 5.56e+05 | 6.56e+05 | — | — |
| numba loop, all cores | 5.45e+05 | 4.45e+06 | 5.80e+06 | — |
