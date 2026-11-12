
### Simulation (ms per trajectory)

| grid | steps | states | implementation | median | IQR | cell updates / s |
|---:|---:|---:|---|---:|---:|---:|
| 32 | 200 | 2 | legacy: transition | 24.8 | 24.7-25.7 | 8.25e+06 |
| 32 | 200 | 2 | stica: STICA.run (CAX) | 2.23 | 2.18-2.27 | 9.16e+07 |
| 32 | 200 | 2 | stica: simulate | 1.29 | 1.28-1.32 | 1.58e+08 |
| 64 | 200 | 3 | legacy: transition | 295 | 294-298 | 2.78e+06 |
| 64 | 200 | 3 | stica: STICA.run (CAX) | 2.65 | 2.62-2.67 | 3.09e+08 |
| 64 | 200 | 3 | stica: simulate | 1.47 | 1.44-1.49 | 5.57e+08 |
| 64 | 200 | 4 | legacy: transition | 1.21e+03 | 1.21e+03-1.22e+03 | 6.77e+05 |
| 64 | 200 | 4 | stica: STICA.run (CAX) | 2.77 | 2.75-2.81 | 2.95e+08 |
| 64 | 200 | 4 | stica: simulate | 1.52 | 1.5-1.53 | 5.41e+08 |
| 64 | 500 | 2 | legacy: transition | 117 | 116-120 | 1.76e+07 |
| 64 | 500 | 2 | stica: STICA.run (CAX) | 2.77 | 2.75-2.78 | 7.40e+08 |
| 64 | 500 | 2 | stica: simulate | 1.66 | 1.66-1.68 | 1.23e+09 |
| 100 | 1000 | 2 | legacy: transition | 471 | 463-472 | 2.12e+07 |
| 100 | 1000 | 2 | stica: STICA.run (CAX) | 4.56 | 4.53-4.69 | 2.19e+09 |
| 100 | 1000 | 2 | stica: simulate | 3.65 | 3.51-3.73 | 2.74e+09 |

### Optimisation (ms per iteration and chain)

| grid | steps | states | implementation | median | IQR | cell updates / s |
|---:|---:|---:|---|---:|---:|---:|
| 32 | 31 | 2 | legacy: Optimizer.step | 37.9 | 37.9-39.3 | 8.37e+05 |
| 32 | 31 | 2 | stica: run_ga_batch x1 | 0.421 | 0.42-0.423 | 7.54e+07 |
| 32 | 31 | 2 | stica: run_ga_batch x16 | 0.0389 | 0.0388-0.0393 | 8.17e+08 |
| 64 | 63 | 2 | legacy: Optimizer.step | 135 | 134-135 | 1.92e+06 |
| 64 | 63 | 2 | stica: run_ga_batch x1 | 0.57 | 0.569-0.57 | 4.53e+08 |
| 64 | 63 | 2 | stica: run_ga_batch x16 | 0.0923 | 0.0908-0.0971 | 2.80e+09 |
| 100 | 99 | 2 | legacy: Optimizer.step | 319 | 318-323 | 3.10e+06 |
| 100 | 99 | 2 | stica: run_ga_batch x1 | 0.731 | 0.73-0.744 | 1.35e+09 |
| 100 | 99 | 2 | stica: run_ga_batch x16 | 0.236 | 0.234-0.236 | 4.20e+09 |
