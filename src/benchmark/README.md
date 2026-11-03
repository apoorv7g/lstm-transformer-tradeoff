# src/benchmark

**Status: to do.** Comes after training.

Will measure how much compute each model needs on the CPU, then produce the final report.

| Planned file | What it will measure or do |
|---|---|
| `latency.py` | Time per prediction (batch 1 and 32, 1 and 8 CPU threads) |
| `memory.py` | Peak RAM use |
| `flops.py` | Number of operations per prediction |
| `run_grid.py` | Runs all of the above for every model, each in a fresh process so results do not interfere |
| `report.py` | Summary tables, statistical tests and accuracy-vs-speed plots |

Planned command: `python -m src.benchmark.run_grid`. Settings: `configs/benchmark.yaml`.
Full method: `docs/benchmarking_protocol.md`.
