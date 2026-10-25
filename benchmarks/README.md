# Benchmarks

| script | what it runs | results |
|---|---|---|
| `benchmark_suites.py` | OpenML-CC18, Grinsztajn, TabArena and MTEB on official splits | [public_suites.md](results/public_suites.md) |
| `embeddings_experiment.py` | heads on frozen text and image embeddings | [embeddings.md](results/embeddings.md) |
| `overnight.py` | 13 tests: learning curves, regression, calibration, imbalance, stress, shift, many classes, OOD, stability, speed, online, LLM probes | [overnight_report.md](results/overnight_report.md) |

Install the extras first: `pip install -e ".[bench]"` (plus `torch transformers datasets sentence-transformers torchvision` for the embedding and probe tests). Every script is resumable and writes CSVs to `results/`; `--local` runs a short offline smoke test.

Note: `overnight.py` also contains an experimental median-tree *regressor*. Its `mt+raw` variant scored R² 0.41 against 0.71 for the plain version, which points to a bug in the raw-feature term, so the regressor is not part of the package yet.
