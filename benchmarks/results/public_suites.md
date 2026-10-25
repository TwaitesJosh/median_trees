# Public benchmark suites

`python benchmarks/benchmark_suites.py --suites cc18,grinsztajn,tabarena,mteb` — official OpenML splits
(repeat 0, 3 folds), training sets capped at 50,000 rows, every model at its library defaults.
`mt` = median trees (the package defaults), `mt-probe` = `MedianTreeClassifier(probe=True)`,
`hgb` = scikit-learn `HistGradientBoostingClassifier`, `et` = `ExtraTreesClassifier(256)`.
W/T/L = datasets won / tied (within 0.1 points) / lost against `hgb`.

### OpenML-CC18: 69 datasets

| method | mean acc | mean rank | normalized acc | W/T/L vs hgb | median Δacc vs hgb | mean AUC (binary, 33) | mean log-loss (multiclass, 36) | median fit s |
|---|---|---|---|---|---|---|---|---|
| mt-probe | 87.10 | 2.30 | 0.597 | 32/9/28 | +0.01 | 0.8917 | 0.3664 | 0.68 |
| mt | 86.91 | 2.43 | 0.587 | 33/7/29 | +0.01 | 0.8854 | 0.3878 | 0.37 |
| hgb | 86.74 | 2.50 | 0.551 | – | – | 0.8869 | 0.4438 | 0.44 |
| et | 86.19 | 2.77 | 0.435 | 28/4/37 | −0.18 | 0.8784 | 1.2764 | 0.16 |

### Grinsztajn et al. 2022, classification (suites 337 + 334): 15 datasets finished

| method | mean acc | mean rank | normalized acc | W/T/L vs hgb | median Δacc vs hgb | mean AUC | median fit s |
|---|---|---|---|---|---|---|---|
| hgb | 76.51 | 1.97 | 0.754 | – | – | 0.8325 | 0.17 |
| mt | 76.44 | 2.37 | 0.686 | 6/3/6 | −0.02 | 0.8310 | 1.64 |
| mt-probe | 76.50 | 2.60 | 0.596 | 4/1/10 | −0.20 | 0.8300 | 2.29 |
| et | 75.88 | 3.07 | 0.290 | 4/0/11 | −0.82 | 0.8202 | 0.25 |

### TabArena-v0.1, classification tasks (suite 457): 34 datasets

| method | mean acc | mean rank | normalized acc | W/T/L vs hgb | median Δacc vs hgb | mean AUC (binary, 27) | mean log-loss (multiclass, 7) | median fit s |
|---|---|---|---|---|---|---|---|---|
| mt | 86.79 | 2.12 | 0.739 | 16/8/10 | +0.08 | 0.8324 | 0.3356 | 0.57 |
| mt-probe | 86.73 | 2.21 | 0.676 | 15/10/9 | +0.02 | 0.8322 | 0.3250 | 0.84 |
| hgb | 86.72 | 2.47 | 0.543 | – | – | 0.8262 | 0.4539 | 0.18 |
| et | 86.09 | 3.21 | 0.278 | 7/7/20 | −0.37 | 0.8085 | 0.7272 | 0.18 |

### MTEB English classification, all-MiniLM-L6-v2 embeddings: 9 datasets finished

| method | mean acc | mean rank | W/T/L vs hgb | mean log-loss (multiclass, 5) | median fit s |
|---|---|---|---|---|---|
| mlp | 84.18 | 2.44 | 9/0/0 | 0.5312 | 4.90 |
| mt-probe | 83.79 | 2.50 | 9/0/0 | 0.5284 | 4.25 |
| mt+raw | 83.80 | 2.61 | 9/0/0 | 0.5352 | 3.43 |
| linear | 83.58 | 2.89 | 9/0/0 | 0.5503 | 8.02 |
| knn | 80.49 | 5.11 | 4/0/5 | 1.4993 | 0.18 |
| hgb | 69.59 | 5.44 | – | 7.4581 | 1.84 |

Missing: Grinsztajn 8 of 23 tasks, CC18 `sick`, MTEB 3 of 12 datasets did not finish (load or fit errors).
