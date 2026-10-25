# Frozen embeddings

`python benchmarks/embeddings_experiment.py` — text: all-MiniLM-L6-v2 sentence embeddings (384-d);
images: ImageNet ResNet-18 penultimate layer (512-d). 3 seeds, 70/15/15 splits, test accuracy in %.
`linear` = logistic regression, C tuned on validation; `mlp` = scikit-learn MLP (512 hidden, early
stopping); `knn` = k tuned on validation; `mt+raw` = `MedianTreeClassifier(raw_features=True)`.

| dataset | classes | linear | knn | mlp | hgb | mt | mt+raw |
|---|---|---|---|---|---|---|---|
| ag_news | 4 | 89.28 | **91.02** | 90.77 | 89.12 | 90.08 | 90.28 |
| emotion | 6 | 69.01 | 70.52 | **73.56** | 66.46 | 68.89 | 71.08 |
| 20newsgroups | 20 | 71.04 | **75.76** | 71.09 | 68.26 | 72.30 | 72.21 |
| cifar10 | 10 | 87.92 | 85.25 | 88.12 | 86.62 | 87.54 | **88.63** |
| cifar100 | 100 | 68.85 | 61.26 | 67.82 | 29.31 | 65.66 | **69.47** |
| fashion-mnist | 10 | 89.20 | 87.92 | 89.80 | 88.61 | 89.47 | **90.06** |

Median fit time (s): mt+raw 2–25, linear 13–102, mlp 3–26, hgb 3–12.
