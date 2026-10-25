"""
Median trees on embeddings: can they beat the usual linear probe?

Step 1 computes embeddings once and caches them (cache/<dataset>__<model>.npz):
  text    sentence-transformers all-MiniLM-L6-v2 (384-d), on
            ag_news (4 classes), emotion (6), 20newsgroups (20), banking77 (77)
  images  ImageNet-pretrained torchvision ResNet-18, penultimate layer (512-d), on
            cifar10 (10 classes), cifar100 (100), fashion (10)
Every embedding is L2-normalised (cosine geometry), for every method.

Step 2 compares, on the same 70/15/15 splits (3 seeds by default):
  linear      logistic regression, C chosen on validation: the standard "linear probe"
  knn         k-nearest neighbours (cosine), k chosen on validation
  mlp         one hidden layer (512), early stopping: the usual next step after linear
  hgb         HistGradientBoostingClassifier, defaults
  mt          median trees, your fixed configuration (supervised oblique + dropout + tail)
  mt+raw      ... plus the raw embedding in the linear head (contains a linear probe)
  mt+raw+rot  ... with a random rotation of the embedding first (embedding axes carry no
              individual meaning, so axis-aligned and pairwise splits may prefer new axes)
  mt+raw+rot+ada  ... plus Adagrad in the head
  mt-probe    the automatic probe (probe=True): picks depth, oblique splits, raw features,
              tree count and dropout itself from a quick validation micro-forest
  mt-probe+rot    ... after the same random rotation

Requirements:  pip install sentence-transformers datasets torch torchvision
Keep blog_benchmark.py and median_trees.py in the same folder; delete __pycache__ first.

Usage:
  python embeddings_experiment.py                           # everything
  python embeddings_experiment.py --datasets banking77,cifar100
  python embeddings_experiment.py --methods linear,mt+raw,mt+raw+rot --seeds 5
  python embeddings_experiment.py --embed-only              # just build the cache
  python embeddings_experiment.py --local                   # offline test, synthetic data
Images: CIFAR-10 and Fashion-MNIST are read from scikit-learn's OpenML cache (already on
disk if you ran the earlier benchmarks), CIFAR-100 from Hugging Face; torchvision's slow
original download is only a fallback. Embedding is much faster with a GPU (used
automatically); on CPU expect roughly 10-30 minutes per image dataset, once.
"""

import argparse
import csv
import os
import time
import warnings

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

from median_trees import MedianTreeClassifier


def split(X, y, seed):
    """70 / 15 / 15 stratified train / validation / test split."""
    i_tr, rest = train_test_split(np.arange(len(y)), train_size=0.7, stratify=y, random_state=seed)
    i_va, i_te = train_test_split(rest, train_size=0.5, stratify=y[rest], random_state=seed)
    return X[i_tr], y[i_tr], X[i_va], y[i_va], X[i_te], y[i_te]

warnings.filterwarnings("ignore")

MAX_ROWS = 50000
TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
TEXT = {  # name: (hugging face id, config, text field, label field)
    "ag_news": ("fancyzhx/ag_news", None, "text", "label"),
    "emotion": ("dair-ai/emotion", "split", "text", "label"),
    "20newsgroups": ("SetFit/20_newsgroups", None, "text", "label"),
    "banking77": ("PolyAI/banking77", None, "text", "label"),
}
IMAGES = {"cifar10": "CIFAR10", "cifar100": "CIFAR100", "fashion": "FashionMNIST"}

BASE = dict(oblique_frac=0.5, supervised=True, n_candidates=6, eta="auto",
            drop_p=0.20, tail_avg=True, max_epochs=25, patience=4)
MT_VARIANTS = {
    "mt": dict(),
    "mt+raw": dict(raw_features=True),
    "mt+raw+rot": dict(raw_features=True),
    "mt+raw+rot+ada": dict(raw_features=True, adagrad=True),
    "mt-probe": dict(probe=True),
    "mt-probe+rot": dict(probe=True),
}
ALL_METHODS = ["linear", "knn", "mlp", "hgb"] + list(MT_VARIANTS)


# ----------------------------------------------------------------------------- embeddings

def _subsample(n, y, seed=0):
    if n <= MAX_ROWS:
        return np.arange(n)
    idx, _ = train_test_split(np.arange(n), train_size=MAX_ROWS, stratify=y, random_state=seed)
    return np.sort(idx)


def _l2(E):
    E = np.asarray(E, dtype=np.float32)
    return E / np.maximum(np.linalg.norm(E, axis=1, keepdims=True), 1e-12)


def embed_text(name, cache):
    path = os.path.join(cache, f"{name}__minilm.npz")
    if os.path.exists(path):
        d = np.load(path)
        return d["X"], d["y"]
    from datasets import load_dataset, concatenate_datasets
    from sentence_transformers import SentenceTransformer
    hf_id, config, tf, lf = TEXT[name]
    try:
        ds = load_dataset(hf_id, config)
    except Exception as e:  # older script-based datasets: try the auto-converted parquet copy
        print(f"  {hf_id}: {type(e).__name__}; trying the parquet conversion")
        ds = load_dataset(hf_id, config, revision="refs/convert/parquet")
    ds = concatenate_datasets([ds[s] for s in ds.keys()])
    texts = np.array(ds[tf], dtype=object)
    y = np.array(ds[lf])
    keep = y >= 0                                     # some test splits hide labels as -1
    texts, y = texts[keep], y[keep]
    idx = _subsample(len(y), y)
    texts, y = texts[idx], LabelEncoder().fit_transform(y[idx])
    print(f"  embedding {len(y)} texts with {TEXT_MODEL} ...", flush=True)
    model = SentenceTransformer(TEXT_MODEL)
    X = model.encode(list(texts), batch_size=256, show_progress_bar=True,
                     convert_to_numpy=True, normalize_embeddings=True)
    X = _l2(X)
    os.makedirs(cache, exist_ok=True)
    np.savez_compressed(path, X=X, y=y)
    return X, y


def _images_openml(data_id, shape):
    """Images from OpenML via scikit-learn: no download if an earlier benchmark already
    fetched this dataset (scikit-learn caches it under ~/scikit_learn_data)."""
    from sklearn.datasets import fetch_openml
    b = fetch_openml(data_id=data_id, as_frame=False, parser="auto")
    imgs = np.asarray(b.data, dtype=np.float32).reshape((-1,) + shape)   # channel-first
    return imgs, LabelEncoder().fit_transform(np.asarray(b.target).astype(str))


def _images_hf(hf_id, label_field):
    """Images from Hugging Face (usually far faster than the original CIFAR host)."""
    from datasets import load_dataset, concatenate_datasets
    ds = load_dataset(hf_id)
    ds = concatenate_datasets([ds[s] for s in ds.keys()])
    img_col = next(c for c, f in ds.features.items() if type(f).__name__ == "Image")
    imgs = np.stack([np.asarray(im.convert("RGB")) for im in ds[img_col]]).astype(np.float32)
    return imgs.transpose(0, 3, 1, 2), LabelEncoder().fit_transform(np.asarray(ds[label_field]))


# name: list of (loader, args) tried in order; torchvision's own download is the last resort
IMAGE_SOURCES = {
    "cifar10": [(_images_openml, (40927, (3, 32, 32))), (_images_hf, ("uoft-cs/cifar10", "label"))],
    "fashion": [(_images_openml, (40996, (1, 28, 28))), (_images_hf, ("zalando-datasets/fashion_mnist", "label"))],
    "cifar100": [(_images_hf, ("uoft-cs/cifar100", "fine_label"))],
}


def _images_torchvision(name, cache):
    import torchvision
    cls = getattr(torchvision.datasets, IMAGES[name])
    root = os.path.join(cache, "torchvision")
    imgs, labels = [], []
    for tr in (True, False):
        d = cls(root, train=tr, download=True)
        a = np.asarray(d.data, dtype=np.float32)
        imgs.append(a[:, None] if a.ndim == 3 else a.transpose(0, 3, 1, 2))
        labels.append(np.asarray(d.targets))
    return np.concatenate(imgs), LabelEncoder().fit_transform(np.concatenate(labels))


def load_images(name, cache):
    """(n, C, H, W) float32 pixels in 0..255 and integer labels, from the fastest source that works."""
    for loader, args in IMAGE_SOURCES.get(name, []):
        try:
            print(f"  loading {name} via {loader.__name__}{args[:1]} ...", flush=True)
            return loader(*args)
        except Exception as e:
            print(f"    failed ({type(e).__name__}: {str(e)[:100]}); trying the next source", flush=True)
    print(f"  loading {name} via torchvision download (can be slow) ...", flush=True)
    return _images_torchvision(name, cache)


def embed_images(name, cache, batch=256, net=None):
    path = os.path.join(cache, f"{name}__resnet18.npz")
    if os.path.exists(path):
        d = np.load(path)
        return d["X"], d["y"]
    import torch
    import torch.nn.functional as F
    imgs, labels = load_images(name, cache)
    if imgs.max() <= 1.0:                       # some sources store pixels as 0..1
        imgs = imgs * 255.0
    idx = _subsample(len(labels), labels)
    imgs, y = imgs[idx], labels[idx]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if net is None:
        import torchvision
        net = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.DEFAULT)
        net.fc = torch.nn.Identity()
    net.eval().to(dev)
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    out = []
    print(f"  embedding {len(y)} images with ResNet-18 on {dev} ...", flush=True)
    t0 = time.time()
    with torch.no_grad():
        for k in range(0, len(y), batch):
            xb = torch.as_tensor(imgs[k:k + batch], device=dev) / 255.0
            if xb.shape[1] == 1:                # grayscale -> 3 channels
                xb = xb.expand(-1, 3, -1, -1)
            xb = F.interpolate(xb, size=(224, 224), mode="bilinear", align_corners=False)
            out.append(net((xb - mean) / std).float().cpu().numpy())
            if (k // batch) % 20 == 0:
                print(f"    {min(k + batch, len(y))}/{len(y)}  {time.time() - t0:.0f}s", flush=True)
    X = _l2(np.concatenate(out))
    os.makedirs(cache, exist_ok=True)
    np.savez_compressed(path, X=X, y=y)
    return X, y


def synthetic_embeddings(seed=0, n=8000, d=128, classes=20):
    """Offline stand-in: clustered classes with curved boundaries, randomly rotated, L2-normalised."""
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(classes * 3, d))
    lab = rng.integers(0, classes * 3, n)
    Z = 0.35 * centers[lab] + rng.normal(size=(n, d))
    Z += 0.3 * np.sin(2 * Z[:, ::-1])
    Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    return _l2(Z @ Q), lab % classes


# ----------------------------------------------------------------------------- methods

def fit_predict(method, Xtr, ytr, Xva, yva, Xte, seed, trees):
    if method == "linear":
        best = None
        for C in (0.1, 1.0, 10.0, 100.0):
            m = LogisticRegression(C=C, max_iter=2000).fit(Xtr, ytr)
            s = m.score(Xva, yva)
            if best is None or s > best[0]:
                best = (s, m)
        return best[1].predict(Xte)
    if method == "knn":
        best = None
        for k in (5, 15, 30):
            m = KNeighborsClassifier(n_neighbors=k, weights="distance", algorithm="brute").fit(Xtr, ytr)
            s = m.score(Xva, yva)
            if best is None or s > best[0]:
                best = (s, m)
        return best[1].predict(Xte)
    if method == "mlp":
        m = MLPClassifier(hidden_layer_sizes=(512,), early_stopping=True, max_iter=200,
                          random_state=seed).fit(Xtr, ytr)
        return m.predict(Xte)
    if method == "hgb":
        return HistGradientBoostingClassifier(random_state=seed).fit(Xtr, ytr).predict(Xte)
    params = {**BASE, **MT_VARIANTS[method]}
    if "rot" in method:
        Q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(Xtr.shape[1],) * 2))
        Q = Q.astype(np.float32)
        Xtr, Xva, Xte = Xtr @ Q, Xva @ Q, Xte @ Q
    m = MedianTreeClassifier(n_trees=trees, random_state=seed, **params).fit(Xtr, ytr, Xva, yva)
    return m.predict(Xte)


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default=",".join(list(TEXT) + list(IMAGES)))
    ap.add_argument("--methods", default=",".join(ALL_METHODS))
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--trees", type=int, default=1024)
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--embed-only", action="store_true")
    ap.add_argument("--local", action="store_true", help="offline test on synthetic embeddings")
    ap.add_argument("--out", default="results/embeddings.csv")
    args = ap.parse_args()
    methods = args.methods.split(",")
    names = ["synthetic"] if args.local else args.datasets.split(",")
    trees = 256 if args.local else args.trees

    data = {}
    for name in names:
        try:
            if args.local:
                data[name] = synthetic_embeddings()
            elif name in TEXT:
                data[name] = embed_text(name, args.cache)
            elif name in IMAGES:
                data[name] = embed_images(name, args.cache)
            else:
                print(f"unknown dataset {name}; choose from {list(TEXT) + list(IMAGES)}")
                continue
            X, y = data[name]
            print(f"{name}: {len(y)} rows, {X.shape[1]} dims, {len(np.unique(y))} classes", flush=True)
        except Exception as e:
            print(f"skipped {name}: {type(e).__name__}: {str(e)[:150]}")
    if args.embed_only:
        return

    Xw = np.random.default_rng(0).normal(size=(600, 16)).astype(np.float32)   # compile kernels
    yw = (Xw[:, 0] > 0).astype(int) + (Xw[:, 1] > 0)
    for v in MT_VARIANTS.values():
        MedianTreeClassifier(n_trees=8, random_state=0, **{**BASE, **v}).fit(Xw, yw).predict(Xw)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    new = not os.path.exists(args.out)
    f = open(args.out, "a", newline="")
    w = csv.writer(f)
    if new:
        w.writerow(["dataset", "seed", "method", "acc", "train_s"])
    for name, (X, y) in data.items():
        print(f"\n=== {name}: n={len(y)} d={X.shape[1]} classes={len(np.unique(y))}, {args.seeds} seeds ===",
              flush=True)
        res = {}
        for meth in methods:
            accs, times = [], []
            for seed in range(args.seeds):
                Xtr, ytr, Xva, yva, Xte, yte = split(X, y, seed)
                t0 = time.perf_counter()
                pred = fit_predict(meth, Xtr, ytr, Xva, yva, Xte, seed, trees)
                tt = time.perf_counter() - t0
                accs.append(float((pred == yte).mean()))
                times.append(tt)
                w.writerow([name, seed, meth, accs[-1], tt])
                f.flush()
            res[meth] = np.array(accs)
            diff = ""
            if "linear" in res and meth != "linear":
                d = 100 * (res[meth] - res["linear"])
                diff = f"{d.mean():+6.2f} ± {d.std():4.2f} vs linear"
            print(f"  {meth:>15}  acc {100 * np.mean(accs):6.2f}  {diff:>24}  train {np.mean(times):7.2f}s",
                  flush=True)
    f.close()
    print("\n(± after a difference is its spread across seeds, paired by split)")


if __name__ == "__main__":
    main()
