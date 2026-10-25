"""
Overnight test battery for median trees: 13 tests, run in priority order, resumable.

  1  learning_curve   accuracy vs training-set size (100 .. 30k rows) against strong baselines,
                      incl. TabPFN (<= 10k rows) and RealMLP: is small data a real strength?
  2  regression       Grinsztajn regression suites (336 numerical, 335 categorical):
                      a median-tree regressor (trees + SGD head on leaves) vs boosting & co.
  3  semi_supervised  trees built from ALL training rows without labels, head trained on 1-30%
                      of the labels: does unlabelled data help? (boosting can't do this)
  4  calibration      ECE, Brier, log-loss, split-conformal coverage and set size
  5  imbalance        minority class at native, 10, 3, 1, 0.3%: PR-AUC, ROC-AUC, balanced acc
  6  stress           junk features, heavy-tailed monotone transform, outliers, rotation,
                      label noise, MCAR / informative missingness
  7  shift            extrapolation splits (train on low values of a feature, test on high)
                      and a time-ordered split, vs random splits
  8  many_classes     2 .. 500 classes (synthetic) plus real many-class data
  9  ood              detect a held-out class: max-softmax vs leaf-rarity vs kNN distance
 10  stability        prediction churn across seeds + one-at-a-time hyperparameter sensitivity
 11  speed            fit / predict time vs rows, features, classes and threads
 12  online           prequential accuracy on drifting streams: frozen trees + online head
 13  probes           probes on LLM hidden states (default gpt2): linear vs MLP vs median trees

Data: OpenML (Grinsztajn suites 337/336/335 and a few CC18 multiclass datasets), cached.
Every unit of work (test, dataset, setting, seed, method) is appended to the results CSV as
soon as it finishes; rerunning skips finished units, so you can stop and restart any time.
Each test has a time budget (minutes, --budget test=minutes,...); when it runs out the test
stops after the current unit and the next test starts.

Methods that need missing libraries are skipped with a note; install what you want compared:
  pip install openml lightgbm xgboost catboost threadpoolctl
  pip install pytabkit                 # RealMLP
  pip install tabpfn                   # TabPFN v2 by default (GPU recommended; no licence login needed)
  pip install torch transformers datasets   # probes test

Usage (median_trees.py v5 and benchmark_suites.py in the same folder):
  nohup python overnight.py < /dev/null > overnight.log 2>&1 &     # everything (~12-14 h at most)
  (the < /dev/null matters: any library that stops to ask a question fails and is skipped instead of
   waiting all night for an answer)
  python overnight.py --tests regression,semi_supervised
  python overnight.py --summary              # report from whatever has finished
  python overnight.py --local                # 5-minute offline smoke test on built-in data
Outputs: results/overnight.csv, results/overnight_report.md, results/overnight_errors.log
"""

import argparse
import csv
import os
import sys
import time
import traceback
import warnings

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.calibration import CalibratedClassifierCV
from sklearn.datasets import load_breast_cancer, load_diabetes, load_digits, make_classification, make_regression
from sklearn.ensemble import (ExtraTreesClassifier, ExtraTreesRegressor, HistGradientBoostingClassifier,
                              HistGradientBoostingRegressor, RandomForestClassifier, RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, RidgeCV, SGDClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

import numba
from median_trees import LeafSGD, MedianTreeClassifier, MedianTreeEmbedder

warnings.filterwarnings("ignore")

MT_BASE = dict(oblique_frac=0.5, supervised=True, n_candidates=6, eta="auto",
               drop_p=0.20, tail_avg=True, max_epochs=25, patience=4)
ALL_TESTS = ["learning_curve", "regression", "semi_supervised", "calibration", "imbalance", "stress",
             "shift", "many_classes", "ood", "stability", "speed", "online", "probes"]
BUDGET_MIN = dict(learning_curve=150, regression=90, semi_supervised=60, calibration=60, imbalance=45,
                  stress=90, shift=30, many_classes=45, ood=30, stability=75, speed=45, online=30, probes=60)
MULTICLASS_IDS = {6: "letter", 32: "pendigits", 182: "satimage", 1478: "har", 1497: "wall-robot-nav",
                  40984: "segment", 28: "optdigits"}
MANY_CLASS_IDS = {6: "letter", 300: "isolet", 40923: "devnagari", 40996: "fashion-mnist"}
ELECTRICITY = 151

ARGS = None
TREES = 1024


# ============================================================================= bookkeeping

class Recorder:
    FIELDS = ["test", "dataset", "setting", "seed", "method", "metric", "value"]

    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.done = set()
        if os.path.exists(path):
            for r in csv.DictReader(open(path)):
                self.done.add((r["test"], r["dataset"], r["setting"], r["seed"], r["method"]))
        new = not os.path.exists(path)
        self.f = open(path, "a", newline="")
        self.w = csv.writer(self.f)
        if new:
            self.w.writerow(self.FIELDS)
        self.err = open(path.replace(".csv", "_errors.log"), "a")

    def is_done(self, test, ds, setting, seed, method):
        return (test, str(ds), str(setting), str(seed), method) in self.done

    def add(self, test, ds, setting, seed, method, metrics):
        for k, v in metrics.items():
            if v is not None and np.isfinite(v):
                self.w.writerow([test, ds, setting, seed, method, k, f"{float(v):.6g}"])
        self.f.flush()
        self.done.add((test, str(ds), str(setting), str(seed), method))

    def error(self, msg):
        self.err.write(msg + "\n")
        self.err.flush()


class Budget(Exception):
    pass


class Test:
    """Shared run loop: time budget, skipping finished units, error isolation."""

    def __init__(self, name, rec):
        self.name, self.rec = name, rec
        self.t0 = time.time()
        self.limit = 60 * BUDGET_MIN.get(name, 60) * ARGS.budget_scale
        self.warned = set()

    def check(self):
        if time.time() - self.t0 > self.limit:
            raise Budget()

    def unit(self, ds, setting, seed, method, fn):
        """fn() -> dict of metrics. Skips if done; records errors without stopping the test."""
        if self.rec.is_done(self.name, ds, setting, seed, method):
            return None
        self.check()
        t = time.perf_counter()
        try:
            out = fn()
        except Unavailable as e:
            if method not in self.warned:
                print(f"    [{method}] skipped: {e}", flush=True)
                self.warned.add(method)
            return None
        except Exception as e:
            msg = f"[{self.name}] {ds} | {setting} | seed {seed} | {method}: {type(e).__name__}: {e}"
            print("    FAILED " + msg[:220], flush=True)
            self.rec.error(msg + "\n" + traceback.format_exc())
            return None
        if out is None:
            return None
        out.setdefault("fit_s", time.perf_counter() - t)
        self.rec.add(self.name, ds, setting, seed, method, out)
        main = next((f"{k} {v:.4f}" for k, v in out.items() if k in ("acc", "r2", "auroc", "pr_auc", "preq_acc")), "")
        print(f"    {str(setting):>18} s{seed} {method:>14}  {main}  ({out['fit_s']:.1f}s)", flush=True)
        return out


class Unavailable(Exception):
    pass


# ============================================================================= data

def _cache_path(key):
    d = os.path.join(ARGS.cache, "overnight")
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"{key}.npz")


def _finish(name, X, y, kind):
    X = np.ascontiguousarray(X, dtype=np.float32)
    X[~np.isfinite(X)] = np.nan
    keep = ~np.all(np.isnan(X), axis=0)
    X = X[:, keep]
    if kind == "clf":
        y = LabelEncoder().fit_transform(np.asarray(y).astype(str))
    else:
        y = np.asarray(y, dtype=np.float64)
        ok = np.isfinite(y)
        X, y = X[ok], y[ok]
    if len(y) > ARGS.max_rows:
        if kind == "clf":
            idx, _ = train_test_split(np.arange(len(y)), train_size=ARGS.max_rows, stratify=y, random_state=0)
        else:
            idx = np.random.default_rng(0).choice(len(y), ARGS.max_rows, replace=False)
        idx = np.sort(idx)
        X, y = X[idx], y[idx]
    return dict(name=name, X=X, y=y, kind=kind, C=int(y.max()) + 1 if kind == "clf" else 0)


def _load_npz(key):
    p = _cache_path(key)
    if os.path.exists(p):
        d = np.load(p, allow_pickle=True)
        return dict(name=str(d["name"]), X=d["X"], y=d["y"], kind=str(d["kind"]), C=int(d["C"]))
    return None


def _save_npz(key, ds):
    np.savez_compressed(_cache_path(key), **ds)


def load_openml_dataset(did, keep_order=False):
    key = f"ds{did}{'_ordered' if keep_order else ''}"
    ds = _load_npz(key)
    if ds is not None:
        return ds
    import openml
    from benchmark_suites import encode_frame
    d = openml.datasets.get_dataset(did, download_data=True, download_qualities=False,
                                    download_features_meta_data=False)
    X_df, y, cat, _ = d.get_data(target=d.default_target_attribute)
    _, _, X = encode_frame(X_df, cat)
    if keep_order:                      # streams: no subsampling, original row order
        X = np.ascontiguousarray(X, np.float32)
        y = LabelEncoder().fit_transform(np.asarray(y).astype(str))
        ds = dict(name=d.name, X=X, y=y, kind="clf", C=int(y.max()) + 1)
    else:
        ds = _finish(d.name, X, y, "clf")
    _save_npz(key, ds)
    return ds


def load_suite(suite_id):
    """All tasks of an OpenML suite as datasets (classification or regression)."""
    import openml
    from benchmark_suites import encode_frame
    out = []
    try:
        tids = [int(t) for t in openml.study.get_suite(suite_id).tasks]
    except Exception as e:
        print(f"  could not list suite {suite_id}: {e}")
        return out
    for tid in tids:
        key = f"task{tid}"
        ds = _load_npz(key)
        if ds is None:
            try:
                task = openml.tasks.get_task(tid, download_splits=False, download_data=False,
                                             download_qualities=False, download_features_meta_data=False)
                kind = "clf" if task.task_type_id == openml.tasks.TaskType.SUPERVISED_CLASSIFICATION else "reg"
                d = openml.datasets.get_dataset(task.dataset_id, download_data=True, download_qualities=False,
                                                download_features_meta_data=False)
                X_df, y, cat, _ = d.get_data(target=task.target_name)
                _, _, X = encode_frame(X_df, cat)
                ds = _finish(d.name, X, y, kind)
                _save_npz(key, ds)
            except Exception as e:
                print(f"  task {tid}: load failed ({type(e).__name__}: {str(e)[:100]})")
                continue
        out.append(ds)
    return out


# --- offline stand-ins (--local)

def _local_clf():
    out = []
    b = load_breast_cancer()
    out.append(dict(name="breast_cancer", X=b.data.astype(np.float32), y=b.target, kind="clf", C=2))
    for s in (0, 1):
        X, y = make_classification(4000, 12, n_informative=6, n_clusters_per_class=3, flip_y=0.03, random_state=s)
        out.append(dict(name=f"synth_bin{s}", X=X.astype(np.float32), y=y, kind="clf", C=2))
    return out


def _local_multi():
    b = load_digits()
    X, y = make_classification(4000, 20, n_informative=12, n_classes=6, n_clusters_per_class=2, random_state=0)
    return [dict(name="digits", X=b.data.astype(np.float32), y=b.target, kind="clf", C=10),
            dict(name="synth_multi", X=X.astype(np.float32), y=y, kind="clf", C=6)]


def _local_reg():
    b = load_diabetes()
    X, y = make_regression(3000, 10, n_informative=6, noise=10, random_state=0)
    Xf, yf = make_regression(3000, 8, n_informative=5, noise=5, random_state=1)
    yf = np.sin(Xf[:, 0]) * 30 + Xf[:, 1] ** 2 * 5 + yf * 0.2
    return [dict(name="diabetes", X=b.data.astype(np.float32), y=b.target.astype(float), kind="reg", C=0),
            dict(name="synth_lin", X=X.astype(np.float32), y=y, kind="reg", C=0),
            dict(name="synth_nonlin", X=Xf.astype(np.float32), y=yf, kind="reg", C=0)]


def clf_datasets():
    return _local_clf() if ARGS.local else load_suite(337)


def reg_datasets():
    return _local_reg() if ARGS.local else load_suite(336) + load_suite(335)


def multiclass_datasets(ids=MULTICLASS_IDS):
    if ARGS.local:
        return _local_multi()
    out = []
    for did in ids:
        try:
            out.append(load_openml_dataset(did))
        except Exception as e:
            print(f"  dataset {did}: load failed ({type(e).__name__}: {str(e)[:100]})")
    return out


def stratified_split(y, test_size, seed, cap_train=None, cap_test=None):
    idx = np.arange(len(y))
    try:
        tr, te = train_test_split(idx, test_size=test_size, stratify=y, random_state=seed)
    except ValueError:
        tr, te = train_test_split(idx, test_size=test_size, random_state=seed)
    rng = np.random.default_rng(seed)
    if cap_train and len(tr) > cap_train:
        tr = subsample(tr, y, cap_train, seed)
    if cap_test and len(te) > cap_test:
        te = rng.choice(te, cap_test, replace=False)
    return tr, te


def subsample(idx, y, n, seed):
    if n >= len(idx):
        return idx
    try:
        keep, _ = train_test_split(idx, train_size=n, stratify=y[idx], random_state=seed)
    except ValueError:
        keep = np.random.default_rng(seed).choice(idx, n, replace=False)
    return keep


# ============================================================================= models

def _imp(est):
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), est)


def _need(module):
    try:
        return __import__(module)
    except ImportError:
        raise Unavailable(f"pip install {module}")


def _device():
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


class TunedLGBM:
    """20 random LightGBM configurations, early stopping on a 15% validation split,
    the best refit on all training rows at its best iteration."""

    def __init__(self, seed, kind="clf", n_iter=20):
        self.seed, self.kind, self.n_iter = seed, kind, n_iter

    def fit(self, X, y):
        lgb = _need("lightgbm")
        rng = np.random.default_rng(self.seed)
        strat = y if self.kind == "clf" else None
        try:
            tr, va = train_test_split(np.arange(len(y)), test_size=0.15, stratify=strat, random_state=self.seed)
        except ValueError:
            tr, va = train_test_split(np.arange(len(y)), test_size=0.15, random_state=self.seed)
        Est = lgb.LGBMClassifier if self.kind == "clf" else lgb.LGBMRegressor
        best = None
        for _ in range(self.n_iter):
            p = dict(num_leaves=int(rng.choice([7, 15, 31, 63, 127])), learning_rate=float(rng.choice([0.02, 0.05, 0.1])),
                     min_child_samples=int(rng.choice([3, 10, 20, 50])), subsample=float(rng.uniform(0.6, 1.0)),
                     subsample_freq=1, colsample_bytree=float(rng.uniform(0.5, 1.0)),
                     reg_lambda=float(rng.choice([0.0, 1.0, 10.0])))
            m = Est(n_estimators=3000, random_state=self.seed, verbose=-1, **p)
            m.fit(X[tr], y[tr], eval_set=[(X[va], y[va])], callbacks=[lgb.early_stopping(100, verbose=False)])
            score = list(m.best_score_["valid_0"].values())[0]
            if best is None or score < best[0]:
                best = (score, p, max(1, m.best_iteration_ or 100))
        self.model_ = Est(n_estimators=best[2], random_state=self.seed, verbose=-1, **best[1]).fit(X, y)
        self.classes_ = getattr(self.model_, "classes_", None)
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def predict(self, X):
        return self.model_.predict(X)


def _tabpfn(cls_name, seed):
    """TabPFN v2 by default: its weights are not behind Prior Labs' licence login (2.5 and later are).
    --tabpfn-version latest uses the package default instead."""
    tp = _need("tabpfn")
    cls = getattr(tp, cls_name)
    if ARGS.tabpfn_version == "latest":
        return cls(device="auto", random_state=seed)
    try:
        from tabpfn.constants import ModelVersion
        version = getattr(ModelVersion, ARGS.tabpfn_version.upper())
    except (ImportError, AttributeError):
        raise Unavailable(f"this tabpfn package has no ModelVersion.{ARGS.tabpfn_version.upper()}")
    try:
        return cls.create_default_for_version(version, device="auto", random_state=seed)
    except TypeError:
        est = cls.create_default_for_version(version)
        est.set_params(device="auto", random_state=seed)
        return est


def make_clf(name, seed, info):
    n, d, C = info["n"], info["d"], info["C"]
    if name.startswith("mt"):
        p = dict(MT_BASE)
        if name == "mt+raw":
            p["raw_features"] = True
        elif name == "mt-axis":
            p["oblique_frac"] = 0.0
        elif name == "mt-probe":
            p = dict(probe=True)
        elif name == "mt-impute":
            return _imp(MedianTreeClassifier(n_trees=TREES, random_state=seed, **p))
        elif name != "mt":
            raise ValueError(name)
        p.update(info.get("mt_override", {}))
        return MedianTreeClassifier(n_trees=info.get("trees", TREES), random_state=seed, **p)
    if name == "hgb":
        return HistGradientBoostingClassifier(random_state=seed)
    if name == "hgb-balanced":
        return HistGradientBoostingClassifier(random_state=seed, class_weight="balanced")
    if name == "hgb-isotonic":
        return CalibratedClassifierCV(HistGradientBoostingClassifier(random_state=seed), method="isotonic", cv=3)
    if name == "rf":
        return _imp(RandomForestClassifier(300, n_jobs=-1, random_state=seed))
    if name == "et":
        return _imp(ExtraTreesClassifier(300, n_jobs=-1, random_state=seed))
    if name == "logreg":
        return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                             LogisticRegression(max_iter=3000))
    if name == "logreg-balanced":
        return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                             LogisticRegression(max_iter=3000, class_weight="balanced"))
    if name == "knn":
        return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                             KNeighborsClassifier(15, weights="distance"))
    if name == "lgbm":
        return _need("lightgbm").LGBMClassifier(random_state=seed, verbose=-1)
    if name == "lgbm-tuned":
        return TunedLGBM(seed, "clf")
    if name == "xgb":
        return _need("xgboost").XGBClassifier(tree_method="hist", random_state=seed, n_jobs=-1)
    if name == "catboost":
        return _need("catboost").CatBoostClassifier(random_seed=seed, verbose=0, thread_count=-1)
    if name == "realmlp":
        pk = _need("pytabkit")
        return _imp(pk.RealMLP_TD_Classifier(random_state=seed, device=_device(), verbosity=0))
    if name == "tabpfn":
        if n > ARGS.tabpfn_max_rows or d > 500 or C > 10:
            return None
        return _tabpfn("TabPFNClassifier", seed)
    raise ValueError(f"unknown classifier {name}")


def fit_proba(name, Xtr, ytr, Xte, C, seed, **info):
    """Fits a classifier and returns test probabilities over all C classes (rare classes can be
    missing from a small training set; they get probability 0)."""
    present = np.unique(ytr)
    if len(present) < 2:
        P = np.zeros((len(Xte), C))
        P[:, present[0]] = 1.0
        return P, None
    enc = np.searchsorted(present, ytr)
    m = make_clf(name, seed, dict(n=len(ytr), d=Xtr.shape[1], C=len(present), **info))
    if m is None:
        return None, None
    m.fit(Xtr, enc)
    Ps = np.asarray(m.predict_proba(Xte), dtype=np.float64)
    P = np.zeros((len(Xte), C))
    P[:, present[np.asarray(m.classes_).astype(int)] if getattr(m, "classes_", None) is not None else present] = Ps
    return P, m


# ----------------------------------------------------------------------------- median-tree regressor

@numba.njit(cache=True)
def _reg_epoch(L, R, y, order, W, V, b, lr, lr_raw, lr_b, l2, n_leaves, use_raw):
    T = L.shape[1]
    d = R.shape[1]
    for k in range(len(order)):
        i = order[k]
        s = b[0]
        for t in range(T):
            s += W[t * n_leaves + L[i, t]]
        if use_raw:
            for j in range(d):
                s += V[j] * R[i, j]
        g = s - y[i]
        for t in range(T):
            r = t * n_leaves + L[i, t]
            W[r] -= lr * (g + l2 * W[r])
        if use_raw:
            for j in range(d):
                V[j] -= lr_raw * (g * R[i, j] + l2 * V[j])
        b[0] -= lr_b * g


class MedianTreeRegressor:
    """Supervised median trees (splits scored on the standardized target) + an SGD head on the
    leaves with squared loss, learning-rate grid and early stopping on a 10% validation split."""

    def __init__(self, n_trees=1024, depth=7, oblique_frac=0.5, n_candidates=6, raw=False,
                 max_epochs=40, patience=4, l2=1e-6, random_state=0):
        self.n_trees, self.depth, self.oblique_frac = n_trees, depth, oblique_frac
        self.n_candidates, self.raw, self.max_epochs = n_candidates, raw, max_epochs
        self.patience, self.l2, self.random_state = patience, l2, random_state

    def _scale(self, X):
        Z = np.ascontiguousarray((X - self.mean_) / self.sd_, dtype=np.float32)
        return Z, np.ascontiguousarray(np.nan_to_num(Z, nan=0.0), dtype=np.float32)

    def fit(self, X, y):
        X = np.asarray(X, np.float32)
        y = np.asarray(y, np.float64)
        rng = np.random.default_rng(self.random_state)
        va = rng.random(len(y)) < 0.1
        if va.sum() < 5:
            va[:5] = True
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.mean_ = np.nan_to_num(np.nanmean(X[~va], 0)).astype(np.float32)
            sd = np.nan_to_num(np.nanstd(X[~va], 0))
        self.sd_ = np.where(sd > 0, sd, 1.0).astype(np.float32)
        self.ym_, self.ys_ = y[~va].mean(), y[~va].std() + 1e-12
        z = ((y - self.ym_) / self.ys_).astype(np.float32)
        Z, R = self._scale(X)
        self.emb_ = MedianTreeEmbedder(n_trees=self.n_trees, depth=self.depth, oblique_frac=self.oblique_frac,
                                       n_candidates=self.n_candidates, random_state=self.random_state)
        self.emb_.fit(Z[~va], targets=z[~va, None])
        L = self.emb_.apply(Z)
        nl, T = 1 << self.depth, self.n_trees
        Ltr, Lva, Rtr, Rva, ztr, zva = L[~va], L[va], R[~va], R[va], z[~va], z[va]
        sq = max(1.0, float((Rtr ** 2).sum(1).mean()))
        best = None
        for eta in (0.05, 0.2, 0.6):
            W = np.zeros(T * nl, np.float32)
            V = np.zeros(R.shape[1], np.float32)
            b = np.zeros(1, np.float32)
            hist, snaps, waited, local = [], [], 0, None
            for ep in range(self.max_epochs):
                dec = 1.0 / (1.0 + 0.3 * ep)
                order = rng.permutation(len(ztr)).astype(np.int64)
                _reg_epoch(Ltr, Rtr, ztr, order, W, V, b, np.float32(eta / T * dec),
                           np.float32(0.5 * eta / sq * dec), np.float32(0.01 * dec), np.float32(self.l2), nl, self.raw)
                rmse = float(np.sqrt(np.mean((self._pred(Lva, Rva, W, V, b) - zva) ** 2)))
                snaps.append((W.copy(), V.copy(), b.copy()))
                snaps = snaps[-3:]
                if local is None or rmse < local[0]:
                    local, waited = (rmse, W.copy(), V.copy(), b.copy()), 0
                else:
                    waited += 1
                    if waited >= self.patience:
                        break
            if len(snaps) >= 2:                 # tail average of the last few epochs
                Wa, Va, ba = (sum(s[i] for s in snaps) / len(snaps) for i in range(3))
                r = float(np.sqrt(np.mean((self._pred(Lva, Rva, Wa, Va, ba) - zva) ** 2)))
                if r < local[0]:
                    local = (r, Wa, Va, ba)
            if best is None or local[0] < best[0]:
                best = local
        _, self.W_, self.V_, self.b_ = best
        return self

    def _pred(self, L, R, W, V, b):
        nl = 1 << self.depth
        rows = L + (np.arange(L.shape[1]) * nl)[None, :]
        s = W[rows].sum(1) + b[0]
        if self.raw:
            s = s + R @ V
        return s

    def predict(self, X):
        Z, R = self._scale(np.asarray(X, np.float32))
        return self._pred(self.emb_.apply(Z), R, self.W_, self.V_, self.b_) * self.ys_ + self.ym_


def make_reg(name, seed, info):
    n, d = info["n"], info["d"]
    if name == "mt":
        return MedianTreeRegressor(n_trees=TREES, random_state=seed)
    if name == "mt+raw":
        return MedianTreeRegressor(n_trees=TREES, raw=True, random_state=seed)
    if name == "hgb":
        return HistGradientBoostingRegressor(random_state=seed)
    if name == "rf":
        return _imp(RandomForestRegressor(300, n_jobs=-1, random_state=seed))
    if name == "et":
        return _imp(ExtraTreesRegressor(300, n_jobs=-1, random_state=seed))
    if name == "ridge":
        return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                             RidgeCV(alphas=np.logspace(-3, 4, 15)))
    if name == "lgbm":
        return _need("lightgbm").LGBMRegressor(random_state=seed, verbose=-1)
    if name == "lgbm-tuned":
        return TunedLGBM(seed, "reg")
    if name == "xgb":
        return _need("xgboost").XGBRegressor(tree_method="hist", random_state=seed, n_jobs=-1)
    if name == "catboost":
        return _need("catboost").CatBoostRegressor(random_seed=seed, verbose=0, thread_count=-1)
    if name == "realmlp":
        return _imp(_need("pytabkit").RealMLP_TD_Regressor(random_state=seed, device=_device(), verbosity=0))
    if name == "tabpfn":
        if n > ARGS.tabpfn_max_rows or d > 500:
            return None
        return _tabpfn("TabPFNRegressor", seed)
    raise ValueError(f"unknown regressor {name}")


# ============================================================================= metrics

def clf_metrics(P, y, C, extra=True):
    P = np.clip(P, 1e-12, 1.0)
    P = P / P.sum(1, keepdims=True)
    pred = P.argmax(1)
    out = dict(acc=float((pred == y).mean()))
    out["logloss"] = float(-np.mean(np.log(P[np.arange(len(y)), y])))
    if extra:
        Y = np.eye(C)[y]
        out["brier"] = float(np.mean(((P - Y) ** 2).sum(1)))
        conf, hit = P.max(1), (pred == y)
        bins = np.minimum((conf * 15).astype(int), 14)
        out["ece"] = float(sum(abs(hit[bins == b].mean() - conf[bins == b].mean()) * (bins == b).mean()
                               for b in range(15) if (bins == b).any()))
        rec = [(pred[y == c] == c).mean() for c in range(C) if (y == c).any()]
        out["bal_acc"] = float(np.mean(rec))
    if C == 2 and len(np.unique(y)) == 2:
        out["auc"] = float(roc_auc_score(y, P[:, 1]))
        out["pr_auc"] = float(average_precision_score(y, P[:, 1]))
    return out


def reg_metrics(pred, y):
    rmse = float(np.sqrt(np.mean((pred - y) ** 2)))
    return dict(r2=float(1 - np.mean((pred - y) ** 2) / max(np.var(y), 1e-12)), rmse=rmse)


def conformal(P_cal, y_cal, P_te, y_te, alpha=0.1):
    s = 1 - P_cal[np.arange(len(y_cal)), y_cal]
    n = len(s)
    q = np.quantile(s, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n), method="higher")
    sets = P_te >= 1 - q
    return dict(cov90=float(sets[np.arange(len(y_te)), y_te].mean()), setsize90=float(sets.sum(1).mean()))


def seeds_for(n_train=None):
    k = ARGS.seeds
    if n_train is not None and n_train <= 1000:
        k = 2 * k
    return range(k)


# ============================================================================= tests

def t_learning_curve(T):
    methods = ARGS.methods or "mt,mt-probe,hgb,lgbm,lgbm-tuned,xgb,catboost,rf,logreg,realmlp,tabpfn"
    sizes = [100, 300, 1000] if ARGS.local else [100, 300, 1000, 3000, 10000, 30000]
    for ds in clf_datasets() + multiclass_datasets():
        print(f"  {ds['name']}: n={len(ds['y'])} d={ds['X'].shape[1]} C={ds['C']}", flush=True)
        X, y, C = ds["X"], ds["y"], ds["C"]
        for n in sizes:
            if n > 0.7 * len(y):
                continue
            for seed in seeds_for(n):
                pool, te = stratified_split(y, 0.3, seed, cap_test=10000)
                tr = subsample(pool, y, n, seed)
                for m in methods.split(","):
                    if m == "lgbm-tuned" and n > 10000:
                        continue

                    def fn(m=m):
                        P, _ = fit_proba(m, X[tr], y[tr], X[te], C, seed)
                        return None if P is None else clf_metrics(P, y[te], C, extra=False)
                    T.unit(ds["name"], n, seed, m, fn)


def t_regression(T):
    methods = ARGS.methods or "mt,mt+raw,hgb,lgbm,lgbm-tuned,xgb,catboost,rf,et,ridge,realmlp,tabpfn"
    for ds in reg_datasets():
        print(f"  {ds['name']}: n={len(ds['y'])} d={ds['X'].shape[1]}", flush=True)
        X, y = ds["X"], ds["y"]
        for seed in range(ARGS.seeds):
            tr, te = train_test_split(np.arange(len(y)), test_size=0.3, random_state=seed)
            for m in methods.split(","):
                def fn(m=m):
                    est = make_reg(m, seed, dict(n=len(tr), d=X.shape[1]))
                    if est is None:
                        return None
                    est.fit(X[tr], y[tr])
                    return reg_metrics(np.asarray(est.predict(X[te]), np.float64).ravel(), y[te])
                T.unit(ds["name"], "full", seed, m, fn)


def _mt_head_on(L_lab, y_lab, seed, n_leaves, classes):
    try:
        tr, va = train_test_split(np.arange(len(y_lab)), test_size=0.2, stratify=y_lab, random_state=seed)
    except ValueError:
        tr, va = train_test_split(np.arange(len(y_lab)), test_size=0.2, random_state=seed)
    return LeafSGD(loss="softmax", eta="auto", drop_p=0.2, tail_avg=True, max_epochs=25, patience=4,
                   random_state=seed).fit(L_lab[tr], y_lab[tr], L_lab[va], y_lab[va], n_leaves=n_leaves,
                                          classes=classes)


def _softmax(S):
    E = np.exp(S - S.max(1, keepdims=True))
    return E / E.sum(1, keepdims=True)


def mt_label_free(X_trees, X_lab, y_lab, X_te, C, seed):
    """Label-free median trees grown on X_trees (all rows, or only the labelled ones), head on the labels."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mu = np.nan_to_num(np.nanmean(X_trees, 0))
        sd = np.nan_to_num(np.nanstd(X_trees, 0))
    sd = np.where(sd > 0, sd, 1.0)
    Z = lambda A: np.ascontiguousarray((A - mu) / sd, dtype=np.float32)
    emb = MedianTreeEmbedder(n_trees=TREES, depth=7, oblique_frac=0.5, random_state=seed).fit(Z(X_trees))
    classes = np.unique(y_lab)
    head = _mt_head_on(emb.apply(Z(X_lab)), y_lab, seed, 128, classes)
    S = head.decision_function(emb.apply(Z(X_te)))
    P = np.zeros((len(X_te), C))
    P[:, classes] = _softmax(np.asarray(S, dtype=np.float64))
    return P


def t_semi_supervised(T):
    methods = (ARGS.methods or "mt-ssl,mt-label-free,mt,hgb,lgbm,logreg,tabpfn").split(",")
    fracs = [0.01, 0.03, 0.1, 0.3, 1.0]
    for ds in clf_datasets():
        print(f"  {ds['name']}: n={len(ds['y'])}", flush=True)
        X, y, C = ds["X"], ds["y"], ds["C"]
        for frac in fracs:
            for seed in range(ARGS.seeds):
                pool, te = stratified_split(y, 0.3, seed, cap_train=20000, cap_test=10000)
                n_lab = max(20, int(round(frac * len(pool))))
                lab = subsample(pool, y, n_lab, seed)
                for m in methods:
                    def fn(m=m):
                        if m == "mt-ssl":
                            P = mt_label_free(X[pool], X[lab], y[lab], X[te], C, seed)
                        elif m == "mt-label-free":
                            P = mt_label_free(X[lab], X[lab], y[lab], X[te], C, seed)
                        else:
                            P, _ = fit_proba(m, X[lab], y[lab], X[te], C, seed)
                        if P is None:
                            return None
                        r = clf_metrics(P, y[te], C, extra=False)
                        r["n_labels"] = len(lab)
                        return r
                    T.unit(ds["name"], frac, seed, m, fn)


def t_calibration(T):
    methods = (ARGS.methods or "mt,mt+raw,hgb,hgb-isotonic,lgbm,catboost,rf,logreg").split(",")
    for ds in clf_datasets() + multiclass_datasets():
        print(f"  {ds['name']}: n={len(ds['y'])} C={ds['C']}", flush=True)
        X, y, C = ds["X"], ds["y"], ds["C"]
        for seed in range(ARGS.seeds):
            tr, te = stratified_split(y, 0.3, seed, cap_train=20000, cap_test=10000)
            fit_i, cal_i = stratified_split(y[tr], 0.2, seed)
            fit_i, cal_i = tr[fit_i], tr[cal_i]
            for m in methods:
                def fn(m=m):
                    P_both, _ = fit_proba(m, X[fit_i], y[fit_i], np.vstack([X[cal_i], X[te]]), C, seed)
                    if P_both is None:
                        return None
                    P_cal, P_te = P_both[:len(cal_i)], P_both[len(cal_i):]
                    r = clf_metrics(P_te, y[te], C)
                    r.update(conformal(P_cal, y[cal_i], P_te, y[te]))
                    return r
                T.unit(ds["name"], "-", seed, m, fn)


def t_imbalance(T):
    methods = (ARGS.methods or "mt,hgb,hgb-balanced,lgbm,catboost,rf,logreg-balanced").split(",")
    rates = ["native", 0.1, 0.03, 0.01, 0.003]
    for ds in clf_datasets():
        if ds["C"] != 2:
            continue
        X, y = ds["X"], ds["y"]
        minority = int(np.argmin(np.bincount(y)))
        yb = (y == minority).astype(int)
        print(f"  {ds['name']}: n={len(y)} minority share {yb.mean():.3f}", flush=True)
        for rate in rates:
            for seed in range(ARGS.seeds):
                rng = np.random.default_rng(seed)
                pos, neg = np.where(yb == 1)[0], np.where(yb == 0)[0]
                if rate != "native":
                    want = int(round(rate / (1 - rate) * len(neg)))
                    if want >= len(pos):
                        continue
                    pos = rng.choice(pos, want, replace=False)
                keep = np.concatenate([pos, neg])
                tr, te = stratified_split(yb[keep], 0.3, seed, cap_train=30000, cap_test=20000)
                tr, te = keep[tr], keep[te]
                if yb[te].sum() < 10 or yb[tr].sum() < 10:
                    continue
                for m in methods:
                    def fn(m=m):
                        P, _ = fit_proba(m, X[tr], yb[tr], X[te], 2, seed)
                        return None if P is None else clf_metrics(P, yb[te], 2)
                    T.unit(ds["name"], rate, seed, m, fn)


def _perturb(kind, val, X_tr, X_te, y_tr, C, seed):
    rng = np.random.default_rng(seed + 12345)
    Xtr, Xte, ytr = X_tr.copy(), X_te.copy(), y_tr.copy()
    if kind == "junk":
        k = int(val)
        Xtr = np.hstack([Xtr, rng.normal(size=(len(Xtr), k)).astype(np.float32)])
        Xte = np.hstack([Xte, rng.normal(size=(len(Xte), k)).astype(np.float32)])
    elif kind in ("exp", "rotate"):
        mu, sd = np.nanmean(Xtr, 0), np.nanstd(Xtr, 0)
        sd = np.where(sd > 0, sd, 1.0)
        Ztr, Zte = (Xtr - mu) / sd, (Xte - mu) / sd
        if kind == "exp":                         # monotone, heavy right tail
            Xtr, Xte = np.exp(2 * np.clip(Ztr, -8, 8)), np.exp(2 * np.clip(Zte, -8, 8))
        else:
            Q, _ = np.linalg.qr(rng.normal(size=(Xtr.shape[1],) * 2))
            Xtr, Xte = np.nan_to_num(Ztr) @ Q, np.nan_to_num(Zte) @ Q
    elif kind == "outliers":                      # 1% of entries scaled by 1000
        for A in (Xtr, Xte):
            m = rng.random(A.shape) < 0.01
            A[m] *= 1000.0
    elif kind == "label_noise":
        flip = rng.random(len(ytr)) < float(val)
        ytr[flip] = (ytr[flip] + rng.integers(1, C, flip.sum())) % C
    elif kind == "mcar":
        for A in (Xtr, Xte):
            A[rng.random(A.shape) < float(val)] = np.nan
    elif kind == "mnar":                          # large values go missing: informative missingness
        med = np.nanmedian(Xtr, 0)
        for A in (Xtr, Xte):
            A[(A > med) & (rng.random(A.shape) < float(val))] = np.nan
    return np.ascontiguousarray(Xtr, np.float32), np.ascontiguousarray(Xte, np.float32), ytr


def t_stress(T):
    methods = (ARGS.methods or "mt,mt-axis,hgb,lgbm,rf,logreg").split(",")
    settings = [("none", 0), ("junk", 10), ("junk", 50), ("junk", 200), ("exp", 0), ("outliers", 0),
                ("rotate", 0), ("label_noise", 0.1), ("label_noise", 0.2), ("label_noise", 0.4),
                ("mcar", 0.1), ("mcar", 0.3), ("mcar", 0.5), ("mnar", 0.5)]
    for ds in clf_datasets():
        print(f"  {ds['name']}: n={len(ds['y'])} d={ds['X'].shape[1]}", flush=True)
        X, y, C = ds["X"], ds["y"], ds["C"]
        for kind, val in settings:
            tag = kind if kind in ("none", "exp", "outliers", "rotate") else f"{kind}={val}"
            for seed in range(max(1, ARGS.seeds - 2)):
                tr, te = stratified_split(y, 0.3, seed, cap_train=10000, cap_test=10000)
                Xtr, Xte, ytr = _perturb(kind, val, X[tr], X[te], y[tr], C, seed)
                ms = methods + (["mt-impute"] if kind in ("mcar", "mnar") else [])
                for m in ms:
                    def fn(m=m):
                        P, _ = fit_proba(m, Xtr, ytr, Xte, C, seed)
                        return None if P is None else clf_metrics(P, y[te], C, extra=False)
                    T.unit(ds["name"], tag, seed, m, fn)


def t_shift(T):
    methods = (ARGS.methods or "mt,mt-axis,hgb,lgbm,rf,logreg").split(",")
    for ds in clf_datasets():
        X, y, C = ds["X"], ds["y"], ds["C"]
        print(f"  {ds['name']}", flush=True)
        Xf = np.nan_to_num(X)
        corr = np.nan_to_num([abs(np.corrcoef(Xf[:, j], y)[0, 1]) if Xf[:, j].std() > 0 else 0
                              for j in range(X.shape[1])])
        order = np.argsort(-corr)
        for seed in range(max(1, ARGS.seeds - 2)):
            j = int(order[seed % len(order)])
            q = np.nanquantile(X[:, j], 0.7)
            low = np.where(X[:, j] <= q)[0]
            high = np.where(X[:, j] > q)[0]
            if len(high) < 100 or len(low) < 100:
                continue
            rng = np.random.default_rng(seed)
            splits = {"random": stratified_split(y, 0.3, seed, cap_train=20000, cap_test=10000),
                      "extrapolate": (rng.permutation(low)[:20000], rng.permutation(high)[:10000])}
            for tag, (tr, te) in splits.items():
                for m in methods:
                    def fn(m=m, tr=tr, te=te):
                        P, _ = fit_proba(m, X[tr], y[tr], X[te], C, seed)
                        return None if P is None else clf_metrics(P, y[te], C, extra=False)
                    T.unit(ds["name"], tag, seed, m, fn)
    # time-ordered split on electricity
    try:
        ds = _stream_electricity()
    except Exception as e:
        print(f"  electricity unavailable: {e}")
        return
    X, y = ds["X"], ds["y"]
    cut = int(0.7 * len(y))
    splits = {"random": stratified_split(y, 0.3, 0), "temporal": (np.arange(cut), np.arange(cut, len(y)))}
    for tag, (tr, te) in splits.items():
        for m in methods:
            def fn(m=m, tr=tr, te=te):
                P, _ = fit_proba(m, X[tr], y[tr], X[te], 2, 0)
                return None if P is None else clf_metrics(P, y[te], 2, extra=False)
            T.unit("electricity", tag, 0, m, fn)


def t_many_classes(T):
    methods = (ARGS.methods or "mt,mt+raw,hgb,lgbm,logreg,rf,knn").split(",")
    jobs = []
    for C in ([2, 10, 50] if ARGS.local else [2, 10, 50, 100, 250, 500]):
        jobs.append((f"synthetic-C{C}", C))
    for ds in multiclass_datasets(MANY_CLASS_IDS) + _cifar100():
        jobs.append((ds["name"], ds))
    for name, spec in jobs:
        for seed in range(max(1, ARGS.seeds - 3)):
            if isinstance(spec, int):
                n = 6000 if ARGS.local else 30000
                X, y = make_classification(n, 40, n_informative=30, n_redundant=5, n_classes=spec,
                                           n_clusters_per_class=1, class_sep=1.5, random_state=seed)
                X, C = X.astype(np.float32), spec
            else:
                X, y, C = spec["X"], spec["y"], spec["C"]
            tr, te = stratified_split(y, 0.3, seed, cap_train=30000, cap_test=10000)
            print(f"  {name}: C={C} n_train={len(tr)}", flush=True)
            for m in methods:
                if m in ("hgb", "lgbm", "xgb", "catboost") and C > ARGS.boost_max_classes:
                    continue

                def fn(m=m):
                    P, _ = fit_proba(m, X[tr], y[tr], X[te], C, seed)
                    return None if P is None else clf_metrics(P, y[te], C, extra=False)
                T.unit(name, C, seed, m, fn)


def _cifar100():
    p = ARGS.cifar100_npz
    if p and os.path.exists(p):
        d = np.load(p)
        return [dict(name="cifar100-resnet18", X=d["X"].astype(np.float32), y=d["y"], kind="clf",
                     C=int(d["y"].max()) + 1)]
    return []


def t_ood(T):
    for ds in multiclass_datasets():
        X, y, C = ds["X"], ds["y"], ds["C"]
        print(f"  {ds['name']}: C={C}", flush=True)
        for seed in range(max(1, ARGS.seeds - 3)):
            for k in range(min(C, 5)):
                tr, te = stratified_split(y, 0.3, seed, cap_train=20000, cap_test=10000)
                tr_in = tr[y[tr] != k]
                te_in, te_out = te[y[te] != k], te[y[te] == k]
                if len(te_out) < 20:
                    continue
                te_all = np.concatenate([te_in, te_out])
                is_out = np.r_[np.zeros(len(te_in)), np.ones(len(te_out))]
                keep = np.array([c for c in range(C) if c != k])
                y_tr = np.searchsorted(keep, y[tr_in])
                y_te_in = np.searchsorted(keep, y[te_in])
                setting = f"class{k}"
                cache = {}

                def fit_model(name):
                    if name not in cache:
                        cache[name] = fit_proba(name, X[tr_in], y_tr, X[te_all], C - 1, seed)
                    return cache[name]

                def msp(name):
                    def fn():
                        P, _ = fit_model(name)
                        return dict(auroc=float(roc_auc_score(is_out, 1 - P.max(1))),
                                    acc_in=float((P[:len(te_in)].argmax(1) == y_te_in).mean()))
                    return fn

                def rarity():
                    P, m = fit_model("mt")
                    emb = m.stages_[0][0]
                    Ltr = emb.apply(m._prep_X(X[tr_in])[0])
                    Lte = emb.apply(m._prep_X(X[te_all])[0])
                    nl = 1 << emb.depth
                    off = np.arange(Ltr.shape[1]) * nl
                    cnt = np.bincount((Ltr + off).ravel(), minlength=Ltr.shape[1] * nl)
                    score = -np.log1p(cnt[Lte + off]).mean(1)
                    return dict(auroc=float(roc_auc_score(is_out, score)))

                def knn():
                    imp = SimpleImputer(strategy="median", keep_empty_features=True).fit(X[tr_in])
                    sc = StandardScaler().fit(imp.transform(X[tr_in]))
                    Z = lambda A: sc.transform(imp.transform(A))
                    ref = subsample(np.arange(len(tr_in)), y_tr, min(20000, len(tr_in)), seed)
                    nn = NearestNeighbors(n_neighbors=10).fit(Z(X[tr_in])[ref])
                    dist, _ = nn.kneighbors(Z(X[te_all]))
                    return dict(auroc=float(roc_auc_score(is_out, dist.mean(1))))

                T.unit(ds["name"], setting, seed, "mt:max-softmax", msp("mt"))
                T.unit(ds["name"], setting, seed, "mt:leaf-rarity", rarity)
                T.unit(ds["name"], setting, seed, "hgb:max-softmax", msp("hgb"))
                T.unit(ds["name"], setting, seed, "rf:max-softmax", msp("rf"))
                T.unit(ds["name"], setting, seed, "knn-distance", knn)


def t_stability(T):
    datasets = clf_datasets()[:8]
    # (a) churn: same split, different model seeds
    for ds in datasets:
        X, y, C = ds["X"], ds["y"], ds["C"]
        tr, te = stratified_split(y, 0.3, 0, cap_train=20000, cap_test=10000)
        print(f"  churn {ds['name']}", flush=True)
        for m in (ARGS.methods or "mt,mt-probe,hgb,lgbm,rf").split(","):
            def fn(m=m):
                preds, accs = [], []
                for s in range(5):
                    P, _ = fit_proba(m, X[tr], y[tr], X[te], C, s)
                    if P is None:
                        return None
                    preds.append(P.argmax(1))
                    accs.append((preds[-1] == y[te]).mean())
                churn = np.mean([(preds[a] != preds[b]).mean() for a in range(5) for b in range(a + 1, 5)])
                return dict(acc=float(np.mean(accs)), acc_sd=float(np.std(accs)), churn=float(churn))
            T.unit(ds["name"], "churn", 0, m, fn)
    # (b) one-at-a-time sensitivity around the default configuration
    grid = dict(depth=[5, 6, 8, 9], n_trees=[128, 256, 512, 2048], drop_p=[0.0, 0.1, 0.3, 0.4],
                oblique_frac=[0.0, 0.25, 0.75, 1.0], n_candidates=[1, 3, 12], max_samples=[1024, 16384])
    configs = [("default", {})] + [(f"{k}={v}", {k: v}) for k, vs in grid.items() for v in vs]
    for ds in datasets:
        X, y, C = ds["X"], ds["y"], ds["C"]
        print(f"  sensitivity {ds['name']}", flush=True)
        for seed in range(3):
            tr, te = stratified_split(y, 0.3, seed, cap_train=20000, cap_test=10000)
            for tag, ov in configs:
                def fn(ov=ov):
                    ov = dict(ov)
                    trees = ov.pop("n_trees", TREES)
                    P, _ = fit_proba("mt", X[tr], y[tr], X[te], C, seed, mt_override=ov, trees=trees)
                    return clf_metrics(P, y[te], C, extra=False)
                T.unit(ds["name"], tag, seed, "mt", fn)


def _threads(k):
    try:
        from threadpoolctl import threadpool_limits
        return threadpool_limits(limits=k)
    except ImportError:
        import contextlib
        return contextlib.nullcontext()


def t_speed(T):
    methods = (ARGS.methods or "mt,hgb,lgbm,xgb,catboost,rf").split(",")
    max_n = ARGS.speed_max_n
    axes = [("rows", n, dict(n=n, d=20, C=2)) for n in [1000, 10000, 100000, 300000, 1000000] if n <= max_n]
    axes += [("features", d, dict(n=20000, d=d, C=2)) for d in [10, 100, 500, 2000]]
    axes += [("classes", c, dict(n=20000, d=20, C=c)) for c in [2, 10, 50, 100]]
    n_cpu = numba.config.NUMBA_NUM_THREADS
    axes += [("threads", k, dict(n=100000, d=20, C=2, threads=k)) for k in sorted({1, 2, 4, 8, n_cpu}) if k <= n_cpu]
    if ARGS.local:
        axes = [a for a in axes if a[2]["n"] <= 20000 and a[2]["d"] <= 100]
    for axis, val, spec in axes:
        X, y = make_classification(int(spec["n"] * 1.25), spec["d"], n_informative=min(spec["d"], 15), n_redundant=0,
                                   n_classes=spec["C"], n_clusters_per_class=1, random_state=0)
        X = X.astype(np.float32)
        tr, te = np.arange(spec["n"]), np.arange(spec["n"], len(y))
        k = spec.get("threads", n_cpu)
        print(f"  {axis}={val}", flush=True)
        for m in methods:
            def fn(m=m):
                numba.set_num_threads(k)
                try:
                    with _threads(k):
                        est = make_clf(m, 0, dict(n=len(tr), d=X.shape[1], C=spec["C"]))
                        if hasattr(est, "get_params"):
                            for key in est.get_params():
                                if key.split("__")[-1] in ("n_jobs", "thread_count"):
                                    est.set_params(**{key: k})
                        t0 = time.perf_counter()
                        est.fit(X[tr], y[tr])
                        t1 = time.perf_counter()
                        acc = float((est.predict(X[te]) == y[te]).mean())
                        t2 = time.perf_counter()
                finally:
                    numba.set_num_threads(n_cpu)
                return dict(fit_s=t1 - t0, predict_ms_per_1k=1000 * (t2 - t1) / len(te) * 1000, acc=acc)
            T.unit("synthetic", f"{axis}={val}", 0, m, fn)


def _stream_electricity():
    if ARGS.local:
        X, y = make_classification(12000, 8, n_informative=5, random_state=3)
        return dict(name="electricity-standin", X=X.astype(np.float32), y=y)
    return load_openml_dataset(ELECTRICITY, keep_order=True)


def _stream_rotating(n=40000, d=10, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d)).astype(np.float32)
    w0, w1 = rng.normal(size=d), rng.normal(size=d)
    a = np.linspace(0, np.pi, n)[:, None]
    w = np.cos(a) * w0 + np.sin(a) * w1
    y = ((X * w).sum(1) + 0.3 * rng.normal(size=n) > 0).astype(int)
    return dict(name="rotating-hyperplane", X=X, y=y)


def _stream_abrupt(n=40000, d=6, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 10, size=(n, d)).astype(np.float32)
    thr = np.repeat([8.0, 9.0, 7.0, 9.5], n // 4 + 1)[:n]
    y = ((X[:, 0] + X[:, 1] > thr) ^ (rng.random(n) < 0.05)).astype(int)
    return dict(name="abrupt-drift", X=X, y=y)


def t_online(T):
    streams = [_stream_electricity(), _stream_rotating(), _stream_abrupt()]
    if ARGS.local:
        streams = [dict(name=s["name"], X=s["X"][:8000], y=s["y"][:8000]) for s in streams]
    chunk, warm, window, every = 500, 2000, 5000, 4
    for st in streams:
        X, y = st["X"], st["y"]
        classes = np.unique(y)
        print(f"  {st['name']}: n={len(y)}", flush=True)
        starts = list(range(warm, len(y) - chunk + 1, chunk))

        def run(kind):
            hits, t0 = [], time.perf_counter()
            if kind in ("mt-static", "mt-window"):
                model = MedianTreeClassifier(n_trees=TREES, random_state=0, **MT_BASE).fit(X[:warm], y[:warm])
            elif kind in ("hgb-static", "hgb-window"):
                model = HistGradientBoostingClassifier(random_state=0).fit(X[:warm], y[:warm])
            elif kind == "mt-online":
                mu, sd = np.nanmean(X[:warm], 0), np.nanstd(X[:warm], 0) + 1e-6
                Z = lambda A: np.ascontiguousarray((A - mu) / sd, dtype=np.float32)
                emb = MedianTreeEmbedder(n_trees=TREES, depth=7, oblique_frac=0.5, n_candidates=6, random_state=0)
                Yw = np.eye(len(classes), dtype=np.float32)[np.searchsorted(classes, y[:warm])]
                emb.fit(Z(X[:warm]), targets=Yw - Yw.mean(0))
                model = SGDClassifier(loss="log_loss", alpha=1e-5, random_state=0)
                F = emb.transform(Z(X[:warm]))
                for _ in range(5):
                    model.partial_fit(F, y[:warm], classes=classes)
            elif kind == "sgd-raw":
                sc = StandardScaler().fit(np.nan_to_num(X[:warm]))
                model = SGDClassifier(loss="log_loss", alpha=1e-4, random_state=0)
                for _ in range(5):
                    model.partial_fit(sc.transform(np.nan_to_num(X[:warm])), y[:warm], classes=classes)
            for c, s in enumerate(starts):
                sl = slice(s, s + chunk)
                if kind == "mt-online":
                    F = emb.transform(Z(X[sl]))
                    hits.append(model.predict(F) == y[sl])
                    model.partial_fit(F, y[sl])
                elif kind == "sgd-raw":
                    F = sc.transform(np.nan_to_num(X[sl]))
                    hits.append(model.predict(F) == y[sl])
                    model.partial_fit(F, y[sl])
                else:
                    hits.append(model.predict(X[sl]) == y[sl])
                    if kind.endswith("window") and (c + 1) % every == 0:
                        lo = max(0, s + chunk - window)
                        if len(np.unique(y[lo:s + chunk])) > 1:
                            model = (MedianTreeClassifier(n_trees=TREES, random_state=c, **MT_BASE)
                                     if kind.startswith("mt") else HistGradientBoostingClassifier(random_state=0)
                                     ).fit(X[lo:s + chunk], y[lo:s + chunk])
            h = np.concatenate(hits)
            return dict(preq_acc=float(h.mean()), last_quarter_acc=float(h[-len(h) // 4:].mean()),
                        fit_s=time.perf_counter() - t0)

        for kind in ["mt-static", "mt-online", "mt-window", "hgb-static", "hgb-window", "sgd-raw"]:
            T.unit(st["name"], "prequential", 0, kind, lambda kind=kind: run(kind))


PROBE_TASKS = {"emotion": ("dair-ai/emotion", "split", "text", "label", "train", "test"),
               "ag_news": ("fancyzhx/ag_news", None, "text", "label", "train", "test"),
               "sst2": ("stanfordnlp/sst2", None, "sentence", "label", "train", "validation")}


def _hidden_states(task, model_name, layers_frac=(0, 0.25, 0.5, 0.75, 1.0), n_train=8000, n_test=2000):
    slug = model_name.replace("/", "_")
    path = _cache_path(f"probe_{task}_{slug}")
    if os.path.exists(path):
        d = np.load(path)
        return {k: d[k] for k in d.files}
    torch = _need("torch")
    tf = _need("transformers")
    from datasets import load_dataset
    hf, cfg, tcol, lcol, s_tr, s_te = PROBE_TASKS[task]
    data = load_dataset(hf, cfg) if cfg else load_dataset(hf)
    out = {}
    tok = tf.AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    dev = _device()
    model = tf.AutoModel.from_pretrained(model_name, output_hidden_states=True).to(dev).eval()
    for split, cap, key in ((s_tr, n_train, "tr"), (s_te, n_test, "te")):
        texts = np.array(data[split][tcol], dtype=object)
        labels = np.array(data[split][lcol])
        ok = labels >= 0
        texts, labels = texts[ok], labels[ok]
        if len(labels) > cap:
            idx = subsample(np.arange(len(labels)), labels, cap, 0)
            texts, labels = texts[idx], labels[idx]
        reps = None
        with torch.no_grad():
            for i in range(0, len(texts), 32):
                enc = tok(list(texts[i:i + 32]), return_tensors="pt", padding=True, truncation=True,
                          max_length=128).to(dev)
                hs = model(**enc).hidden_states
                L = len(hs) - 1
                pick = sorted({int(round(f * L)) for f in layers_frac})
                m = enc["attention_mask"].unsqueeze(-1).float()
                pooled = [((hs[l] * m).sum(1) / m.sum(1)).float().cpu().numpy() for l in pick]
                if reps is None:
                    reps = {l: [] for l in pick}
                for l, p in zip(pick, pooled):
                    reps[l].append(p)
        for l in reps:
            out[f"{key}_L{l}"] = np.concatenate(reps[l]).astype(np.float32)
        out[f"y{key}"] = labels
    np.savez_compressed(path, **out)
    return out


def t_probes(T):
    if ARGS.local:
        print("  skipped in --local (needs a language model download)")
        return
    probes = (ARGS.methods or "linear,mlp,knn,hgb,mt,mt+raw").split(",")
    for task in PROBE_TASKS:
        try:
            H = _hidden_states(task, ARGS.probe_model)
        except Unavailable as e:
            print(f"  probes skipped: {e}")
            return
        except Exception as e:
            print(f"  {task}: hidden states failed ({type(e).__name__}: {str(e)[:150]})")
            continue
        ytr, yte = H["ytr"], H["yte"]
        C = int(max(ytr.max(), yte.max())) + 1
        for key in sorted(k for k in H if k.startswith("tr_L")):
            layer = key[3:]
            sc = StandardScaler().fit(H[key])
            Xtr, Xte = sc.transform(H[key]).astype(np.float32), sc.transform(H["te_" + layer]).astype(np.float32)
            print(f"  {task} {layer}: d={Xtr.shape[1]}", flush=True)
            for seed in range(max(1, ARGS.seeds - 2)):
                for m in probes:
                    def fn(m=m):
                        if m == "linear":
                            tr, va = stratified_split(ytr, 0.2, seed)
                            best = max((LogisticRegression(C=c, max_iter=3000).fit(Xtr[tr], ytr[tr]) for c in
                                        (0.01, 0.1, 1.0)), key=lambda e: e.score(Xtr[va], ytr[va]))
                            P = best.predict_proba(Xte)
                            P = np.column_stack([P[:, list(best.classes_).index(c)] if c in best.classes_
                                                 else np.zeros(len(Xte)) for c in range(C)])
                        elif m == "mlp":
                            e = MLPClassifier((256,), early_stopping=True, max_iter=200, random_state=seed).fit(Xtr, ytr)
                            P = np.zeros((len(Xte), C))
                            P[:, e.classes_] = e.predict_proba(Xte)
                        else:
                            P, _ = fit_proba(m, Xtr, ytr, Xte, C, seed)
                        return clf_metrics(P, yte, C, extra=False)
                    T.unit(task, layer, seed, m, fn)


TESTS = {name: globals()[f"t_{name}"] for name in ALL_TESTS}


# ============================================================================= report

PRIMARY = dict(learning_curve=["acc", "logloss"], regression=["r2"], semi_supervised=["acc"],
               calibration=["ece", "logloss", "brier", "acc", "cov90", "setsize90"],
               imbalance=["pr_auc", "auc", "bal_acc"], stress=["acc"], shift=["acc"], many_classes=["acc", "fit_s"],
               ood=["auroc"], stability=["acc", "churn"], speed=["fit_s", "predict_ms_per_1k"],
               online=["preq_acc", "last_quarter_acc"], probes=["acc"])
LOWER_BETTER = {"logloss", "brier", "ece", "fit_s", "predict_ms_per_1k", "churn", "rmse", "setsize90"}


def _fmt(v, metric):
    if not np.isfinite(v):
        return "–"
    if metric in ("fit_s",):
        return f"{v:.2f}"
    if metric in ("predict_ms_per_1k",):
        return f"{v:.2f}"
    if metric in ("acc", "auc", "pr_auc", "bal_acc", "r2", "auroc", "preq_acc", "last_quarter_acc", "cov90"):
        return f"{100 * v:.2f}" if metric != "r2" else f"{v:.4f}"
    return f"{v:.4f}"


def report(path, out_md):
    if not os.path.exists(path):
        print("no results yet")
        return
    df = pd.read_csv(path)
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    lines = ["# Median trees: overnight test report", "",
             "Each cell: mean over datasets of the per-dataset mean over seeds, using only datasets where every "
             "method in the table finished (n = number of datasets). Accuracy-type metrics in %. "
             "In parentheses: mean rank of the method across those datasets (1 = best).", ""]
    for test in ALL_TESTS:
        g = df[df.test == test]
        if g.empty:
            continue
        lines += [f"## {test}", ""]
        for metric in PRIMARY[test]:
            gm = g[g.metric == metric]
            if gm.empty:
                continue
            per = gm.groupby(["setting", "dataset", "method"])["value"].mean().reset_index()
            settings = list(dict.fromkeys(per["setting"]))
            try:
                settings = sorted(settings, key=lambda s: float(s))
            except ValueError:
                pass
            methods = list(dict.fromkeys(per["method"]))
            rows, ranks = [], {}
            for s in settings:
                piv = per[per.setting == s].pivot(index="dataset", columns="method", values="value")
                piv = piv.dropna(axis=1, how="all")
                full = piv.dropna()
                use = full if len(full) else piv
                means = use.mean()
                asc = metric in LOWER_BETTER
                rk = use.rank(axis=1, ascending=asc).mean() if len(full) else None
                rows.append((s, len(use), means, rk))
            lines.append(f"**{metric}**" + (" (lower is better)" if metric in LOWER_BETTER else ""))
            lines.append("")
            lines.append("| setting | n | " + " | ".join(methods) + " |")
            lines.append("|---|---|" + "---|" * len(methods))
            for s, n, means, rk in rows:
                cells = []
                for m in methods:
                    v = means.get(m, np.nan)
                    cell = _fmt(v, metric)
                    if rk is not None and m in rk and np.isfinite(rk[m]):
                        cell += f" ({rk[m]:.1f})"
                    cells.append(cell)
                lines.append(f"| {s} | {n} | " + " | ".join(cells) + " |")
            lines.append("")
            if len(settings) > 1 and test not in ("speed",):
                lines.append("_(n) is the number of datasets; numbers in parentheses after values are mean ranks._")
                lines.append("")
    text = "\n".join(lines)
    with open(out_md, "w") as f:
        f.write(text + "\n")
    print(text)
    print(f"\nsaved {out_md}")


# ============================================================================= main

def main():
    global ARGS, TREES
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tests", default=",".join(ALL_TESTS))
    ap.add_argument("--methods", default=None, help="override a test's method list (comma-separated)")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--trees", type=int, default=1024)
    ap.add_argument("--max-rows", type=int, default=50000)
    ap.add_argument("--budget", default="", help="per-test minutes, e.g. regression=120,speed=20")
    ap.add_argument("--budget-scale", type=float, default=1.0, help="multiply every test's time budget")
    ap.add_argument("--tabpfn-max-rows", type=int, default=10000)
    ap.add_argument("--boost-max-classes", type=int, default=100,
                    help="skip boosting above this many classes (one tree per class per round gets very slow)")
    ap.add_argument("--speed-max-n", type=int, default=300000, help="1,000,000 needs ~6 GB RAM for median trees")
    ap.add_argument("--tabpfn-version", default="v2",
                    help="TabPFN weights: v2 (no licence login needed) or latest (asks for a Prior Labs login)")
    ap.add_argument("--probe-model", default="gpt2")
    ap.add_argument("--cifar100-npz", default="cache/cifar100__resnet18.npz",
                    help="ResNet-18 CIFAR-100 embeddings from embeddings_experiment.py, if present")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--out", default="results/overnight.csv")
    ap.add_argument("--summary", action="store_true")
    ap.add_argument("--local", action="store_true")
    ARGS = ap.parse_args()
    for kv in filter(None, ARGS.budget.split(",")):
        k, v = kv.split("=")
        BUDGET_MIN[k] = float(v)
    if ARGS.local:
        ARGS.out = ARGS.out.replace(".csv", "_local.csv") if ARGS.out == "results/overnight.csv" else ARGS.out
        ARGS.trees, ARGS.seeds = min(ARGS.trees, 64), min(ARGS.seeds, 2)
        for k in BUDGET_MIN:
            BUDGET_MIN[k] = min(BUDGET_MIN[k], 1.5)
    TREES = ARGS.trees
    out_md = ARGS.out.replace(".csv", "_report.md")
    if ARGS.summary:
        report(ARGS.out, out_md)
        return

    print("compiling kernels ...", flush=True)          # numba compile before anything is timed
    Xw = np.random.default_rng(0).normal(size=(600, 10)).astype(np.float32)
    for yw in ((Xw[:, 0] > 0).astype(int), (Xw[:, 0] > 0).astype(int) + (Xw[:, 1] > 0)):
        MedianTreeClassifier(n_trees=8, random_state=0, **MT_BASE).fit(Xw, yw).predict_proba(Xw)
    MedianTreeRegressor(n_trees=8).fit(Xw, Xw[:, 0]).predict(Xw)

    rec = Recorder(ARGS.out)
    t_all = time.time()
    for name in ARGS.tests.split(","):
        if name not in TESTS:
            print(f"unknown test {name!r}; choose from {ALL_TESTS}")
            continue
        print(f"\n===== {name} (budget {BUDGET_MIN[name] * ARGS.budget_scale:.0f} min) =====", flush=True)
        T = Test(name, rec)
        try:
            TESTS[name](T)
        except Budget:
            print(f"  time budget reached; moving on (rerun later to continue {name})", flush=True)
        except KeyboardInterrupt:
            print("\ninterrupted; everything finished so far is saved")
            break
        except Exception as e:
            print(f"  test {name} crashed: {type(e).__name__}: {e}", flush=True)
            rec.error(f"[{name}] CRASH {traceback.format_exc()}")
        print(f"  {name} done in {(time.time() - T.t0) / 60:.1f} min", flush=True)
    print(f"\nall done in {(time.time() - t_all) / 3600:.2f} h")
    report(ARGS.out, out_md)


if __name__ == "__main__":
    main()
