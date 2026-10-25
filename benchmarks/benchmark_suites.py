"""
Median trees against four public benchmark families, with one harness, official splits,
resumable results and a summary table.

  cc18        OpenML-CC18 (suite 99): 72 classification tasks, the general-purpose gamut
  grinsztajn  Grinsztajn et al. 2022, classification only: suite 337 (numerical features)
              and suite 334 (numerical + categorical). The paper's 45 also count regression.
  tabarena    TabArena-v0.1 (suite 457, alias "tabarena-v0.1"); regression tasks are skipped.
  mteb        The 12 English MTEB classification datasets, embedded once with a
              sentence-transformers model and cached; official train/test splits.

Protocol
  OpenML suites: the task's own predefined splits (repeat 0, first --folds folds). Training
  folds above --max-train rows and test folds above --max-test rows are subsampled
  (stratified) to keep the run tractable. Every model tunes only on its own training fold:
  median trees hold out 10% internally, gradient boosting uses its default early stopping.
  MTEB: the official test split; 10% of the train split is held out for the methods that
  need a validation set (linear C, k for kNN, median trees' early stopping), the same rows
  for every method.

  Reported per fold: accuracy, ROC AUC (binary) and log-loss (multiclass) - the latter two
  are TabArena's official metrics - plus fit and predict time.

Methods (pick with --methods; optional ones run only if their library is installed)
  tabular:  hgb, et, mt, mt-probe   [+ lgbm, xgb, catboost, logreg]
  mteb:     linear, knn, mlp, hgb, mt+raw, mt-probe
  mt        = the fixed configuration used throughout (supervised oblique splits, dropout,
              tail averaging, eta="auto")
  mt-probe  = MedianTreeClassifier(probe=True)
  --mt-params '{"refit": true}' adds keyword arguments to every median-tree method; keys your
  version of median_trees.py does not have are dropped with a warning.

Usage (keep median_trees.py in the same folder; pip install openml, and for MTEB also
datasets + sentence-transformers):
  python benchmark_suites.py --suites cc18 --folds 3
  python benchmark_suites.py --suites grinsztajn,tabarena
  python benchmark_suites.py --suites mteb --seeds 3
  python benchmark_suites.py --summary            # just print the tables from results so far
  python benchmark_suites.py --local              # offline smoke test on stand-in data
Interrupt any time; rerunning skips everything already in the results file.
"""

import argparse
import csv
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from median_trees import MedianTreeClassifier

warnings.filterwarnings("ignore")

OPENML_SUITES = {"cc18": [99], "grinsztajn": [337, 334], "tabarena": [457]}
SUITE_ALIASES = {457: "tabarena-v0.1"}

# MTEB English classification: name -> (hugging face id, config); the mteb package is the fallback
MTEB = {
    "AmazonCounterfactual": ("mteb/amazon_counterfactual", "en"),
    "AmazonPolarity": ("mteb/amazon_polarity", None),
    "AmazonReviews": ("mteb/amazon_reviews_multi", "en"),
    "Banking77": ("mteb/banking77", None),
    "Emotion": ("mteb/emotion", None),
    "Imdb": ("mteb/imdb", None),
    "MassiveIntent": ("mteb/amazon_massive_intent", "en"),
    "MassiveScenario": ("mteb/amazon_massive_scenario", "en"),
    "MTOPDomain": ("mteb/mtop_domain", "en"),
    "MTOPIntent": ("mteb/mtop_intent", "en"),
    "ToxicConversations": ("mteb/toxic_conversations_50k", None),
    "TweetSentimentExtraction": ("mteb/tweet_sentiment_extraction", None),
}

MT_BASE = dict(oblique_frac=0.5, supervised=True, n_candidates=6, eta="auto",
               drop_p=0.20, tail_avg=True, max_epochs=25, patience=4)
MT_METHODS = {"mt": MT_BASE, "mt+raw": dict(MT_BASE, raw_features=True), "mt-probe": dict(probe=True)}
TABULAR_DEFAULT = "hgb,et,mt,mt-probe"
MTEB_DEFAULT = "linear,knn,mlp,hgb,mt+raw,mt-probe"
FIELDS = ["suite", "task", "dataset", "fold", "method", "n_train", "n_test", "d", "C",
          "acc", "auc", "logloss", "fit_s", "pred_s"]


# ============================================================================= models

_MT_KEYS = set(MedianTreeClassifier().get_params())
_warned = set()


def make_mt(params, extra, trees, seed):
    p = {**params, **extra}
    unknown = set(p) - _MT_KEYS
    for k in unknown - _warned:
        print(f"  [note] your median_trees.py has no '{k}' option; ignoring it", flush=True)
        _warned.add(k)
    p = {k: v for k, v in p.items() if k in _MT_KEYS}
    return MedianTreeClassifier(n_trees=trees, random_state=seed, **p)


def make_tabular(method, cat_mask, seed, trees, mt_extra):
    """Returns (model, input) where input is 'ord' (categories as integer codes) or 'oh'."""
    if method == "hgb":
        cf = cat_mask if cat_mask.any() else None
        return HistGradientBoostingClassifier(categorical_features=cf, random_state=seed), "ord"
    if method == "et":
        return make_pipeline(SimpleImputer(strategy="median"),
                             ExtraTreesClassifier(n_estimators=256, n_jobs=-1, random_state=seed)), "oh"
    if method == "logreg":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(max_iter=2000)), "oh"
    if method == "lgbm":
        from lightgbm import LGBMClassifier
        return LGBMClassifier(random_state=seed, verbose=-1), "ord"
    if method == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(random_state=seed, tree_method="hist"), "ord"
    if method == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(random_seed=seed, verbose=0, thread_count=-1), "ord"
    if method in MT_METHODS:
        return make_mt(MT_METHODS[method], mt_extra, trees, seed), "oh"
    raise ValueError(f"unknown method {method!r}")


def full_proba(model, X, C):
    """predict_proba expanded to all C classes (a training fold can miss a rare class)."""
    P = np.asarray(model.predict_proba(X), dtype=np.float64)
    out = np.zeros((len(X), C))
    out[:, np.asarray(model.classes_).astype(int)] = P
    return out


def scores(P, y, C):
    acc = float((P.argmax(1) == y).mean())
    auc = ll = np.nan
    if C == 2 and len(np.unique(y)) == 2:
        auc = float(roc_auc_score(y, P[:, 1]))
    if C > 2:
        Q = np.clip(P, 1e-15, 1.0)
        ll = float(log_loss(y, Q / Q.sum(1, keepdims=True), labels=np.arange(C)))
    return acc, auc, ll


# ============================================================================= tabular data

def encode_frame(X_df, cat_ind, max_onehot=32):
    """Two views of one table: 'ord' keeps categories as integer codes (for gradient boosting,
    with a mask of the columns it may treat as categorical); 'oh' one-hot encodes categories
    with up to `max_onehot` levels and keeps larger ones as codes (median trees, extra trees)."""
    ord_cols, cat_mask, oh_cols = [], [], []
    for j, col in enumerate(X_df.columns):
        s = X_df[col]
        if cat_ind[j] or not pd.api.types.is_numeric_dtype(s):
            codes = pd.Categorical(s).codes.astype(np.float32)
            codes[codes < 0] = np.nan
            k = int(np.nanmax(codes)) + 1 if np.isfinite(codes).any() else 0
            ord_cols.append(codes)
            cat_mask.append(1 < k <= 255)
            if 2 < k <= max_onehot:
                oh = np.zeros((len(codes), k), dtype=np.float32)
                ok = np.isfinite(codes)
                oh[np.where(ok)[0], codes[ok].astype(int)] = 1.0
                oh_cols.append(oh)
            else:
                oh_cols.append(codes[:, None])
        else:
            v = pd.to_numeric(s, errors="coerce").to_numpy(dtype=np.float32)
            v[~np.isfinite(v)] = np.nan
            ord_cols.append(v)
            cat_mask.append(False)
            oh_cols.append(v[:, None])
    X_ord = np.ascontiguousarray(np.column_stack(ord_cols), dtype=np.float32)
    X_oh = np.ascontiguousarray(np.hstack(oh_cols), dtype=np.float32)
    return X_ord, np.array(cat_mask, dtype=bool), X_oh


def openml_tasks(suite):
    import openml
    tids = []
    for sid in OPENML_SUITES[suite]:
        try:
            s = openml.study.get_suite(sid)
        except Exception:
            s = openml.study.get_suite(SUITE_ALIASES[sid])
        tids += [int(t) for t in s.tasks]
    return list(dict.fromkeys(tids))


def load_openml_task(tid, n_folds_wanted):
    """Returns None for non-classification tasks, else a dict with both views, labels and
    the task's predefined (train, test) index pairs."""
    import openml
    try:
        task = openml.tasks.get_task(tid, download_splits=True, download_data=False,
                                     download_qualities=False, download_features_meta_data=False)
    except TypeError:
        task = openml.tasks.get_task(tid)
    if task.task_type_id != openml.tasks.TaskType.SUPERVISED_CLASSIFICATION:
        return None
    ds = openml.datasets.get_dataset(task.dataset_id, download_data=True, download_qualities=False,
                                     download_features_meta_data=False)
    X_df, y_ser, cat_ind, _ = ds.get_data(target=task.target_name)
    X_ord, cat_mask, X_oh = encode_frame(X_df, cat_ind)
    y = LabelEncoder().fit_transform(y_ser.astype(str).to_numpy())
    _, n_folds, _ = task.get_split_dimensions()
    splits = [task.get_train_test_split_indices(fold=f, repeat=0, sample=0)
              for f in range(min(n_folds_wanted, n_folds))]
    return dict(name=ds.name, X_ord=X_ord, X_oh=X_oh, cat_mask=cat_mask, y=y, splits=splits)


def local_tabular_tasks(n_folds):
    """Offline stand-ins for the OpenML suites (smoke test only)."""
    from sklearn.datasets import load_breast_cancer, load_digits, load_wine
    out = []
    for name, loader in (("breast_cancer", load_breast_cancer), ("wine", load_wine), ("digits", load_digits)):
        b = loader()
        out.append((name, pd.DataFrame(b.data), b.target.astype(str), [False] * b.data.shape[1]))
    rng = np.random.default_rng(0)                         # mixed numeric / categorical / missing
    n = 3000
    df = pd.DataFrame({"x0": rng.normal(size=n), "x1": rng.exponential(size=n),
                       "c0": rng.choice(list("abcdefgh"), n), "c1": rng.choice(["u", "v"], n)})
    df.loc[rng.random(n) < 0.1, "x1"] = np.nan
    logit = df.x0 + (df.c0 < "d") - (df.c1 == "u") * np.nan_to_num(df.x1)
    out.append(("synthetic_mixed", df, (logit + rng.logistic(size=n) > 0).astype(str), [False, False, True, True]))
    tasks = []
    for k, (name, df, y_raw, cat_ind) in enumerate(out):
        X_ord, cat_mask, X_oh = encode_frame(df, cat_ind)
        y = LabelEncoder().fit_transform(y_raw)
        skf = StratifiedKFold(n_splits=max(2, n_folds), shuffle=True, random_state=0)
        splits = list(skf.split(X_ord, y))[:n_folds]
        tasks.append((-(k + 1), dict(name=name, X_ord=X_ord, X_oh=X_oh, cat_mask=cat_mask, y=y, splits=splits)))
    return tasks


def _cap(idx, y, cap, seed):
    if len(idx) <= cap:
        return idx
    try:
        keep, _ = train_test_split(idx, train_size=cap, stratify=y[idx], random_state=seed)
    except ValueError:                                     # a class with one member
        keep = np.random.default_rng(seed).choice(idx, cap, replace=False)
    return np.sort(keep)


def run_tabular(suite, tasks, args, rec):
    methods = args.methods.split(",") if args.methods else TABULAR_DEFAULT.split(",")
    mt_extra = json.loads(args.mt_params) if args.mt_params else {}
    skip = {int(s) for s in args.skip.split(",") if s} if args.skip else set()
    for i, tid in enumerate(tasks):
        if (tid[0] if isinstance(tid, tuple) else tid) in skip:
            continue
        if rec.task_done(suite, tid, args.folds, methods):
            continue
        t0 = time.time()
        try:
            data = tid[1] if isinstance(tid, tuple) else load_openml_task(tid, args.folds)
            tid = tid[0] if isinstance(tid, tuple) else tid
        except Exception as e:
            print(f"[{suite} {i + 1}/{len(tasks)}] task {tid}: load failed ({type(e).__name__}: {str(e)[:100]})")
            continue
        if data is None:
            print(f"[{suite} {i + 1}/{len(tasks)}] task {tid}: not classification, skipped")
            continue
        y, C = data["y"], int(data["y"].max()) + 1
        if data["X_oh"].shape[1] > args.max_features:
            print(f"[{suite}] {data['name']}: {data['X_oh'].shape[1]} features > --max-features, skipped")
            continue
        print(f"\n[{suite} {i + 1}/{len(tasks)}] task {tid} {data['name']}: n={len(y)} "
              f"d={data['X_ord'].shape[1]} (one-hot {data['X_oh'].shape[1]}) C={C} "
              f"loaded in {time.time() - t0:.0f}s", flush=True)
        for f, (tr, te) in enumerate(data["splits"]):
            tr = _cap(np.asarray(tr), y, args.max_train, f)
            te = _cap(np.asarray(te), y, args.max_test, f)
            for meth in methods:
                if rec.done(suite, tid, f, meth):
                    continue
                try:
                    model, view = make_tabular(meth, data["cat_mask"], f, args.trees, mt_extra)
                except ImportError:
                    print(f"  {meth}: library not installed, skipped")
                    continue
                X = data["X_ord"] if view == "ord" else data["X_oh"]
                try:
                    t1 = time.perf_counter()
                    model.fit(X[tr], y[tr])
                    t2 = time.perf_counter()
                    P = full_proba(model, X[te], C)
                    t3 = time.perf_counter()
                except Exception as e:
                    print(f"  fold {f} {meth}: FAILED {type(e).__name__}: {str(e)[:120]}", flush=True)
                    continue
                acc, auc, ll = scores(P, y[te], C)
                rec.add(suite, tid, data["name"], f, meth, len(tr), len(te), X.shape[1], C,
                        acc, auc, ll, t2 - t1, t3 - t2)
                print(f"  fold {f} {meth:>9}  acc {100 * acc:6.2f}  "
                      f"{'auc %.4f' % auc if np.isfinite(auc) else 'logloss %.4f' % ll if np.isfinite(ll) else ''}"
                      f"  fit {t2 - t1:6.1f}s", flush=True)


# ============================================================================= MTEB

def _hf_splits(hf_id, config):
    from datasets import load_dataset
    try:
        ds = load_dataset(hf_id, config) if config else load_dataset(hf_id)
    except Exception:
        ds = load_dataset(hf_id, config, revision="refs/convert/parquet") if config else \
            load_dataset(hf_id, revision="refs/convert/parquet")
    return ds


def _mteb_splits(name):
    import mteb
    task = mteb.get_task(f"{name}Classification", languages=["eng"])
    task.load_data()
    ds = task.dataset
    if "en" in ds:
        ds = ds["en"]
    return ds


def load_mteb_texts(name):
    hf_id, config = MTEB[name]
    try:
        ds = _hf_splits(hf_id, config)
    except Exception as e:
        print(f"  {hf_id}: {type(e).__name__}; trying the mteb package", flush=True)
        ds = _mteb_splits(name)

    def cols(split):
        d = ds[split]
        names = d.column_names
        tf = next(c for c in ("text", "sentence", "content") if c in names)
        lf = "label" if "label" in names else next(c for c in names if "label" in c)
        return np.array(d[tf], dtype=object), np.array(d[lf])

    tr_texts, tr_y = cols("train")
    te_texts, te_y = cols("test")
    return tr_texts, tr_y, te_texts, te_y


def embed_mteb(name, model_name, cache, max_train, max_test):
    slug = model_name.split("/")[-1]
    path = os.path.join(cache, f"mteb__{name}__{slug}.npz")
    if os.path.exists(path):
        d = np.load(path)
        return d["Xtr"], d["ytr"], d["Xte"], d["yte"]
    tr_t, tr_y, te_t, te_y = load_mteb_texts(name)
    enc = LabelEncoder().fit(np.concatenate([tr_y.astype(str), te_y.astype(str)]))
    ytr, yte = enc.transform(tr_y.astype(str)), enc.transform(te_y.astype(str))
    itr = _cap(np.arange(len(ytr)), ytr, max_train, 0)     # subsample before embedding
    ite = _cap(np.arange(len(yte)), yte, max_test, 0)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    print(f"  embedding {len(itr)} + {len(ite)} texts with {model_name} ...", flush=True)
    kw = dict(batch_size=256, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True)
    Xtr = model.encode(list(tr_t[itr]), **kw).astype(np.float32)
    Xte = model.encode(list(te_t[ite]), **kw).astype(np.float32)
    os.makedirs(cache, exist_ok=True)
    np.savez_compressed(path, Xtr=Xtr, ytr=ytr[itr], Xte=Xte, yte=yte[ite])
    return Xtr, ytr[itr], Xte, yte[ite]


def local_mteb(seed=0, n=6000, d=96, classes=12):
    rng = np.random.default_rng(seed)
    centers = rng.normal(size=(classes * 3, d))
    lab = rng.integers(0, classes * 3, n)
    Z = 0.35 * centers[lab] + rng.normal(size=(n, d))
    Z += 0.3 * np.sin(2 * Z[:, ::-1])
    Z = (Z / np.linalg.norm(Z, axis=1, keepdims=True)).astype(np.float32)
    y = lab % classes
    return Z[:5000], y[:5000], Z[5000:], y[5000:]


def fit_embedding(method, Xtr, ytr, Xva, yva, seed, trees, mt_extra):
    """Fits on (Xtr, ytr), using (Xva, yva) only for selection / early stopping."""
    if method == "linear":
        cands = [LogisticRegression(C=c, max_iter=2000).fit(Xtr, ytr) for c in (0.1, 1.0, 10.0, 100.0)]
        return max(cands, key=lambda m: m.score(Xva, yva))
    if method == "knn":
        cands = [KNeighborsClassifier(n_neighbors=k, weights="distance", algorithm="brute").fit(Xtr, ytr)
                 for k in (5, 15, 30)]
        return max(cands, key=lambda m: m.score(Xva, yva))
    if method == "mlp":
        return MLPClassifier(hidden_layer_sizes=(512,), early_stopping=True, max_iter=200,
                             random_state=seed).fit(Xtr, ytr)
    if method == "hgb":
        return HistGradientBoostingClassifier(random_state=seed).fit(Xtr, ytr)
    if method in MT_METHODS:
        return make_mt(MT_METHODS[method], mt_extra, trees, seed).fit(Xtr, ytr, Xva, yva)
    raise ValueError(f"unknown method {method!r}")


def run_mteb(names, args, rec):
    methods = args.methods.split(",") if args.methods else MTEB_DEFAULT.split(",")
    mt_extra = json.loads(args.mt_params) if args.mt_params else {}
    for name in names:
        if rec.task_done("mteb", name, args.seeds, methods):
            continue
        try:
            Xtr_all, ytr_all, Xte, yte = (local_mteb() if args.local else
                                          embed_mteb(name, args.text_model, args.cache,
                                                     args.max_train, args.max_test))
        except Exception as e:
            print(f"[mteb] {name}: load failed ({type(e).__name__}: {str(e)[:120]})", flush=True)
            continue
        C = int(max(ytr_all.max(), yte.max())) + 1
        print(f"\n[mteb] {name}: train {len(ytr_all)} test {len(yte)} d={Xtr_all.shape[1]} C={C}", flush=True)
        for seed in range(args.seeds):
            try:
                itr, iva = train_test_split(np.arange(len(ytr_all)), test_size=0.1,
                                            stratify=ytr_all, random_state=seed)
            except ValueError:
                itr, iva = train_test_split(np.arange(len(ytr_all)), test_size=0.1, random_state=seed)
            for meth in methods:
                if rec.done("mteb", name, seed, meth):
                    continue
                try:
                    t1 = time.perf_counter()
                    m = fit_embedding(meth, Xtr_all[itr], ytr_all[itr], Xtr_all[iva], ytr_all[iva],
                                      seed, args.trees, mt_extra)
                    t2 = time.perf_counter()
                    P = full_proba(m, Xte, C)
                    t3 = time.perf_counter()
                except Exception as e:
                    print(f"  seed {seed} {meth}: FAILED {type(e).__name__}: {str(e)[:120]}", flush=True)
                    continue
                acc, auc, ll = scores(P, yte, C)
                rec.add("mteb", name, name, seed, meth, len(itr), len(yte), Xte.shape[1], C,
                        acc, auc, ll, t2 - t1, t3 - t2)
                print(f"  seed {seed} {meth:>9}  acc {100 * acc:6.2f}  fit {t2 - t1:6.1f}s", flush=True)


# ============================================================================= results

class Recorder:
    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.keys = set()
        if os.path.exists(path):
            for r in csv.DictReader(open(path)):
                self.keys.add((r["suite"], str(r["task"]), str(r["fold"]), r["method"]))
        new = not os.path.exists(path)
        self.f = open(path, "a", newline="")
        self.w = csv.writer(self.f)
        if new:
            self.w.writerow(FIELDS)

    def done(self, suite, task, fold, method):
        return (suite, str(task), str(fold), method) in self.keys

    def task_done(self, suite, task, n_folds, methods):
        task = task[0] if isinstance(task, tuple) else task
        return all(self.done(suite, task, f, m) for f in range(n_folds) for m in methods)

    def add(self, *row):
        self.w.writerow(row)
        self.f.flush()
        self.keys.add((row[0], str(row[1]), str(row[3]), row[4]))


def summarize(path, ref="hgb", out_md=None):
    if not os.path.exists(path):
        print("no results yet")
        return
    df = pd.read_csv(path)
    lines = []
    for suite, g in df.groupby("suite", sort=False):
        per = g.groupby(["dataset", "method"]).agg(
            acc=("acc", "mean"), auc=("auc", "mean"), logloss=("logloss", "mean"),
            fit_s=("fit_s", "median")).reset_index()
        acc = per.pivot(index="dataset", columns="method", values="acc").dropna()   # datasets every method finished
        if acc.empty:
            continue
        methods = list(acc.columns)
        ranks = acc.rank(axis=1, ascending=False).mean()
        lo, hi = acc.min(1), acc.max(1)
        norm = acc.sub(lo, axis=0).div((hi - lo).replace(0, np.nan), axis=0).fillna(1.0).mean()
        auc = per.pivot(index="dataset", columns="method", values="auc").reindex(acc.index)
        ll = per.pivot(index="dataset", columns="method", values="logloss").reindex(acc.index)
        fit = per.pivot(index="dataset", columns="method", values="fit_s").reindex(acc.index)
        lines.append(f"\n### {suite}: {len(acc)} datasets (every method finished), mean over folds/seeds\n")
        lines.append("| method | mean acc | mean rank | normalized acc | W/T/L vs " + ref +
                     " | median Δacc vs " + ref + " | mean AUC (binary) | mean log-loss (multiclass) | median fit s |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for m in sorted(methods, key=lambda m: ranks[m]):
            wtl = med = "–"
            if ref in acc and m != ref:
                d = 100 * (acc[m] - acc[ref])
                wtl = f"{(d > 0.1).sum()}/{(d.abs() <= 0.1).sum()}/{(d < -0.1).sum()}"
                med = f"{d.median():+.2f}"
            a, l_ = auc[m].dropna(), ll[m].dropna()
            a_s = f"{a.mean():.4f} ({len(a)})" if len(a) else "–"
            l_s = f"{l_.mean():.4f} ({len(l_)})" if len(l_) else "–"
            lines.append(f"| {m} | {100 * acc[m].mean():.2f} | {ranks[m]:.2f} | {norm[m]:.3f} | {wtl} | {med} | "
                         f"{a_s} | {l_s} | {fit[m].median():.2f} |")
        missing = sorted(set(per.dataset) - set(acc.index))
        if missing:
            lines.append(f"\n(not in the table, some method missing: {', '.join(map(str, missing[:15]))}"
                         f"{' …' if len(missing) > 15 else ''})")
    text = "\n".join(lines)
    print(text)
    if out_md:
        with open(out_md, "w") as fh:
            fh.write("W/T/L: datasets where the method beats / ties (within 0.1 points) / loses to the reference.\n"
                     "Normalized acc: per dataset, 0 = worst method, 1 = best, averaged.\n" + text + "\n")
        print(f"\nsaved {out_md}")


# ============================================================================= main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--suites", default="cc18,grinsztajn,tabarena,mteb")
    ap.add_argument("--methods", default=None, help=f"tabular default {TABULAR_DEFAULT}; mteb default {MTEB_DEFAULT}")
    ap.add_argument("--folds", type=int, default=3, help="OpenML folds to run (of the task's predefined ones)")
    ap.add_argument("--seeds", type=int, default=3, help="MTEB: validation splits / model seeds")
    ap.add_argument("--trees", type=int, default=1024)
    ap.add_argument("--max-train", type=int, default=50000)
    ap.add_argument("--max-test", type=int, default=20000)
    ap.add_argument("--max-features", type=int, default=5000)
    ap.add_argument("--skip", default="", help="comma-separated OpenML task ids to skip")
    ap.add_argument("--mt-params", default=None, help='JSON added to every median-tree method, e.g. \'{"refit": true}\'')
    ap.add_argument("--mteb-datasets", default=",".join(MTEB))
    ap.add_argument("--text-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--cache", default="cache")
    ap.add_argument("--out", default="results/suites.csv")
    ap.add_argument("--ref", default="hgb", help="reference method for W/T/L in the summary")
    ap.add_argument("--summary", action="store_true", help="only print the summary of existing results")
    ap.add_argument("--local", action="store_true", help="offline smoke test on stand-in data")
    args = ap.parse_args()
    if args.local:
        if args.out == "results/suites.csv":
            args.out = "results/suites_local.csv"
        args.trees = min(args.trees, 64)
    if args.summary:
        summarize(args.out, args.ref, args.out.replace(".csv", "_summary.md"))
        return

    # compile the numba kernels before anything is timed
    Xw = np.random.default_rng(0).normal(size=(600, 10)).astype(np.float32)
    for yw in ((Xw[:, 0] > 0).astype(int), (Xw[:, 0] > 0).astype(int) + (Xw[:, 1] > 0)):
        for p in MT_METHODS.values():
            make_mt(p, {}, 8, 0).fit(Xw, yw).predict_proba(Xw)

    rec = Recorder(args.out)
    t0 = time.time()
    for suite in args.suites.split(","):
        if suite == "mteb":
            run_mteb(["synthetic"] if args.local else args.mteb_datasets.split(","), args, rec)
        elif suite in OPENML_SUITES:
            if args.local:
                tasks = local_tabular_tasks(args.folds)
            else:
                try:
                    tasks = openml_tasks(suite)
                except Exception as e:
                    print(f"[{suite}] could not list the suite ({type(e).__name__}: {str(e)[:120]})")
                    continue
                print(f"\n===== {suite}: {len(tasks)} tasks =====", flush=True)
            run_tabular(suite, tasks, args, rec)
        else:
            print(f"unknown suite {suite!r}; choose from {list(OPENML_SUITES) + ['mteb']}")
    print(f"\ndone in {(time.time() - t0) / 60:.1f} min")
    summarize(args.out, args.ref, args.out.replace(".csv", "_summary.md"))


if __name__ == "__main__":
    main()
