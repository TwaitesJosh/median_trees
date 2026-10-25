import pickle

import numpy as np
import pytest
import scipy.sparse as sp
from sklearn.base import clone
from sklearn.datasets import load_breast_cancer, load_digits

from median_trees import MedianTreeClassifier, MedianTreeEmbedder

SMALL = dict(n_trees=32, max_epochs=5)


@pytest.fixture(scope="module")
def binary():
    X, y = load_breast_cancer(return_X_y=True)
    return X.astype(np.float32), y


@pytest.fixture(scope="module")
def multiclass():
    X, y = load_digits(return_X_y=True)
    return X.astype(np.float32), y


def test_binary_fit_predict(binary):
    X, y = binary
    m = MedianTreeClassifier(random_state=0, **SMALL).fit(X, y)
    P = m.predict_proba(X)
    assert P.shape == (len(y), 2)
    np.testing.assert_allclose(P.sum(1), 1.0, atol=1e-5)
    assert (m.predict(X) == y).mean() > 0.9


def test_multiclass_fit_predict(multiclass):
    X, y = multiclass
    m = MedianTreeClassifier(random_state=0, **SMALL).fit(X, y)
    assert m.predict_proba(X).shape == (len(y), 10)
    assert (m.predict(X) == y).mean() > 0.9


def test_string_labels(binary):
    X, y = binary
    labels = np.array(["no", "yes"])[y]
    m = MedianTreeClassifier(random_state=0, **SMALL).fit(X, labels)
    assert set(m.predict(X)) <= {"no", "yes"}


def test_explicit_validation_set(binary):
    X, y = binary
    m = MedianTreeClassifier(random_state=0, **SMALL).fit(X[:400], y[:400], X[400:], y[400:])
    assert (m.predict(X[400:]) == y[400:]).mean() > 0.85


def test_missing_values(binary):
    X, y = binary
    X = X.copy()
    X[np.random.default_rng(0).random(X.shape) < 0.2] = np.nan
    m = MedianTreeClassifier(random_state=0, **SMALL).fit(X, y)
    assert np.isfinite(m.predict_proba(X)).all()


def test_rejects_infinity(binary):
    X, y = binary
    X = X.copy()
    X[0, 0] = np.inf
    with pytest.raises(ValueError):
        MedianTreeClassifier(**SMALL).fit(X, y)


def test_reproducible(binary):
    X, y = binary
    a = MedianTreeClassifier(random_state=3, **SMALL).fit(X, y).predict_proba(X)
    b = MedianTreeClassifier(random_state=3, **SMALL).fit(X, y).predict_proba(X)
    # Hogwild SGD is parallel, so allow tiny floating-point differences
    assert np.abs(a - b).max() < 0.05


def test_clone_and_pickle(binary):
    X, y = binary
    m = MedianTreeClassifier(random_state=0, **SMALL)
    assert clone(m).get_params() == m.get_params()
    m.fit(X, y)
    m2 = pickle.loads(pickle.dumps(m))
    np.testing.assert_allclose(m.predict_proba(X), m2.predict_proba(X))


def test_raw_features_and_probe(multiclass):
    X, y = multiclass
    for kw in (dict(raw_features=True), dict(probe=True, n_trees=32)):
        p = {**SMALL, **kw}
        m = MedianTreeClassifier(random_state=0, **p).fit(X, y)
        assert (m.predict(X) == y).mean() > 0.85


def test_embedder_transform(binary):
    X, y = binary
    emb = MedianTreeEmbedder(n_trees=16, depth=5, random_state=0).fit(X)
    F = emb.transform(X)
    assert sp.issparse(F) and F.shape == (len(X), 16 * 32)
    np.testing.assert_array_equal(np.asarray(F.sum(1)).ravel(), 16)
    assert 0.5 < emb.leaf_balance(X) <= 1.0
