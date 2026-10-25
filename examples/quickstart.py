"""Median trees in a few lines, against scikit-learn's gradient boosting."""
import time

from sklearn.datasets import fetch_openml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split

from median_trees import MedianTreeClassifier

# Devnagari handwritten characters: 92,000 images, 46 classes (OpenML 40923)
X, y = fetch_openml(data_id=40923, return_X_y=True, as_frame=False)
X = X.astype("float32")
X_tr, X_te, y_tr, y_te = train_test_split(X, y, train_size=30000, test_size=10000, stratify=y, random_state=0)

for name, model in [("median trees", MedianTreeClassifier(random_state=0)),
                    ("gradient boosting", HistGradientBoostingClassifier(random_state=0))]:
    t = time.time()
    model.fit(X_tr, y_tr)
    acc = (model.predict(X_te) == y_te).mean()
    print(f"{name:>18}: accuracy {acc:.3f}, trained in {time.time() - t:.0f}s")
