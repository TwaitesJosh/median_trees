"""Median trees: random trees cut at data quantiles, a sparse linear head on their leaves."""

from ._core import (LeafSGD, MedianTreeClassifier, MedianTreeEmbedder,
                    TunedMedianTreeClassifier)

__all__ = ["MedianTreeClassifier", "MedianTreeEmbedder", "TunedMedianTreeClassifier", "LeafSGD"]
__version__ = "0.1.0"
