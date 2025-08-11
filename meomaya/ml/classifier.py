import math
from typing import Dict, List


class Classifier:
    """Pure-Python centroid-based classifier (nearest mean in cosine space)."""

    def __init__(self):
        self.class_to_centroid: Dict[str, List[float]] = {}
        self._fitted = False

    def train(self, X: List[List[float]], y: List[str]):
        """
        Trains the classifier.

        Args:
            X (List[List[float]]): A list of feature vectors.
            y (List[str]): A list of labels.

        Raises:
            TypeError: If inputs are not of the expected types.
            ValueError: If inputs are empty or have inconsistent dimensions.
        """
        # Validate input types
        if not isinstance(X, list):
            raise TypeError("Input X must be a list.")
        if not isinstance(y, list):
            raise TypeError("Input y must be a list.")

        # Check for empty inputs
        if not X:
            raise ValueError("Input X cannot be empty.")
        if not y:
            raise ValueError("Input y cannot be empty.")

        # Validate X structure and content
        if not all(isinstance(i, list) for i in X):
            raise TypeError("Input X must be a list of lists.")

        # Check that all elements in X are numbers
        for i, row in enumerate(X):
            if not all(isinstance(j, (int, float)) for j in row):
                raise TypeError(
                    f"All elements in X must be numbers. Found non-numeric value in row {i}."
                )

        # Validate y content
        if not all(isinstance(i, str) for i in y):
            raise TypeError("All elements in y must be strings.")

        # Check dimensions match
        if len(X) != len(y):
            raise ValueError(
                f"X and y must have the same length. X has {len(X)} samples, y has {len(y)} labels."
            )

        # Check that all feature vectors have the same length
        if len(set(len(row) for row in X)) > 1:
            raise ValueError("All feature vectors in X must have the same length.")

        # Check for NaN or infinite values
        for i, row in enumerate(X):
            for j, val in enumerate(row):
                if not (
                    isinstance(val, (int, float))
                    and not (val != val or abs(val) == float("inf"))
                ):
                    raise ValueError(
                        f"Found NaN or infinite value at position [{i}][{j}] in X."
                    )

        # Compute centroid per class
        sums: Dict[str, List[float]] = {}
        counts: Dict[str, int] = {}
        for vec, label in zip(X, y):
            if label not in sums:
                sums[label] = [0.0] * len(vec)
                counts[label] = 0
            row = sums[label]
            for i, v in enumerate(vec):
                row[i] += float(v)
            counts[label] += 1
        self.class_to_centroid = {
            label: [v / max(1, counts[label]) for v in sums[label]] for label in sums
        }
        self._fitted = True

    def classify(self, X: List[List[float]]) -> List[str]:
        """
        Classifies a list of feature vectors.

        Args:
            X (List[List[float]]): A list of feature vectors.

        Returns:
            List[str]: A list of predicted labels.

        Raises:
            TypeError: If input is not of the expected type.
            ValueError: If input is empty or has inconsistent dimensions.
        """
        # Validate input type
        if not isinstance(X, list):
            raise TypeError("Input X must be a list.")

        # Check for empty input
        if not X:
            raise ValueError("Input X cannot be empty.")

        # Validate X structure and content
        if not all(isinstance(i, list) for i in X):
            raise TypeError("Input X must be a list of lists.")

        # Check that all elements in X are numbers
        for i, row in enumerate(X):
            if not all(isinstance(j, (int, float)) for j in row):
                raise TypeError(
                    f"All elements in X must be numbers. Found non-numeric value in row {i}."
                )

        # Check that all feature vectors have the same length
        if len(set(len(row) for row in X)) > 1:
            raise ValueError("All feature vectors in X must have the same length.")

        # Check for NaN or infinite values
        for i, row in enumerate(X):
            for j, val in enumerate(row):
                if not (
                    isinstance(val, (int, float))
                    and not (val != val or abs(val) == float("inf"))
                ):
                    raise ValueError(
                        f"Found NaN or infinite value at position [{i}][{j}] in X."
                    )

        if not self._fitted:
            raise ValueError("Classifier not trained. Call train() first.")
        preds: List[str] = []
        for vec in X:
            best_label = None
            best_score = -1.0
            for label, centroid in self.class_to_centroid.items():
                score = _cosine_similarity(vec, centroid)
                if score > best_score:
                    best_score = score
                    best_label = label
            preds.append(best_label or "unknown")
        return preds


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = 0.0
    na = 0.0
    nb = 0.0
    for x, y in zip(a, b):
        dot += x * y
        na += x * x
        nb += y * y
    if na == 0 or nb == 0:
        return 0.0
    return dot / math.sqrt(na * nb)
