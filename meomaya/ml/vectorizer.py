import math
from typing import Dict, List


def _tokenize(text: str) -> List[str]:
    return [tok for tok in text.lower().split() if tok]


class Vectorizer:
    """Pure-Python TF-IDF implementation with train/eval consistency."""

    def __init__(self):
        self._fitted = False
        self.vocab: Dict[str, int] = {}
        self.vocabulary = {}
        self.idf = {}

    def fit(self, texts: List[str]) -> None:
        if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
            raise TypeError("Input to fit() must be a list of strings.")
        
        # Handle empty texts case
        if not texts:
            self.vocab = {}
            self.vocabulary = {}
            self.idf = {}
            self._fitted = False  # Don't mark as fitted if no data
            return
            
        # Build vocabulary and document frequencies
        doc_freq: Dict[str, int] = {}
        for text in texts:
            tokens = set(_tokenize(text))
            for tok in tokens:
                doc_freq[tok] = doc_freq.get(tok, 0) + 1
        # Freeze vocab (sorted for determinism)
        self.vocab = {tok: i for i, tok in enumerate(sorted(doc_freq))}
        self.vocabulary = dict(self.vocab)
        N = max(1, len(texts))
        # Smooth idf
        if doc_freq:
            self.idf = {
                tok: math.log((1 + N) / (1 + doc_freq[tok])) + 1.0
                for tok in sorted(doc_freq)
            }
        else:
            self.idf = {}
        self._fitted = True

    def transform(self, texts: List[str]) -> List[List[float]]:
        if not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
            raise TypeError("Input to transform() must be a list of strings.")
        if not self._fitted:
            raise ValueError(
                "Vectorizer not fitted. Call fit() first or use fit_transform()."
            )
        vectors: List[List[float]] = []
        V = len(self.vocab)
        idf_list = [self.idf.get(tok, 0.0) for tok in sorted(self.vocab)]
        # Optional NumPy acceleration if available
        try:
            import numpy as np  # type: ignore

            idf_arr = np.array(idf_list, dtype=float)
            for text in texts:
                tokens = _tokenize(text)
                counts: Dict[int, int] = {}
                for tok in tokens:
                    idx = self.vocab.get(tok)
                    if idx is not None:
                        counts[idx] = counts.get(idx, 0) + 1
                if not counts:
                    vectors.append([0.0] * V)
                    continue
                idxs = np.fromiter(counts.keys(), dtype=int)
                cnts = np.fromiter(counts.values(), dtype=float)
                total = float(cnts.sum()) or 1.0
                tf = cnts / total
                vec = np.zeros(V, dtype=float)
                vec[idxs] = tf * idf_arr[idxs]
                vectors.append(vec.tolist())
        except Exception:
            for text in texts:
                tokens = _tokenize(text)
                counts: Dict[int, int] = {}
                for tok in tokens:
                    idx = self.vocab.get(tok)
                    if idx is not None:
                        counts[idx] = counts.get(idx, 0) + 1
                total = sum(counts.values()) or 1
                vec = [0.0] * V
                for idx, cnt in counts.items():
                    tf = cnt / total
                    vec[idx] = tf * idf_list[idx]
                vectors.append(vec)
        return vectors

    def fit_transform(self, texts: List[str]) -> List[List[float]]:
        self.fit(texts)
        return self.transform(texts)

    def vectorize(self, texts: List[str]) -> List[List[float]]:
        if not self._fitted:
            return self.fit_transform(texts)
        return self.transform(texts)
