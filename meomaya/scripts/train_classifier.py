from __future__ import annotations

import argparse
import json
import os
import random
from typing import List

import numpy as np

from meomaya.ml.vectorizer import Vectorizer
from meomaya.ml.classifier import Classifier


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)


def load_jsonl(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("train_jsonl", type=str, help="JSONL with fields: text, label")
    parser.add_argument("out_dir", type=str, help="Output directory")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    data = load_jsonl(args.train_jsonl)
    texts = [d["text"] for d in data]
    labels = [d["label"] for d in data]

    vect = Vectorizer()
    X = vect.fit_transform(texts)
    cls = Classifier()
    cls.train(X, labels)

    # Save artifacts
    with open(os.path.join(args.out_dir, "vectorizer.json"), "w", encoding="utf-8") as f:
        json.dump({"vocab": vect.vocab, "idf": vect.idf}, f)
    with open(os.path.join(args.out_dir, "classifier.json"), "w", encoding="utf-8") as f:
        json.dump({"centroids": cls.class_to_centroid}, f)
    print("Saved to", args.out_dir)


if __name__ == "__main__":
    main()


