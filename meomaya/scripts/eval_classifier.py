from __future__ import annotations

import argparse
import json
from typing import List

from meomaya.ml.vectorizer import Vectorizer
from meomaya.ml.classifier import Classifier


def load_jsonl(path: str) -> List[dict]:
    rows: List[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=str)
    parser.add_argument("eval_jsonl", type=str)
    args = parser.parse_args()

    # Load artifacts
    with open(f"{args.model_dir}/vectorizer.json", "r", encoding="utf-8") as f:
        vec_data = json.load(f)
    vect = Vectorizer()
    vect.vocab = vec_data.get("vocab", {})
    vect.vocabulary = dict(vect.vocab)
    vect.idf = vec_data.get("idf", {})
    vect._fitted = True

    with open(f"{args.model_dir}/classifier.json", "r", encoding="utf-8") as f:
        cls_data = json.load(f)
    cls = Classifier()
    cls.class_to_centroid = cls_data.get("centroids", {})
    cls._fitted = True

    data = load_jsonl(args.eval_jsonl)
    texts = [d["text"] for d in data]
    labels = [d["label"] for d in data]
    X = vect.transform(texts)
    preds = cls.classify(X)
    correct = sum(1 for p, y in zip(preds, labels) if p == y)
    acc = correct / max(1, len(labels))
    print(json.dumps({"accuracy": acc, "n": len(labels)}, indent=2))


if __name__ == "__main__":
    main()


