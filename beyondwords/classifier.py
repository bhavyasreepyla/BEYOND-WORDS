"""A small MLP stored as plain JSON so Python (numpy) and the browser (JS) can both run it."""

import json

import numpy as np

from . import CLASSIFIER_PATH


class Classifier:
    def __init__(self, labels, mean, std, layers):
        self.labels = list(labels)
        self.mean = np.asarray(mean, dtype=np.float32)
        self.std = np.asarray(std, dtype=np.float32)
        self.layers = [(np.asarray(l["W"], dtype=np.float32), np.asarray(l["b"], dtype=np.float32)) for l in layers]

    @classmethod
    def load(cls, path=CLASSIFIER_PATH):
        with open(path) as f:
            data = json.load(f)
        return cls(data["labels"], data["mean"], data["std"], data["layers"])

    def predict_proba(self, features):
        h = (np.atleast_2d(features) - self.mean) / self.std
        for i, (W, b) in enumerate(self.layers):
            h = h @ W + b
            if i < len(self.layers) - 1:
                h = np.maximum(h, 0)
        h = np.exp(h - h.max(axis=1, keepdims=True))
        return h / h.sum(axis=1, keepdims=True)

    def predict(self, features):
        """(label, confidence) for a single feature vector."""
        proba = self.predict_proba(features)[0]
        i = int(proba.argmax())
        return self.labels[i], float(proba[i])


def save(path, labels, mean, std, weights, biases, metrics=None):
    data = {
        "labels": list(labels),
        "mean": np.round(mean, 6).tolist(),
        "std": np.round(std, 6).tolist(),
        "layers": [{"W": np.round(W, 6).tolist(), "b": np.round(b, 6).tolist()} for W, b in zip(weights, biases)],
        "metrics": metrics or {},
    }
    with open(path, "w") as f:
        json.dump(data, f, separators=(",", ":"))
