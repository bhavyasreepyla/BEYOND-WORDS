"""Train the landmark classifier and export it for the Python app and the web demo.

    python scripts/train.py                 # uses data/landmarks.npz
    python scripts/train.py --extra data/my_samples.npz   # add your own recorded samples
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from beyondwords import CLASSIFIER_PATH  # noqa: E402
from beyondwords.classifier import save  # noqa: E402
from beyondwords.labels import LABELS  # noqa: E402
from beyondwords.landmarks import NUM_LANDMARKS, landmarks_to_features  # noqa: E402


def load(path):
    d = np.load(path)
    widths, heights = d["sizes"][:, 0:1], d["sizes"][:, 1:2]
    pts = d["landmarks"] * np.stack([widths, heights, widths], axis=-1)
    return landmarks_to_features(pts, 1, 1), d["labels"]


def augment(X, rng, copies):
    """Mirrored (left-handed) versions plus small rotations and jitter of every sample."""
    pts = X.reshape(-1, NUM_LANDMARKS, 3)
    out = [pts, pts * np.array([-1, 1, 1], dtype=np.float32)]
    for _ in range(copies):
        base = out[rng.integers(0, 2)]
        theta = rng.uniform(-0.35, 0.35, size=len(base))
        c, s = np.cos(theta), np.sin(theta)
        rot = np.zeros((len(base), 3, 3), dtype=np.float32)
        rot[:, 0, 0], rot[:, 0, 1], rot[:, 1, 0], rot[:, 1, 1], rot[:, 2, 2] = c, -s, s, c, 1
        p = base @ rot.transpose(0, 2, 1)
        p[..., 2] *= rng.uniform(0.8, 1.2, size=(len(base), 1)).astype(np.float32)
        p = p + rng.normal(0, 0.012, size=p.shape).astype(np.float32)
        out.append(p)
    pts = np.concatenate(out)
    return landmarks_to_features(pts, 1, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(ROOT / "data/landmarks.npz"))
    parser.add_argument("--extra", nargs="*", default=[], help="Additional .npz files, e.g. from collect_samples.py")
    parser.add_argument("--holdout", help="Separate landmarks .npz used only for evaluation (e.g. a different signer)")
    parser.add_argument("--augment", type=int, default=3)
    args = parser.parse_args()

    rng = np.random.default_rng(0)
    X, y = load(args.data)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, stratify=y, random_state=0)
    for extra in args.extra:
        Xe, ye = load(extra)
        X_train, y_train = np.concatenate([X_train, Xe]), np.concatenate([y_train, ye])

    n = len(y_train)
    X_aug = augment(X_train, rng, args.augment)
    y_aug = np.tile(y_train, len(X_aug) // n)
    print(f"Training on {len(X_aug)} samples ({n} originals + augmentation), validating on {len(X_val)}")

    mean, std = X_aug.mean(axis=0), X_aug.std(axis=0) + 1e-6
    clf = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        alpha=1e-4,
        batch_size=512,
        learning_rate_init=1e-3,
        max_iter=60,
        early_stopping=True,
        n_iter_no_change=6,
        random_state=0,
        verbose=False,
    )
    clf.fit((X_aug - mean) / std, y_aug)

    def evaluate(Xe, ye, name):
        pred = clf.predict((Xe - mean) / std)
        acc = float((pred == ye).mean())
        print(f"\n{name} accuracy: {acc:.2%} on {len(ye)} samples")
        return acc, pred

    val_acc, val_pred = evaluate(X_val, y_val, "Validation")
    print(classification_report(y_val, val_pred, labels=range(len(LABELS)), target_names=LABELS, digits=3, zero_division=0))
    metrics = {"val_accuracy": round(val_acc, 4), "val_samples": int(len(y_val)), "train_samples": int(n)}

    if args.holdout:
        Xh, yh = load(args.holdout)
        hold_acc, _ = evaluate(Xh, yh, "Holdout (unseen signer)")
        metrics.update(holdout_accuracy=round(hold_acc, 4), holdout_samples=int(len(yh)))

    # sklearn classes_ are sorted label indices; keep only labels that were actually present
    labels = [LABELS[i] for i in clf.classes_]
    CLASSIFIER_PATH.parent.mkdir(parents=True, exist_ok=True)
    save(CLASSIFIER_PATH, labels, mean, std, clf.coefs_, clf.intercepts_, metrics)
    web_copy = ROOT / "web" / "model" / CLASSIFIER_PATH.name
    web_copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(CLASSIFIER_PATH, web_copy)

    cm = confusion_matrix(y_val, val_pred, labels=range(len(LABELS)))
    (ROOT / "models" / "metrics.json").write_text(json.dumps({**metrics, "labels": LABELS, "confusion": cm.tolist()}))
    print(f"\nSaved {CLASSIFIER_PATH} and {web_copy}")


if __name__ == "__main__":
    main()
