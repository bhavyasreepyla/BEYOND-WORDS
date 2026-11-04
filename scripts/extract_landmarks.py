"""Run MediaPipe over the Kaggle ASL Alphabet images and save hand landmarks.

    python scripts/extract_landmarks.py --data <path to asl_alphabet_train> --per-class 3000

Images where no hand is found are skipped (that's also why the "nothing" class
is dropped: at runtime "no hand detected" already means nothing).
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from beyondwords.labels import LABELS  # noqa: E402

_landmarker = None


def _init_worker():
    global _landmarker
    from beyondwords.landmarks import create_landmarker

    _landmarker = create_landmarker(min_confidence=0.3)


def _process(job):
    import cv2

    from beyondwords.landmarks import result_to_array, to_mp_image

    path, label = job
    bgr = cv2.imread(path)
    if bgr is None:
        return None
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    lm = result_to_array(_landmarker.detect(to_mp_image(rgb)))
    if lm is None:
        return None
    return lm, label, rgb.shape[1], rgb.shape[0]


def find_train_dir(root):
    root = Path(root)
    for candidate in [root, *root.rglob("*")]:
        if candidate.is_dir() and (candidate / "A").is_dir() and (candidate / "B").is_dir():
            return candidate
    raise SystemExit(f"Could not find class folders (A/, B/, ...) under {root}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", help="Dataset folder. Downloads it with kagglehub if omitted.")
    parser.add_argument("--per-class", type=int, default=3000)
    parser.add_argument("--out", default="data/landmarks.npz")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    args = parser.parse_args()

    if args.data is None:
        import kagglehub

        args.data = kagglehub.dataset_download("grassknoted/asl-alphabet")
    train_dir = find_train_dir(args.data)

    jobs = []
    for idx, label in enumerate(LABELS):
        files = sorted((train_dir / label).iterdir())[: args.per_class]
        jobs += [(str(f), idx) for f in files]
    print(f"{len(jobs)} images from {train_dir} using {args.workers} workers")

    results = []
    with ProcessPoolExecutor(args.workers, initializer=_init_worker) as pool:
        for i, res in enumerate(pool.map(_process, jobs, chunksize=64), 1):
            if res is not None:
                results.append(res)
            if i % 5000 == 0:
                print(f"  {i}/{len(jobs)} processed, {len(results)} hands found")

    X = np.stack([r[0] for r in results])
    y = np.array([r[1] for r in results])
    sizes = np.array([(r[2], r[3]) for r in results])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, landmarks=X, labels=y, sizes=sizes)

    print(f"Saved {len(X)} samples to {args.out} ({len(X) / len(jobs):.1%} detection rate)")
    counts = np.bincount(y, minlength=len(LABELS))
    print("  " + "  ".join(f"{l}:{c}" for l, c in zip(LABELS, counts)))


if __name__ == "__main__":
    main()
