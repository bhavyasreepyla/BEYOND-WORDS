"""Run MediaPipe (hands + body) over PopSign ASL clips and save raw landmark sequences.

    python scripts/extract_signs.py --split train
    python scripts/extract_signs.py --split validation
    python scripts/extract_signs.py --subset game --split train --download

Source: https://huggingface.co/datasets/sign/popsign-images (CC BY 4.0). Each parquet
shard becomes data/signs/<subset>-<split>/<shard>.npz; finished shards are skipped,
so an interrupted run can be resumed. With --download, shards are fetched one at a
time (the next one downloads while the current one is processed) and deleted after.
"""

import argparse
import os
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
REPO = "sign/popsign-images"

_hands = _pose = None


def _init_worker():
    global _hands, _pose
    from beyondwords.landmarks import create_landmarker
    from beyondwords.signs import create_pose_landmarker

    _hands = create_landmarker(min_confidence=0.3, num_hands=2)
    _pose = create_pose_landmarker(min_confidence=0.3)


def _process(frames):
    import cv2

    from beyondwords.landmarks import to_mp_image
    from beyondwords.signs import frame_record

    records = []
    for png in frames:
        bgr = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)
        image = to_mp_image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        records.append(frame_record(_hands.detect(image), _pose.detect(image), bgr.shape[1], bgr.shape[0]))
    return records, bgr.shape[1]


def extract_shard(pool, shard, out, every=1):
    import pyarrow.parquet as pq

    table = pq.read_table(shard, columns=["file", "text", "images"])
    if every > 1:
        table = table.take(list(range(0, table.num_rows, every)))
    meta = table.select(["file", "text"]).to_pylist()
    clips = [[img["bytes"] for img in row] for row in table.column("images").to_pylist()]
    del table

    poses, hands, has, offsets, labels, files, widths = [], [], [], [0], [], [], []
    for m, (records, width) in zip(meta, pool.map(_process, clips, chunksize=4)):
        if not records:
            continue
        poses += [r["pose"] for r in records]
        hands += [r["hands"] for r in records]
        has += [r["has"] for r in records]
        offsets.append(offsets[-1] + len(records))
        labels.append(m["text"])
        files.append(m["file"])
        widths.append(width)
    np.savez_compressed(
        out, pose=np.stack(poses), hands=np.stack(hands), has=np.stack(has), offsets=np.array(offsets),
        labels=np.array(labels), files=np.array(files), widths=np.array(widths),
    )
    return len(labels), len(has)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subset", default="non-game", choices=["non-game", "game"])
    parser.add_argument("--split", default="train", choices=["train", "validation", "test"])
    parser.add_argument("--src", help="Folder with parquet shards (default data/popsign/data/<subset>)")
    parser.add_argument("--download", action="store_true", help="Fetch shards from Hugging Face one by one")
    parser.add_argument("--every", type=int, default=1, help="Keep every Nth clip (shards are sorted by sign, so all signs stay covered)")
    # MediaPipe runs its own threads per process; past ~12 workers they just contend
    parser.add_argument("--workers", type=int, default=min(12, os.cpu_count() or 2))
    args = parser.parse_args()

    src = Path(args.src or ROOT / f"data/popsign/data/{args.subset}")
    out_dir = ROOT / f"data/signs/{args.subset}-{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.download:
        from huggingface_hub import HfApi, hf_hub_download

        names = sorted(
            f for f in HfApi().list_repo_files(REPO, repo_type="dataset")
            if f.startswith(f"data/{args.subset}/{args.split}-") and f.endswith(".parquet")
        )
        names = [n for n in names if not (out_dir / f"{Path(n).stem}.npz").exists()]
        fetch = lambda n: Path(hf_hub_download(REPO, n, repo_type="dataset", local_dir=ROOT / "data/popsign"))  # noqa: E731
    else:
        names = [p for p in sorted(src.glob(f"{args.split}-*.parquet")) if not (out_dir / f"{p.stem}.npz").exists()]
        fetch = Path
    print(f"{len(names)} shards to process into {out_dir}", flush=True)

    with ProcessPoolExecutor(args.workers, initializer=_init_worker) as pool, ThreadPoolExecutor(1) as downloader:
        pending = downloader.submit(fetch, names[0]) if names else None
        for i, name in enumerate(names):
            shard = pending.result()
            pending = downloader.submit(fetch, names[i + 1]) if i + 1 < len(names) else None
            clips, frames = extract_shard(pool, shard, out_dir / f"{shard.stem}.npz", args.every)
            if args.download:
                shard.unlink()
            print(f"  [{i + 1}/{len(names)}] {shard.name}: {clips} clips, {frames} frames", flush=True)


if __name__ == "__main__":
    main()
