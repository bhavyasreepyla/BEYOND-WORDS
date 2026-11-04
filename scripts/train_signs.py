"""Train the word-sign model on PopSign landmark clips and export it to ONNX.

    python scripts/train_signs.py      # trains on data/signs/*-train, validates on data/signs/non-game-validation

Writes models/<name>.onnx + models/<name>.json and copies both to web/model/.
"""

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from beyondwords.signs import FRAME_FEATURES, SEQ_LEN, clip_features, trim_to_hands  # noqa: E402

DISPLAY = {
    "TV": "TV", "callonphone": "call", "frenchfries": "french fries", "glasswindow": "window",
    "haveto": "have to", "hesheit": "he/she", "icecream": "ice cream", "minemy": "my",
    "thankyou": "thank you", "weus": "we", "owie": "hurt", "shhh": "shh",
}


def load_clips(*paths):
    """Clips from .npz files, or from folders of per-shard .npz files."""
    files = []
    for p in map(Path, paths):
        files += sorted(p.glob("*.npz")) if p.is_dir() else [p]
    clips, labels = [], []
    for f in files:
        d = dict(np.load(f))
        pose, hands, has, off, widths = d["pose"], d["hands"], d["has"], d["offsets"], d["widths"]
        for i in range(len(off) - 1):
            sl = slice(off[i], off[i + 1])
            clips.append(trim_to_hands(pose[sl], hands[sl], has[sl]) + (int(widths[i]),))
        labels += d["labels"].tolist()
    return clips, labels


def augment(pose, hands, has, width, rng):
    n = len(has)
    if n > 3 and rng.random() < 0.5:
        a = rng.integers(0, 2)
        b = n - rng.integers(0, 2)
        pose, hands, has = pose[a:b], hands[a:b], has[a:b]
    theta = rng.normal(0, 0.12)
    scale = rng.uniform(0.85, 1.15)
    shift = rng.normal(0, 0.04 * width, size=2)
    c, s = np.cos(theta), np.sin(theta)
    rot = np.array([[c, -s], [s, c]], dtype=np.float32) * scale
    center = np.array([width / 2, width / 2], dtype=np.float32)
    pose = (pose - center) @ rot.T + center + shift
    hxy = ((hands[..., :2] - center) @ rot.T + center + shift) * has[..., None, None]
    hands = np.concatenate([hxy, hands[..., 2:] * scale], axis=-1)
    hands = hands + rng.normal(0, 0.004 * width, size=hands.shape).astype(np.float32) * has[..., None, None]
    if rng.random() < 0.3:
        drop = rng.random(len(has)) < 0.15
        has = has * ~drop[:, None]
        hands = hands * has[..., None, None]
    return pose.astype(np.float32), hands.astype(np.float32), has.astype(np.float32)


def featurize(clips, rng=None):
    out = np.empty((len(clips), SEQ_LEN, FRAME_FEATURES), dtype=np.float32)
    for i, (pose, hands, has, width) in enumerate(clips):
        if rng is not None:
            pose, hands, has = augment(pose, hands, has, width, rng)
        out[i] = clip_features(pose, hands, has, width)
    return out


class ConvBlock(nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.conv = nn.Conv1d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.ff = nn.Sequential(nn.Linear(dim, dim * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(dim * 2, dim))

    def forward(self, x):
        h = self.norm(x)
        h = self.conv(h.transpose(1, 2)).transpose(1, 2)
        return x + self.ff(h)


class SignNet(nn.Module):
    def __init__(self, num_classes, mean, std, dim=256, depth=4, dropout=0.2):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean))
        self.register_buffer("std", torch.tensor(std))
        self.embed = nn.Sequential(nn.Linear(FRAME_FEATURES, dim), nn.LayerNorm(dim), nn.GELU(), nn.Dropout(dropout))
        self.pos = nn.Parameter(torch.zeros(1, SEQ_LEN, dim))
        self.blocks = nn.Sequential(*[ConvBlock(dim, dropout) for _ in range(depth)])
        self.head = nn.Sequential(nn.LayerNorm(dim * 2), nn.Dropout(0.3), nn.Linear(dim * 2, num_classes))

    def forward(self, x):
        h = self.embed((x - self.mean) / self.std) + self.pos
        h = self.blocks(h)
        return self.head(torch.cat([h.mean(dim=1), h.amax(dim=1)], dim=1))


def topk_accuracy(logits, y, k):
    return (logits.topk(k, dim=1).indices == y[:, None]).any(dim=1).float().mean().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", nargs="+", default=[str(ROOT / "data/signs/non-game-train"), str(ROOT / "data/signs/game-train")])
    parser.add_argument("--val", nargs="+", default=[str(ROOT / "data/signs/non-game-validation")])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--signs", help="Comma-separated subset of signs to train on (default: all)")
    parser.add_argument("--name", default="sign_model", help="Output name: models/<name>.onnx + <name>.json")
    args = parser.parse_args()
    device = torch.device(args.device)

    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    train_clips, train_labels = load_clips(*[p for p in args.train if Path(p).exists()])
    val_clips, val_labels = load_clips(*args.val)
    if args.signs:
        wanted = set(args.signs.split(","))
        pick = lambda clips, labels: ([c for c, l in zip(clips, labels) if l in wanted], [l for l in labels if l in wanted])  # noqa: E731
        train_clips, train_labels = pick(train_clips, train_labels)
        val_clips, val_labels = pick(val_clips, val_labels)
    labels = sorted(set(train_labels))
    index = {l: i for i, l in enumerate(labels)}
    y_train = torch.tensor([index[l] for l in train_labels])
    keep = [i for i, l in enumerate(val_labels) if l in index]
    val_clips = [val_clips[i] for i in keep]
    y_val = torch.tensor([index[val_labels[i]] for i in keep])
    X_val = torch.from_numpy(featurize(val_clips)).to(device)
    y_val = y_val.to(device)
    print(f"{len(train_clips)} train clips, {len(val_clips)} validation clips, {len(labels)} signs")

    base = featurize(train_clips)
    mean = base.reshape(-1, FRAME_FEATURES).mean(axis=0)
    std = base.reshape(-1, FRAME_FEATURES).std(axis=0) + 1e-3
    model = SignNet(len(labels), mean, std).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=0.05)
    steps = args.epochs * ((len(train_clips) + args.batch - 1) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=steps, pct_start=0.1)
    # some signs have 10x the clips of others; inverse-sqrt weights keep rare ones learnable
    counts = torch.bincount(y_train, minlength=len(labels)).float().clamp(min=1)
    weights = counts.rsqrt() / counts.rsqrt().mean()
    loss_fn = nn.CrossEntropyLoss(weight=weights.to(device), label_smoothing=0.1)

    best, best_state = 0.0, None
    for epoch in range(args.epochs):
        t0 = time.time()
        model.train()
        X = torch.from_numpy(featurize(train_clips, rng)).to(device)
        y = y_train.to(device)
        perm = torch.randperm(len(X), device=device)
        total = 0.0
        for i in range(0, len(X), args.batch):
            idx = perm[i : i + args.batch]
            loss = loss_fn(model(X[idx]), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            logits = model(X_val)
        top1, top5 = topk_accuracy(logits, y_val, 1), topk_accuracy(logits, y_val, 5)
        if top1 > best:
            best, best_state = top1, {k: v.clone() for k, v in model.state_dict().items()}
            best_top5 = top5
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch:3d}  loss {total / len(X):.3f}  val top1 {top1:.1%}  top5 {top5:.1%}  ({time.time() - t0:.1f}s)")

    model.load_state_dict(best_state)
    model = model.cpu().eval()
    print(f"\nBest validation (unseen signers): top-1 {best:.1%}, top-5 {best_top5:.1%}")
    out = ROOT / "models" / f"{args.name}.onnx"
    torch.onnx.export(
        model, torch.zeros(1, SEQ_LEN, FRAME_FEATURES), out, input_names=["features"], output_names=["logits"],
        dynamic_axes={"features": {0: "batch"}, "logits": {0: "batch"}}, opset_version=17, dynamo=False,
    )
    meta = {
        "labels": labels,
        "display": [DISPLAY.get(l, l) for l in labels],
        "seq_len": SEQ_LEN,
        "features": FRAME_FEATURES,
        "metrics": {"val_top1": round(best, 4), "val_top5": round(best_top5, 4), "val_clips": len(val_clips),
                    "train_clips": len(train_clips)},
    }
    (ROOT / "models" / f"{args.name}.json").write_text(json.dumps(meta, indent=1))
    for f in [f"{args.name}.onnx", f"{args.name}.json"]:
        shutil.copy(ROOT / "models" / f, ROOT / "web" / "model" / f)
    print(f"Saved {out} ({out.stat().st_size / 1e6:.1f} MB)")

    with torch.no_grad():
        proba = model(X_val.cpu()).softmax(dim=1)
    conf, pred = proba.max(dim=1)
    correct = pred == y_val.cpu()
    print("confidence >=   share of signs   accuracy when accepted")
    for t in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        m = conf >= t
        print(f"   {t:.1f}          {m.float().mean():6.1%}          {correct[m].float().mean():6.1%}")


if __name__ == "__main__":
    main()
