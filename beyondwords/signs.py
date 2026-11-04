"""Word-level signs: per-frame body + hands capture and the clip features the sign model sees.

A clip is a list of frame records. Each record holds, in pixel units:
  pose  (15, 2)     nose, mouth, shoulders, elbows, wrists, pinky/index/thumb (NaN when no body)
  hands (2, 21, 3)  slot 0 near the body's landmark-15 wrist, slot 1 near landmark 16
  has   (2,)        which hand slots are filled

`clip_features` must stay in sync with clipFeatures in web/signs.js.
"""

import numpy as np

from . import MODELS_DIR
from .landmarks import NUM_LANDMARKS

POSE_MODEL_PATH = MODELS_DIR / "pose_landmarker_lite.task"
POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
POSE_IDX = [0, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22]
POSE_MIRROR = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13]
SHOULDERS = slice(3, 5)
ELBOWS = slice(5, 7)
WRISTS = slice(7, 9)
SEQ_LEN = 12
FRAME_FEATURES = 2 * len(POSE_IDX) + 2 * (1 + 42 + 63)


def create_pose_landmarker(video=False, min_confidence=0.5):
    import urllib.request

    from mediapipe.tasks.python import BaseOptions, vision

    if not POSE_MODEL_PATH.exists():
        print(f"Downloading MediaPipe pose model to {POSE_MODEL_PATH} ...")
        urllib.request.urlretrieve(POSE_MODEL_URL, POSE_MODEL_PATH)
    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(POSE_MODEL_PATH)),
        running_mode=vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE,
        num_poses=1,
        min_pose_detection_confidence=min_confidence,
        min_pose_presence_confidence=min_confidence,
        min_tracking_confidence=min_confidence,
    )
    return vision.PoseLandmarker.create_from_options(options)


def frame_record(hand_result, pose_result, width, height):
    pose = np.full((len(POSE_IDX), 2), np.nan, dtype=np.float32)
    if pose_result is not None and pose_result.pose_landmarks:
        lms = pose_result.pose_landmarks[0]
        pose = np.array([[lms[i].x * width, lms[i].y * height] for i in POSE_IDX], dtype=np.float32)
    found = [
        np.array([[p.x * width, p.y * height, p.z * width] for p in hand], dtype=np.float32)
        for hand in (hand_result.hand_landmarks if hand_result is not None else [])
    ][:2]
    hands, has = assign_hands(found, pose, width)
    return {"pose": pose, "hands": hands, "has": has}


def assign_hands(found, pose, width):
    """Put each detected hand in the slot of the body wrist it is closest to."""
    hands = np.zeros((2, NUM_LANDMARKS, 3), dtype=np.float32)
    has = np.zeros(2, dtype=np.float32)
    if not found:
        return hands, has
    if not np.isnan(pose).any():
        wrists = pose[WRISTS]
        cost = np.array([[np.linalg.norm(h[0, :2] - w) for w in wrists] for h in found])
        if len(found) == 2:
            slots = [0, 1] if cost[0, 0] + cost[1, 1] <= cost[0, 1] + cost[1, 0] else [1, 0]
        else:
            slots = [int(cost[0].argmin())]
    else:
        # No body: landmark 15 (the signer's left wrist) sits on the image's right side
        slots = [0 if h[0, 0] > width / 2 else 1 for h in found]
        if len(found) == 2 and slots[0] == slots[1]:
            slots = [0, 1] if found[0][0, 0] > found[1][0, 0] else [1, 0]
    for h, s in zip(found, slots):
        hands[s], has[s] = h, 1
    return hands, has


def stack(records):
    return (
        np.stack([r["pose"] for r in records]),
        np.stack([r["hands"] for r in records]),
        np.stack([r["has"] for r in records]),
    )


def trim_to_hands(pose, hands, has):
    """Drop leading/trailing frames without hands; live capture only records while hands are up."""
    seen = np.flatnonzero(has.sum(axis=1) > 0)
    if len(seen) == 0:
        return pose, hands, has
    sl = slice(seen[0], seen[-1] + 1)
    return pose[sl], hands[sl], has[sl]


def clip_features(pose, hands, has, width):
    """(N frames) raw arrays -> (SEQ_LEN, FRAME_FEATURES) model input.

    The clip is mirrored when slot 0 is the busier hand, so the dominant hand is
    always slot 1 and left- and right-handed signers look the same to the model.
    """
    pose, hands, has = pose.copy(), hands.copy(), has.copy()
    if has[:, 0].sum() > has[:, 1].sum():
        pose[..., 0] = width - pose[..., 0]
        pose = pose[:, POSE_MIRROR]
        hands[..., 0] = (width - hands[..., 0]) * has[:, :, None]
        hands, has = hands[:, ::-1], has[:, ::-1]

    valid = ~np.isnan(pose).any(axis=(1, 2))
    if valid.any():
        shoulders = pose[valid][:, SHOULDERS]
        center = shoulders.mean(axis=1).mean(axis=0)
        scale = np.linalg.norm(shoulders[:, 0] - shoulders[:, 1], axis=-1).mean()
    else:
        center = np.array([width / 2, width * 0.75], dtype=np.float32)
        scale = width * 0.35
    scale = max(float(scale), 1e-3)

    body = np.nan_to_num((pose - center) / scale).reshape(len(pose), -1)
    parts = [body]
    for s in range(2):
        h = hands[:, s]
        present = has[:, s : s + 1]
        global_xy = ((h[..., :2] - center) / scale).reshape(len(h), -1) * present
        local = h - h[:, :1]
        norm = np.linalg.norm(local[..., :2], axis=-1).max(axis=-1)[:, None, None]
        local = (local / np.maximum(norm, 1e-6)).reshape(len(h), -1) * present
        parts += [present, global_xy, local]
    frames = np.concatenate(parts, axis=1).astype(np.float32)
    return resample(frames, SEQ_LEN)


def resample(frames, length):
    """Linear interpolation along time to a fixed number of frames."""
    n = len(frames)
    if n == 1:
        return np.repeat(frames, length, axis=0)
    pos = np.linspace(0, n - 1, length)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, n - 1)
    t = (pos - lo)[:, None].astype(np.float32)
    return frames[lo] * (1 - t) + frames[hi] * t


class SignModel:
    """name: "sign_core" (51 everyday signs, more accurate) or "sign_full" (all 250)."""

    def __init__(self, name="sign_core"):
        import json

        import onnxruntime as ort

        model_path = MODELS_DIR / f"{name}.onnx"
        self.session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        meta = json.loads((MODELS_DIR / f"{name}.json").read_text())
        if (meta["seq_len"], meta["features"]) != (SEQ_LEN, FRAME_FEATURES):
            raise ValueError(
                f"{model_path.name} expects {meta['seq_len']}x{meta['features']} inputs but this code computes "
                f"{SEQ_LEN}x{FRAME_FEATURES}; retrain with scripts/train_signs.py"
            )
        self.labels = meta["display"]

    def predict(self, records, width, top=5):
        """[(word, probability), ...] best first, for a list of frame records."""
        pose, hands, has = stack(records)
        x = clip_features(pose, hands, has, width)[None]
        logits = self.session.run(None, {"features": x})[0][0]
        p = np.exp(logits - logits.max())
        p /= p.sum()
        order = p.argsort()[::-1][:top]
        return [(self.labels[i], float(p[i])) for i in order]


def is_signing(record):
    """A hand is up: visible and, when the body is tracked, its wrist above its elbow.

    That's the rule PopSign used to cut its clips, so live segments match training.
    """
    has, pose = record["has"], record["pose"]
    if not has.any():
        return False
    if np.isnan(pose).any():
        return True
    wrists, elbows = pose[WRISTS, 1], pose[ELBOWS, 1]
    return bool(((wrists < elbows) & (has > 0)).any())


class SignSegmenter:
    """Collects frames while a hand is raised; a sign ends when the hands come down.

    Mirrored by SignSegmenter in web/signs.js.
    """

    def __init__(self, gap_seconds=0.35, min_frames=4, max_seconds=3.5):
        self.gap_seconds = gap_seconds
        self.min_frames = min_frames
        self.max_seconds = max_seconds
        self.frames = []
        self.started = self.last_hand = 0.0

    @property
    def active(self):
        return bool(self.frames)

    def push(self, record, now):
        """Returns the finished clip's records when a sign ends, else None."""
        if is_signing(record):
            if not self.frames:
                self.started = now
            self.frames.append(record)
            self.last_hand = now
            if now - self.started > self.max_seconds:
                return self.flush()
            return None
        if self.frames and now - self.last_hand > self.gap_seconds:
            return self.flush()
        return None

    def flush(self):
        frames, self.frames = self.frames, []
        return frames if len(frames) >= self.min_frames else None
