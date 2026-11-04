"""Hand landmark detection and the feature vector the classifier sees.

`landmarks_to_features` must stay in sync with `toFeatures` in web/app.js,
since the browser demo runs the same exported classifier.
"""

import urllib.request

import numpy as np

from . import HAND_MODEL_PATH, HAND_MODEL_URL

NUM_LANDMARKS = 21
NUM_FEATURES = NUM_LANDMARKS * 3


def ensure_hand_model(path=HAND_MODEL_PATH):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading MediaPipe hand model to {path} ...")
        urllib.request.urlretrieve(HAND_MODEL_URL, path)
    return path


def create_landmarker(video=False, min_confidence=0.5, num_hands=1):
    import mediapipe as mp
    from mediapipe.tasks.python import BaseOptions, vision

    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(ensure_hand_model())),
        running_mode=vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE,
        num_hands=num_hands,
        min_hand_detection_confidence=min_confidence,
        min_hand_presence_confidence=min_confidence,
        min_tracking_confidence=min_confidence,
    )
    return vision.HandLandmarker.create_from_options(options)


def to_mp_image(rgb):
    import mediapipe as mp

    return mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))


def result_to_array(result):
    """First detected hand as a (21, 3) array of normalized coordinates, or None."""
    if not result.hand_landmarks:
        return None
    return np.array([[p.x, p.y, p.z] for p in result.hand_landmarks[0]], dtype=np.float32)


def landmarks_to_features(landmarks, width, height):
    """Translation-, scale- and aspect-ratio-invariant features from (..., 21, 3) landmarks.

    MediaPipe normalizes x by image width and y by image height, so landmarks are
    first brought back to a common pixel scale (z uses the same scale as x).
    """
    pts = np.asarray(landmarks, dtype=np.float32) * np.array([width, height, width], dtype=np.float32)
    pts = pts - pts[..., :1, :]
    scale = np.linalg.norm(pts[..., :2], axis=-1).max(axis=-1)[..., None, None]
    pts = pts / np.maximum(scale, 1e-6)
    return pts.reshape(*pts.shape[:-2], NUM_FEATURES)
