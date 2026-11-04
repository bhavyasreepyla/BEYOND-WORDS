import numpy as np
import pytest

from beyondwords import CLASSIFIER_PATH
from beyondwords.labels import DELETE, SPACE
from beyondwords.landmarks import landmarks_to_features
from beyondwords.speller import Speller


def feed(speller, label, seconds, start, fps=30, confidence=0.95):
    events, t = [], start
    for _ in range(int(seconds * fps)):
        e = speller.update(label, confidence, t)
        if e:
            events.append(e)
        t += 1 / fps
    return events, t


def test_holding_a_sign_types_it_once():
    s = Speller()
    events, _ = feed(s, "H", 1.0, 0)
    assert events == [("letter", "H")]
    assert s.text == "H"


def test_brief_flicker_is_ignored():
    s = Speller()
    feed(s, "A", 0.3, 0)
    assert s.text == ""


def test_double_letters_need_a_longer_hold_or_hand_drop():
    s = Speller()
    _, t = feed(s, "L", 0.8, 0)
    _, t = feed(s, "L", 1.0, t)
    assert s.text == "L"
    _, t = feed(s, "L", 0.4, t)
    assert s.text == "LL"
    _, t = feed(s, "L", 5.0, t)
    assert s.text == "LL"
    _, t = feed(s, None, 0.2, t)
    feed(s, "L", 0.8, t)
    assert s.text == "LLL"


def test_space_returns_finished_word_and_delete_erases():
    s = Speller()
    t = 0
    for letter in "HI":
        _, t = feed(s, letter, 0.8, t)
    events, t = feed(s, SPACE, 0.8, t)
    assert events == [("space", "HI")]
    assert s.text == "HI "
    feed(s, DELETE, 0.8, t)
    assert s.text == "HI"


def test_low_confidence_does_not_type():
    s = Speller()
    feed(s, "B", 2.0, 0, confidence=0.3)
    assert s.text == ""


def test_features_ignore_position_scale_and_aspect_ratio():
    rng = np.random.default_rng(1)
    pixels = rng.uniform(100, 300, size=(21, 3)).astype(np.float32)
    a = landmarks_to_features(pixels / [640, 480, 640], 640, 480)
    moved = (pixels * 1.7 + [50, -20, 0]) / [1280, 720, 1280]
    b = landmarks_to_features(moved, 1280, 720)
    assert a.shape == (63,)
    np.testing.assert_allclose(a, b, atol=1e-5)


@pytest.mark.skipif(not CLASSIFIER_PATH.exists(), reason="run scripts/train.py first")
def test_exported_classifier_loads_and_outputs_probabilities():
    from beyondwords.classifier import Classifier

    clf = Classifier.load()
    proba = clf.predict_proba(np.zeros((4, 63), dtype=np.float32))
    assert proba.shape == (4, len(clf.labels))
    np.testing.assert_allclose(proba.sum(axis=1), 1, atol=1e-5)


def test_lowering_the_hand_ends_and_returns_the_word():
    s = Speller()
    t = 0
    for letter in "HI":
        _, t = feed(s, letter, 0.8, t)
    events, t = feed(s, None, 1.5, t)
    assert events == [("space", "HI")]
    assert s.text == "HI "
    events, _ = feed(s, None, 2.0, t)
    assert events == []


def test_segmenter_returns_clip_after_hands_drop():
    from beyondwords.signs import SignSegmenter

    seg = SignSegmenter()
    pose = np.zeros((15, 2), dtype=np.float32)
    pose[6, 1], pose[8, 1] = 300, 100  # right elbow low, right wrist raised
    up = {"has": np.array([0, 1]), "pose": pose}
    down = {"has": np.array([0, 0]), "pose": pose}
    out = [seg.push(up, i / 30) for i in range(20)]
    assert not any(o is not None for o in out)
    results = [seg.push(down, (20 + i) / 30) for i in range(15)]
    clips = [r for r in results if r is not None]
    assert len(clips) == 1 and len(clips[0]) == 20


def test_clip_features_are_mirror_invariant():
    from beyondwords.signs import FRAME_FEATURES, SEQ_LEN, clip_features

    rng = np.random.default_rng(0)
    n, w = 9, 256
    pose = rng.uniform(50, 200, size=(n, 15, 2)).astype(np.float32)
    hands = rng.uniform(50, 200, size=(n, 2, 21, 3)).astype(np.float32)
    hands[:, 0] = 0
    has = np.tile(np.array([0, 1], dtype=np.float32), (n, 1))
    a = clip_features(pose, hands, has, w)
    # the same clip filmed mirrored: x flipped, body sides and hand slots swapped
    mp, mh = pose.copy(), hands[:, ::-1].copy()
    mp[..., 0] = w - mp[..., 0]
    mp = mp[:, [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13]]
    mh[:, 0, :, 0] = w - mh[:, 0, :, 0]
    b = clip_features(mp, mh, has[:, ::-1].copy(), w)
    assert a.shape == (SEQ_LEN, FRAME_FEATURES)
    np.testing.assert_allclose(a, b, atol=1e-4)


def test_hand_resting_below_the_elbow_is_not_signing():
    from beyondwords.signs import is_signing

    pose = np.zeros((15, 2), dtype=np.float32)
    pose[5:7, 1] = 300  # elbows
    pose[7:9, 1] = 400  # wrists hanging below them
    assert not is_signing({"has": np.array([1, 1]), "pose": pose})
    pose[8, 1] = 150
    assert is_signing({"has": np.array([0, 1]), "pose": pose})
    assert is_signing({"has": np.array([1, 0]), "pose": np.full((15, 2), np.nan)})
