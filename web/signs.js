// Word signs: body + both hands over time -> one of 250 ASL signs.
// Mirrors beyondwords/signs.py; the browser and Python must produce identical features.

export const SEQ_LEN = 12;
const POSE_IDX = [0, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22];
const POSE_MIRROR = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13];
const [SHOULDER_L, SHOULDER_R, ELBOW_L, ELBOW_R, WRIST_L, WRIST_R] = [3, 4, 5, 6, 7, 8];
export const FRAME_FEATURES = 2 * POSE_IDX.length + 2 * (1 + 42 + 63);

export function frameRecord(handResult, poseResult, width, height) {
  const lms = poseResult?.landmarks?.[0];
  const pose = lms ? POSE_IDX.map((i) => [lms[i].x * width, lms[i].y * height]) : null;
  const found = (handResult?.landmarks ?? []).slice(0, 2).map((hand) => hand.map((p) => [p.x * width, p.y * height, p.z * width]));
  return { pose, hands: assignHands(found, pose, width) };
}

function assignHands(found, pose, width) {
  const hands = [null, null];
  if (!found.length) return hands;
  let slots;
  if (pose) {
    const d = (hand, w) => Math.hypot(hand[0][0] - pose[w][0], hand[0][1] - pose[w][1]);
    if (found.length === 2) slots = d(found[0], WRIST_L) + d(found[1], WRIST_R) <= d(found[0], WRIST_R) + d(found[1], WRIST_L) ? [0, 1] : [1, 0];
    else slots = [d(found[0], WRIST_L) <= d(found[0], WRIST_R) ? 0 : 1];
  } else {
    slots = found.map((hand) => (hand[0][0] > width / 2 ? 0 : 1));
    if (found.length === 2 && slots[0] === slots[1]) slots = found[0][0][0] > found[1][0][0] ? [0, 1] : [1, 0];
  }
  found.forEach((hand, i) => { hands[slots[i]] = hand; });
  return hands;
}

export function clipFeatures(records, width) {
  let pose = records.map((r) => r.pose);
  let hands = records.map((r) => r.hands);
  const count = (s) => hands.filter((h) => h[s]).length;
  if (count(0) > count(1)) {
    pose = pose.map((p) => p && POSE_MIRROR.map((i) => [width - p[i][0], p[i][1]]));
    hands = hands.map((h) => [h[1], h[0]].map((hand) => hand && hand.map(([x, y, z]) => [width - x, y, z])));
  }

  const valid = pose.filter(Boolean);
  let cx = width / 2, cy = width * 0.75, scale = width * 0.35;
  if (valid.length) {
    cx = cy = scale = 0;
    for (const p of valid) {
      const [l, r] = [p[SHOULDER_L], p[SHOULDER_R]];
      cx += (l[0] + r[0]) / 2;
      cy += (l[1] + r[1]) / 2;
      scale += Math.hypot(l[0] - r[0], l[1] - r[1]);
    }
    cx /= valid.length; cy /= valid.length; scale /= valid.length;
  }
  scale = Math.max(scale, 1e-3);

  const frames = records.map((_, f) => {
    const out = [];
    const p = pose[f];
    for (let i = 0; i < POSE_IDX.length; i++) out.push(p ? (p[i][0] - cx) / scale : 0, p ? (p[i][1] - cy) / scale : 0);
    for (const hand of hands[f]) {
      if (!hand) {
        for (let i = 0; i < 106; i++) out.push(0);
        continue;
      }
      out.push(1);
      for (const [x, y] of hand) out.push((x - cx) / scale, (y - cy) / scale);
      const [wx, wy, wz] = hand[0];
      const local = hand.map(([x, y, z]) => [x - wx, y - wy, z - wz]);
      let norm = 1e-6;
      for (const [x, y] of local) norm = Math.max(norm, Math.hypot(x, y));
      for (const [x, y, z] of local) out.push(x / norm, y / norm, z / norm);
    }
    return out;
  });
  return resample(frames, SEQ_LEN);
}

function resample(frames, length) {
  const n = frames.length;
  if (n === 1) return Array.from({ length }, () => frames[0]);
  return Array.from({ length }, (_, k) => {
    const pos = (k * (n - 1)) / (length - 1);
    const lo = Math.floor(pos);
    const hi = Math.min(lo + 1, n - 1);
    const t = pos - lo;
    return frames[lo].map((v, i) => v * (1 - t) + frames[hi][i] * t);
  });
}

export class SignModel {
  constructor(session, meta) {
    this.session = session;
    this.words = meta.display;
    this.ids = meta.labels;
    this.metrics = meta.metrics;
  }

  static async load(base, name) {
    const [session, meta] = await Promise.all([
      window.ort.InferenceSession.create(`${base}/${name}.onnx`, { executionProviders: ["wasm"] }),
      fetch(`${base}/${name}.json`).then((r) => {
        if (!r.ok) throw new Error(`sign labels: HTTP ${r.status}`);
        return r.json();
      }),
    ]);
    if (meta.features !== FRAME_FEATURES || meta.seq_len !== SEQ_LEN) {
      throw new Error(`the model expects ${meta.seq_len}×${meta.features} inputs but this app computes ${SEQ_LEN}×${FRAME_FEATURES}`);
    }
    return new SignModel(session, meta);
  }

  async predict(records, width, top = 5) {
    const x = clipFeatures(records, width);
    const tensor = new window.ort.Tensor("float32", Float32Array.from(x.flat()), [1, SEQ_LEN, FRAME_FEATURES]);
    const logits = (await this.session.run({ features: tensor })).logits.data;
    let max = -Infinity;
    for (const v of logits) max = Math.max(max, v);
    const exp = Array.from(logits, (v) => Math.exp(v - max));
    const sum = exp.reduce((a, b) => a + b, 0);
    return exp
      .map((v, i) => ({ word: this.words[i], id: this.ids[i], p: v / sum }))
      .sort((a, b) => b.p - a.p)
      .slice(0, top);
  }
}

// Mirrors beyondwords/signs.py:is_signing
export function isSigning({ pose, hands }) {
  if (!hands[0] && !hands[1]) return false;
  if (!pose) return true;
  return (hands[0] && pose[WRIST_L][1] < pose[ELBOW_L][1]) || (hands[1] && pose[WRIST_R][1] < pose[ELBOW_R][1]) || false;
}

// Mirrors beyondwords/signs.py:SignSegmenter
export class SignSegmenter {
  constructor({ gapSeconds = 0.35, minFrames = 4, maxSeconds = 3.5 } = {}) {
    Object.assign(this, { gapSeconds, minFrames, maxSeconds });
    this.frames = [];
    this.started = this.lastHand = 0;
  }

  get active() {
    return this.frames.length > 0;
  }

  push(record, now) {
    if (isSigning(record)) {
      if (!this.frames.length) this.started = now;
      this.frames.push(record);
      this.lastHand = now;
      return now - this.started > this.maxSeconds ? this.flush() : null;
    }
    if (this.frames.length && now - this.lastHand > this.gapSeconds) return this.flush();
    return null;
  }

  flush() {
    const frames = this.frames;
    this.frames = [];
    return frames.length >= this.minFrames ? frames : null;
  }
}
