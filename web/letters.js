// Fingerspelling: one hand's landmarks -> A-Z / space / del, and hold-to-type.

export const SPACE = "space";
export const DELETE = "del";

// Same features as beyondwords/landmarks.py:landmarks_to_features
export function toFeatures(landmarks, width, height) {
  const pts = landmarks.map((p) => [p.x * width, p.y * height, p.z * width]);
  const [wx, wy, wz] = pts[0];
  let scale = 1e-6;
  for (const p of pts) {
    p[0] -= wx; p[1] -= wy; p[2] -= wz;
    scale = Math.max(scale, Math.hypot(p[0], p[1]));
  }
  return pts.flat().map((v) => v / scale);
}

export class LetterClassifier {
  constructor({ labels, mean, std, layers }) {
    this.labels = labels;
    this.mean = mean;
    this.std = std;
    this.layers = layers;
  }

  static async load(url) {
    const r = await fetch(url);
    if (!r.ok) throw new Error(`letter model: HTTP ${r.status}`);
    return new LetterClassifier(await r.json());
  }

  predict(features) {
    let h = features.map((v, i) => (v - this.mean[i]) / this.std[i]);
    this.layers.forEach(({ W, b }, li) => {
      const out = b.slice();
      for (let i = 0; i < h.length; i++) {
        const hi = h[i];
        if (hi === 0) continue;
        const row = W[i];
        for (let j = 0; j < out.length; j++) out[j] += hi * row[j];
      }
      h = li < this.layers.length - 1 ? out.map((v) => Math.max(v, 0)) : out;
    });
    const max = Math.max(...h);
    const exp = h.map((v) => Math.exp(v - max));
    const sum = exp.reduce((a, b) => a + b, 0);
    return exp.map((v, i) => ({ label: this.labels[i], p: v / sum })).sort((a, b) => b.p - a.p);
  }
}

// Mirrors beyondwords/speller.py
export class Speller {
  constructor({ holdSeconds = 0.7, repeatSeconds = 1.4, minConfidence = 0.6, wordPause = 1.0 } = {}) {
    Object.assign(this, { holdSeconds, repeatSeconds, minConfidence, wordPause });
    this.text = "";
    this.candidate = null;
    this.since = 0;
    this.lastCommitted = null;
    this.repeated = false;
    this.goneSince = null;
    this.progress = 0;
  }

  update(label, confidence, now) {
    if (label === null) {
      this.candidate = null;
      this.lastCommitted = null;
      this.progress = 0;
      if (this.goneSince === null) this.goneSince = now;
      else if (now - this.goneSince >= this.wordPause && this.text && !this.text.endsWith(" ")) return this.commit(SPACE);
      return null;
    }
    this.goneSince = null;
    if (confidence < this.minConfidence) {
      this.progress = 0;
      return null;
    }
    if (label !== this.candidate) {
      this.candidate = label;
      this.since = now;
    }
    const isRepeat = label === this.lastCommitted;
    if (isRepeat && this.repeated) {
      this.progress = 0;
      return null;
    }
    const needed = isRepeat ? this.repeatSeconds : this.holdSeconds;
    this.progress = Math.min(1, (now - this.since) / needed);
    if (this.progress < 1) return null;

    this.since = now;
    this.lastCommitted = label;
    this.repeated = isRepeat;
    return this.commit(label);
  }

  commit(label) {
    if (label === DELETE) {
      this.text = this.text.slice(0, -1);
      return { type: "delete" };
    }
    if (label === SPACE) {
      const word = this.currentWord();
      if (this.text && !this.text.endsWith(" ")) this.text += " ";
      return { type: "space", word };
    }
    this.text += label;
    return { type: "letter", value: label };
  }

  currentWord() {
    const words = this.text.trim().split(" ");
    return words[words.length - 1] ?? "";
  }
}
