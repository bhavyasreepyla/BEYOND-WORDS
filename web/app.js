import { FilesetResolver, HandLandmarker, PoseLandmarker } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.1.1/vision_bundle.mjs";
import { DELETE, LetterClassifier, SPACE, Speller, toFeatures } from "./letters.js";
import { SignModel, SignSegmenter, frameRecord } from "./signs.js";

const WASM_URL = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@1.1.1/wasm";
const HAND_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task";
const POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task";
window.ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";

const PHRASE_IDLE_SECONDS = 3.5;
const TRAIL_SECONDS = 0.45;
const FINGERTIPS = [4, 8, 12, 16, 20];
const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [0, 17], [17, 18], [18, 19], [19, 20],
];

const $ = (id) => document.getElementById(id);
const escape = (s) => s.replace(/&/g, "&amp;").replace(/</g, "&lt;");
const store = {
  get(k) { try { return localStorage.getItem(k); } catch { return null; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch { /* storage unavailable */ } },
};

const state = {
  mode: store.get("bw-mode") === "spell" ? "spell" : "words",
  muted: store.get("bw-muted") === "1",
  phrase: [],
  lastActivity: 0,
  lastResult: null,
};

const speller = new Speller();
const segmenter = new SignSegmenter({ minFrames: 8 });
let letters, signModel, handLandmarker, poseLandmarker, vision;
let signModelError = null;

// ---------- speech out ----------

const voiceSelect = $("voice");
let voices = [];

function loadVoices() {
  const all = speechSynthesis.getVoices();
  voices = all.filter((v) => v.lang.startsWith("en"));
  if (!voices.length) voices = all;
  const saved = store.get("bw-voice");
  voiceSelect.innerHTML = voices.map((v, i) => `<option value="${i}">${escape(v.name.replace(/^(Microsoft|Google) /, "").replace(/ Online \(Natural\)/, ""))}</option>`).join("");
  const idx = voices.findIndex((v) => v.name === saved);
  if (idx >= 0) voiceSelect.value = String(idx);
}

if ("speechSynthesis" in window) {
  loadVoices();
  speechSynthesis.addEventListener("voiceschanged", loadVoices);
  voiceSelect.onchange = () => store.set("bw-voice", voices[Number(voiceSelect.value)]?.name ?? "");
} else {
  voiceSelect.closest("label").hidden = true;
}

function speak(text) {
  if (state.muted || !text?.trim() || !("speechSynthesis" in window)) return;
  const u = new SpeechSynthesisUtterance(text.trim().toLowerCase());
  const voice = voices[Number(voiceSelect.value)];
  if (voice) u.voice = voice;
  u.rate = 1.02;
  speechSynthesis.speak(u);
}

const muteBtn = $("mute");
function renderMute() {
  muteBtn.setAttribute("aria-pressed", String(state.muted));
  muteBtn.title = muteBtn.ariaLabel = state.muted ? "Unmute speech" : "Mute speech";
}
muteBtn.onclick = () => {
  state.muted = !state.muted;
  store.set("bw-muted", state.muted ? "1" : "0");
  if (state.muted) speechSynthesis.cancel();
  renderMute();
};
renderMute();

// ---------- captions + conversation ----------

const captionEl = $("caption");
const threadEl = $("thread");
let ghost = "";

function renderCaptions(freshIndex = -1) {
  const words = state.phrase.map((t, i) => `<span class="w${i === freshIndex ? " new" : ""}">${escape(t.text)}</span>`);
  if (state.mode === "spell" && speller.text.trim()) words.push(`<span class="w spelling">${escape(speller.text.trim())}</span>`);
  if (ghost) words.push(`<span class="w ghost">${escape(ghost)}…</span>`);
  captionEl.innerHTML = words.join("");
}

function addWord(text, alts = []) {
  state.phrase.push({ text, alts });
  state.lastActivity = performance.now() / 1000;
  ghost = "";
  renderCaptions(state.phrase.length - 1);
  speak(text);
}

function pushMessage(who, text) {
  if (!text) return;
  $("empty")?.remove();
  const li = document.createElement("li");
  li.className = `msg ${who}`;
  li.innerHTML = `<small>${who === "me" ? "You signed" : "They said"}</small>${escape(text)}`;
  threadEl.insertBefore(li, pendingEl?.isConnected ? pendingEl : null);
  threadEl.scrollTop = threadEl.scrollHeight;
}

function maybeFinishPhrase(now, handsVisible) {
  if (!state.phrase.length || handsVisible || segmenter.active) return;
  if (state.mode === "spell" && speller.text.trim()) return;
  if (now - state.lastActivity < PHRASE_IDLE_SECONDS) return;
  pushMessage("me", state.phrase.map((t) => t.text).join(" "));
  state.phrase = [];
  state.lastResult = null;
  renderCaptions();
}

$("clear").onclick = () => {
  state.phrase = [];
  state.lastResult = null;
  speller.text = "";
  threadEl.innerHTML = `<li class="empty" id="empty">Cleared. Start signing whenever you're ready.</li>`;
  renderCaptions();
  renderWordsIdle();
};

// ---------- speech in (the other person) ----------

const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
const listenBtn = $("listen");
let recognizer = null;
let listening = false;
let pendingEl = null;

function setPending(text) {
  if (!text) {
    pendingEl?.remove();
    pendingEl = null;
    return;
  }
  if (!pendingEl) {
    $("empty")?.remove();
    pendingEl = document.createElement("li");
    pendingEl.className = "msg them pending";
    threadEl.append(pendingEl);
  }
  pendingEl.innerHTML = `<small>Listening</small>${escape(text)}`;
  threadEl.scrollTop = threadEl.scrollHeight;
}

function setListening(on) {
  listening = on;
  listenBtn.setAttribute("aria-pressed", String(on));
  listenBtn.querySelector("span").textContent = on ? "Listening" : "Listen";
  if (on) {
    recognizer = new Recognition();
    recognizer.continuous = true;
    recognizer.interimResults = true;
    recognizer.lang = "en-US";
    recognizer.onresult = (e) => {
      if (speechSynthesis.speaking) return;
      let interim = "";
      for (let i = e.resultIndex; i < e.results.length; i++) {
        const r = e.results[i];
        if (r.isFinal) {
          setPending("");
          pushMessage("them", r[0].transcript.trim());
        } else {
          interim += r[0].transcript;
        }
      }
      setPending(interim.trim());
    };
    recognizer.onerror = (e) => {
      if (e.error === "not-allowed" || e.error === "service-not-allowed") {
        setListening(false);
        pushMessage("them", "Microphone access is blocked, so replies can't be transcribed. Allow the mic in site settings to use Listen.");
      }
    };
    recognizer.onend = () => { if (listening) try { recognizer.start(); } catch { /* already restarting */ } };
    recognizer.start();
  } else if (recognizer) {
    recognizer.onend = null;
    recognizer.stop();
    setPending("");
  }
}

if (Recognition) {
  listenBtn.hidden = false;
  listenBtn.onclick = () => setListening(!listening);
}

// ---------- reading panel ----------

const nowWord = $("now-word");
const nowMeta = $("now-meta");
const pulse = $("pulse");
const altsEl = $("alts");
const liveEl = $("live");

function setNowWord(text, cls) {
  nowWord.className = `now-word ${cls}`;
  nowWord.textContent = text;
}

function renderWordsIdle() {
  pulse.classList.remove("reading");
  liveEl.classList.remove("reading");
  setNowWord("Ready", "idle");
  nowMeta.innerHTML = "Raise your hands, sign a word, then lower them.";
  altsEl.innerHTML = "";
}

function renderReading(guess) {
  pulse.classList.add("reading");
  liveEl.classList.add("reading");
  setNowWord(guess ?? "…", "");
  nowMeta.innerHTML = "Reading… lower your hands to finish the sign.";
  altsEl.innerHTML = "";
}

function renderAlts(options, label) {
  altsEl.innerHTML = `<span class="label">${label}</span>` +
    options.map((o, i) => `<button class="alt" data-i="${i}">${escape(o.word)}<small>${Math.round(o.p * 100)}%</small></button>`).join("");
  altsEl.querySelectorAll(".alt").forEach((b) => {
    b.onclick = () => chooseAlternative(options[Number(b.dataset.i)]);
  });
}

function onSignResult(top) {
  pulse.classList.remove("reading");
  liveEl.classList.remove("reading");
  ghost = "";
  const [best] = top;
  if (best.p >= VOCABS[state.vocab].threshold) {
    addWord(best.word, top.slice(1, 4));
    state.lastResult = { committed: true, index: state.phrase.length - 1 };
    setNowWord(best.word, "ink pop");
    nowMeta.innerHTML = `<b>${Math.round(best.p * 100)}%</b> sure · spoken aloud`;
    renderAlts(top.slice(1, 4), "Not it?");
  } else {
    state.lastResult = { committed: false };
    setNowWord("?", "idle pop");
    nowMeta.innerHTML = "Didn't quite catch that. Was it one of these?";
    renderAlts(top.slice(0, 4), "Best guesses");
    renderCaptions();
  }
}

function chooseAlternative(option) {
  const r = state.lastResult;
  if (r?.committed && state.phrase[r.index]) {
    state.phrase[r.index].text = option.word;
    renderCaptions(r.index);
    speak(option.word);
  } else {
    addWord(option.word);
    state.lastResult = { committed: true, index: state.phrase.length - 1 };
  }
  setNowWord(option.word, "ink pop");
  nowMeta.innerHTML = "Corrected · spoken aloud";
  altsEl.innerHTML = "";
}

const dial = $("dial");
const letterEl = $("letter");
const spellMeta = $("spell-meta");
const barsEl = $("bars");
const display = (l) => ({ [SPACE]: "space", [DELETE]: "del" })[l] ?? l;

function renderSpell(top) {
  if (!top) {
    letterEl.textContent = "—";
    letterEl.className = "dial-letter small";
    dial.style.setProperty("--p", 0);
    barsEl.innerHTML = "";
    spellMeta.innerHTML = speller.text.trim() ? "Word finishes and is spoken in a moment…" : "Hold a letter steady to type it.";
    return;
  }
  const label = top[0].label;
  letterEl.textContent = display(label);
  letterEl.className = `dial-letter${label.length > 1 ? " small" : ""}`;
  dial.style.setProperty("--p", speller.progress);
  spellMeta.innerHTML = top[0].p < speller.minConfidence ? "Not sure yet, adjust your hand." : "Hold steady… lower your hand to finish the word.";
  barsEl.innerHTML = top.slice(0, 3).map(({ label: l, p }) =>
    `<li><span>${display(l)}</span><span class="b"><i style="width:${(p * 100).toFixed(1)}%"></i></span><span>${Math.round(p * 100)}%</span></li>`).join("");
}

// ---------- modes ----------

function setMode(mode) {
  state.mode = mode;
  store.set("bw-mode", mode);
  document.querySelectorAll(".modes button").forEach((b) => b.setAttribute("aria-checked", String(b.dataset.mode === mode)));
  $("now-words").hidden = mode !== "words";
  $("now-spell").hidden = mode !== "spell";
  $("vocab-switch").hidden = mode !== "words";
  $("now-hint").textContent = mode === "words" ? "" : "A–Z";
  $("live-label").textContent = mode === "words" ? "Words" : "Spell";
  if (mode === "words" && speller.text.trim()) addWord(speller.text.trim());
  speller.text = "";
  segmenter.flush();
  ghost = "";
  renderWordsIdle();
  renderSpell(null);
  renderCaptions();
}

document.querySelectorAll(".modes button").forEach((b) => { b.onclick = () => setMode(b.dataset.mode); });

// ---------- drawing ----------

const video = $("video");
const canvas = $("overlay");
const ctx = canvas.getContext("2d");
const trails = new Map();

function lerpColor(t) {
  const a = [255, 181, 71], b = [255, 93, 143];
  return a.map((v, i) => Math.round(v + (b[i] - v) * t));
}

function draw(result, now) {
  if (canvas.width !== video.videoWidth) {
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
  }
  const W = canvas.width, H = canvas.height, unit = W / 640;
  ctx.clearRect(0, 0, W, H);

  const seen = new Set();
  result.landmarks.forEach((hand, i) => {
    const key = result.handedness?.[i]?.[0]?.categoryName ?? String(i);
    seen.add(key);
    const pts = hand.map((p) => [p.x * W, p.y * H]);
    let trail = trails.get(key);
    // the tracker sometimes swaps left/right labels; a big jump means a different hand
    const last = trail?.[1].at(-1);
    if (!trail || (last && Math.hypot(last[0] - pts[8][0], last[1] - pts[8][1]) > W * 0.15)) trail = FINGERTIPS.map(() => []);
    FINGERTIPS.forEach((tip, k) => trail[k].push([pts[tip][0], pts[tip][1], now]));
    trails.set(key, trail);

    ctx.lineCap = ctx.lineJoin = "round";
    ctx.strokeStyle = "rgba(244, 239, 248, 0.5)";
    ctx.lineWidth = 2 * unit;
    ctx.beginPath();
    for (const [a, b] of HAND_CONNECTIONS) {
      ctx.moveTo(...pts[a]);
      ctx.lineTo(...pts[b]);
    }
    ctx.stroke();
    ctx.fillStyle = "rgba(244, 239, 248, 0.9)";
    for (const [x, y] of pts) {
      ctx.beginPath();
      ctx.arc(x, y, 2.6 * unit, 0, Math.PI * 2);
      ctx.fill();
    }
  });

  // fingertip ink trails
  ctx.save();
  ctx.globalCompositeOperation = "lighter";
  for (const [key, trail] of trails) {
    let alive = false;
    trail.forEach((pts, k) => {
      while (pts.length && now - pts[0][2] > TRAIL_SECONDS) pts.shift();
      if (pts.length < 2) return;
      alive = true;
      const strong = k === 1 ? 1 : 0.55;
      // two passes, a wide faint halo then a bright core: a glow without shadowBlur's cost
      for (const [widthScale, alphaScale] of [[3.2, 0.18], [1, 1]]) {
        for (let j = 1; j < pts.length; j++) {
          const age = (now - pts[j][2]) / TRAIL_SECONDS;
          const [r, g, b] = lerpColor(1 - age);
          ctx.strokeStyle = `rgba(${r}, ${g}, ${b}, ${(1 - age) * strong * alphaScale})`;
          ctx.lineWidth = ((1 - age) * 7 * unit * strong + 1) * widthScale;
          ctx.beginPath();
          ctx.moveTo(pts[j - 1][0], pts[j - 1][1]);
          ctx.lineTo(pts[j][0], pts[j][1]);
          ctx.stroke();
        }
      }
    });
    if (!alive && !seen.has(key)) trails.delete(key);
  }
  ctx.restore();
}

// ---------- recognition loop ----------

let lastVideoTime = -1;
let lastFrame = performance.now();
let fps = 0;
const perf = {};
let frameCount = 0;
let lastPose = null;
let previewBusy = false;
let lastPreview = 0;

function showSignError(err) {
  console.error(err);
  pulse.classList.remove("reading");
  liveEl.classList.remove("reading");
  setNowWord("Offline", "idle");
  nowMeta.innerHTML = `Word recognition isn't available: ${escape(err?.message ?? String(err))}. Spell mode still works.`;
  altsEl.innerHTML = "";
}

function stepWords(hands, pose, w, h, now) {
  if (!signModel) {
    if (!nowWord.textContent.startsWith("Offline")) showSignError(signModelError);
    return;
  }
  const clip = segmenter.push(frameRecord(hands, pose, w, h), now);
  if (segmenter.active) {
    if (!pulse.classList.contains("reading")) renderReading(null);
    if (!previewBusy && now - lastPreview > 0.4 && segmenter.frames.length >= 8) {
      previewBusy = true;
      lastPreview = now;
      signModel.predict(segmenter.frames.slice(), w, 1)
        .then(([g]) => {
          if (!segmenter.active) return;
          renderReading(g.word);
          ghost = g.word;
          renderCaptions();
        })
        .catch(() => {})
        .finally(() => { previewBusy = false; });
    }
  }
  if (clip) {
    signModel.predict(clip, w).then(onSignResult).catch(showSignError);
  } else if (!segmenter.active && pulse.classList.contains("reading")) {
    ghost = "";
    renderWordsIdle();
    renderCaptions();
  }
}

function stepSpell(hands, w, h, now) {
  const lm = hands.landmarks[0];
  const top = lm ? letters.predict(toFeatures(lm, w, h)) : null;
  const before = speller.text;
  const event = speller.update(top ? top[0].label : null, top ? top[0].p : 0, now);
  renderSpell(top);
  if (!event) return;
  state.lastActivity = now;
  if (event.type === "space") {
    speller.text = "";
    if (event.word) addWord(event.word);
  } else if (event.type === "delete" && !before.trim() && state.phrase.length) {
    state.phrase.pop();
    renderCaptions();
  } else {
    renderCaptions();
  }
}

function loop() {
  if (video.readyState >= 2 && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const nowMs = performance.now();
    const now = nowMs / 1000;
    const w = video.videoWidth, h = video.videoHeight;
    const t0 = performance.now();
    const hands = handLandmarker.detectForVideo(video, nowMs);
    const t1 = performance.now();
    // the body moves slowly next to the fingers, so tracking it every other frame is plenty
    if (state.mode === "words" && (frameCount++ % 2 === 0 || !lastPose)) lastPose = poseLandmarker.detectForVideo(video, nowMs);
    const pose = state.mode === "words" ? lastPose : null;
    const t2 = performance.now();
    const visible = hands.landmarks.length > 0;
    if (visible) state.lastActivity = now;
    draw(hands, now);
    const t3 = performance.now();
    if (state.mode === "words") stepWords(hands, pose, w, h, now);
    else stepSpell(hands, w, h, now);
    maybeFinishPhrase(now, visible);
    const t4 = performance.now();
    const ema = (k, v) => { perf[k] = 0.9 * (perf[k] ?? v) + 0.1 * v; };
    ema("hands", t1 - t0); ema("pose", t2 - t1); ema("draw", t3 - t2); ema("logic", t4 - t3);

    fps = 0.9 * fps + 0.1 * (1000 / Math.max(nowMs - lastFrame, 1));
    lastFrame = nowMs;
    $("fps").textContent = `${fps.toFixed(0)} fps`;
  }
  requestAnimationFrame(loop);
}

// ---------- startup ----------

async function create(Task, modelAssetPath, extra) {
  const options = (delegate) => ({ baseOptions: { modelAssetPath, delegate }, runningMode: "VIDEO", ...extra });
  try {
    return await Task.createFromOptions(vision, options("GPU"));
  } catch {
    return await Task.createFromOptions(vision, options("CPU"));
  }
}

async function loadModels() {
  vision ??= await FilesetResolver.forVisionTasks(WASM_URL);
  [handLandmarker, poseLandmarker, letters, signModel] = await Promise.all([
    create(HandLandmarker, HAND_MODEL_URL, { numHands: 2 }),
    create(PoseLandmarker, POSE_MODEL_URL, { numPoses: 1 }),
    letters ?? LetterClassifier.load("model/asl_classifier.json"),
    loadSignModel(state.vocab).catch((err) => { signModelError = err; return null; }),
  ]);
}

$("start").onclick = async () => {
  const button = $("start");
  const status = $("status");
  button.disabled = true;
  try {
    status.textContent = "Loading hand, body and sign models…";
    const stream = navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 960 }, height: { ideal: 720 }, facingMode: "user" } });
    await loadModels();
    status.textContent = "Waiting for camera permission…";
    video.srcObject = await stream;
    await video.play();
    $("intro").hidden = true;
    liveEl.hidden = false;
    requestAnimationFrame(loop);
  } catch (err) {
    button.disabled = false;
    status.textContent =
      err?.name === "NotAllowedError" ? "Camera access was blocked. Allow it in your browser's site settings, then try again."
        : err?.name === "NotFoundError" ? "No camera found. Connect one and try again."
          : `Couldn't start: ${err?.message ?? err}`;
  }
};

// ---------- vocabulary ----------

// Everyday: 51 conversational signs, ~56% top-1 on unseen signers. All: 250 signs, ~36%.
// Thresholds come from the confidence tables printed by scripts/train_signs.py.
const VOCABS = {
  core: { file: "sign_core", label: "Everyday", threshold: 0.4 },
  full: { file: "sign_full", label: "All", threshold: 0.3 },
};
const vocabModels = {};
const vocabWords = {};
const vocab = $("vocab");
state.vocab = store.get("bw-vocab") === "full" ? "full" : "core";

function loadSignModel(v) {
  vocabModels[v] ??= SignModel.load("model", VOCABS[v].file);
  return vocabModels[v];
}

async function setVocab(v) {
  state.vocab = v;
  store.set("bw-vocab", v);
  document.querySelectorAll("#vocab-switch button").forEach((b) => b.setAttribute("aria-checked", String(b.dataset.vocab === v)));
  renderVocab($("vocab-search").value);
  if (!handLandmarker) return;
  segmenter.flush();
  try {
    signModel = await loadSignModel(v);
    signModelError = null;
    renderWordsIdle();
  } catch (err) {
    signModel = null;
    signModelError = err;
    showSignError(err);
  }
}

for (const [v, { file, label }] of Object.entries(VOCABS)) {
  fetch(`model/${file}.json`)
    .then((r) => r.json())
    .then((meta) => {
      vocabWords[v] = [...meta.display].sort((a, b) => a.localeCompare(b));
      $("vocab-switch").querySelector(`[data-vocab="${v}"]`).textContent = `${label} · ${vocabWords[v].length}`;
      if (v === state.vocab) renderVocab("");
    })
    .catch(() => {});
}

function renderVocab(q) {
  const words = vocabWords[state.vocab] ?? [];
  $("vocab-open").textContent = `${words.length} signs`;
  $("vocab-search").placeholder = `Search ${words.length} signs`;
  const query = q.trim().toLowerCase();
  $("vocab-grid").innerHTML = words
    .filter((w) => w.toLowerCase().includes(query))
    .map((w) => {
      const slug = w.split("/")[0].trim().replace(/\s+/g, "-");
      return `<a href="https://www.signasl.org/sign/${encodeURIComponent(slug)}" target="_blank" rel="noopener">${escape(w)}</a>`;
    })
    .join("");
}

$("vocab-open").onclick = () => vocab.showModal();
$("vocab-search").oninput = (e) => renderVocab(e.target.value);
document.querySelectorAll("#vocab-switch button").forEach((b) => { b.onclick = () => setVocab(b.dataset.vocab); });
setVocab(state.vocab);

setMode(state.mode);

// Test hooks for automated checks without a camera.
window.beyondWords = {
  state,
  perf,
  speller,
  segmenter,
  async predictSign(records, width) {
    return (await loadSignModel(state.vocab)).predict(records, width);
  },
  async classifyImage(src) {
    vision ??= await FilesetResolver.forVisionTasks(WASM_URL);
    letters ??= await LetterClassifier.load("model/asl_classifier.json");
    this.imageLandmarker ??= await HandLandmarker.createFromOptions(vision, {
      baseOptions: { modelAssetPath: HAND_MODEL_URL, delegate: "CPU" }, runningMode: "IMAGE", numHands: 1,
    });
    const img = new Image();
    img.src = src;
    await img.decode();
    const lm = this.imageLandmarker.detect(img).landmarks?.[0];
    return lm ? letters.predict(toFeatures(lm, img.naturalWidth, img.naturalHeight)).slice(0, 3) : null;
  },
};
