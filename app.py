"""Beyond Words desktop app: sign ASL to your webcam, get live captions and speech.

    python app.py                 # default webcam, Words mode (51 everyday signs)
    python app.py --all-signs     # recognize all 250 signs (less accurate)
    python app.py --spell         # start in fingerspelling mode
    python app.py --camera 1      # another camera
    python app.py --source clip.mp4 --no-speech

Keys: M switch Words/Spell | 1-3 pick an alternative word | BACKSPACE delete
      C clear | Q / ESC quit
"""

import argparse
import time
from collections import deque

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from beyondwords.classifier import Classifier
from beyondwords.labels import DELETE, SPACE
from beyondwords.landmarks import create_landmarker, landmarks_to_features, to_mp_image
from beyondwords.signs import SignModel, SignSegmenter, create_pose_landmarker, frame_record
from beyondwords.speller import Speller

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20),
]
FINGERTIPS = [4, 8, 12, 16, 20]
PANEL_W = 400
PHRASE_IDLE_SECONDS = 3.5
TRAIL_SECONDS = 0.45

# RGB, matching web/style.css
BG = (16, 12, 22)
SURFACE = (25, 20, 31)
LINE = (48, 40, 58)
TEXT = (244, 239, 248)
MUTED = (157, 147, 172)
FAINT = (94, 85, 105)
AMBER = (255, 181, 71)
ROSE = (255, 93, 143)

FONT_PATHS = {
    "regular": ["C:/Windows/Fonts/segoeui.ttf", "/System/Library/Fonts/Helvetica.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"],
    "bold": ["C:/Windows/Fonts/segoeuib.ttf", "/System/Library/Fonts/Helvetica.ttc", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"],
}
_fonts = {}


def font(size, weight="regular"):
    key = (size, weight)
    if key not in _fonts:
        for path in FONT_PATHS[weight]:
            try:
                _fonts[key] = ImageFont.truetype(path, size)
                break
            except OSError:
                continue
        else:
            _fonts[key] = ImageFont.load_default(size=size)
    return _fonts[key]


def mix(a, b, t):
    return tuple(int(x + (y - x) * t) for x, y in zip(a, b))


def bgr(rgb):
    return rgb[::-1]


def display_letter(label):
    return {SPACE: "space", DELETE: "del"}.get(label, label)


class App:
    def __init__(self, args):
        self.mode = "spell" if args.spell else "words"
        self.letters = Classifier.load()
        self.signs = SignModel("sign_full" if args.all_signs else "sign_core")
        # from the confidence tables printed by scripts/train_signs.py
        self.min_sign_confidence = 0.3 if args.all_signs else 0.4
        self.hands = create_landmarker(video=True, num_hands=2)
        self.pose = create_pose_landmarker(video=True)
        self.speller = Speller(hold_seconds=args.hold)
        self.segmenter = SignSegmenter(min_frames=8)
        self.speaker = None
        if not args.no_speech:
            from beyondwords.speech import Speaker

            self.speaker = Speaker()
        self.phrase = []
        self.history = deque(maxlen=6)
        self.last_activity = 0.0
        self.result = None
        self.alts = []
        self.letter_top = []
        self.trails = {}
        self.fps = 0.0

    # ---- state ----

    def say(self, text):
        if self.speaker:
            self.speaker.say(text.lower())

    def add_word(self, word, now):
        self.phrase.append(word)
        self.last_activity = now
        self.say(word)

    def switch_mode(self):
        if self.mode == "spell" and self.speller.text.strip():
            self.add_word(self.speller.text.strip(), time.monotonic())
        self.mode = "spell" if self.mode == "words" else "words"
        self.speller.text = ""
        self.segmenter.flush()
        self.result, self.alts = None, []

    def pick_alternative(self, i):
        if i < len(self.alts):
            word = self.alts[i][0]
            if self.result and self.phrase:
                self.phrase[-1] = word
            else:
                self.phrase.append(word)
            self.result, self.alts = (word, self.alts[i][1]), []
            self.say(word)

    def step(self, frame, now, ts):
        h, w = frame.shape[:2]
        image = to_mp_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        hand_result = self.hands.detect_for_video(image, ts)
        visible = bool(hand_result.hand_landmarks)
        if visible:
            self.last_activity = now

        if self.mode == "words":
            pose_result = self.pose.detect_for_video(image, ts)
            clip = self.segmenter.push(frame_record(hand_result, pose_result, w, h), now)
            if clip:
                top = self.signs.predict(clip, w)
                if top[0][1] >= self.min_sign_confidence:
                    self.add_word(top[0][0], now)
                    self.result, self.alts = top[0], top[1:4]
                else:
                    self.result, self.alts = None, top[:4]
        else:
            self.letter_top = []
            label, conf = None, 0.0
            if visible:
                lm = np.array([[p.x, p.y, p.z] for p in hand_result.hand_landmarks[0]], dtype=np.float32)
                proba = self.letters.predict_proba(landmarks_to_features(lm, w, h))[0]
                self.letter_top = [(self.letters.labels[i], float(proba[i])) for i in proba.argsort()[::-1][:3]]
                label, conf = self.letter_top[0]
            before = self.speller.text
            event = self.speller.update(label, conf, now)
            if event:
                self.last_activity = now
                if event[0] == "space":
                    self.speller.text = ""
                    if event[1]:
                        self.add_word(event[1], now)
                elif event[0] == "delete" and not before.strip() and self.phrase:
                    self.phrase.pop()

        idle = not visible and not self.segmenter.active and not self.speller.text.strip()
        if self.phrase and idle and now - self.last_activity > PHRASE_IDLE_SECONDS:
            self.history.append(" ".join(self.phrase))
            self.phrase, self.result, self.alts = [], None, []
        return hand_result

    # ---- drawing ----

    def draw_hands(self, frame, hand_result, now):
        h, w = frame.shape[:2]
        unit = w / 640
        seen = set()
        for i, hand in enumerate(hand_result.hand_landmarks):
            key = hand_result.handedness[i][0].category_name if hand_result.handedness else str(i)
            seen.add(key)
            pts = [(int(p.x * w), int(p.y * h)) for p in hand]
            trail = self.trails.setdefault(key, [deque() for _ in FINGERTIPS])
            for k, tip in enumerate(FINGERTIPS):
                trail[k].append((pts[tip], now))
            for a, b in HAND_CONNECTIONS:
                cv2.line(frame, pts[a], pts[b], bgr(mix(TEXT, BG, 0.4)), max(1, int(2 * unit)), cv2.LINE_AA)
            for p in pts:
                cv2.circle(frame, p, max(2, int(3 * unit)), bgr(TEXT), -1, cv2.LINE_AA)

        glow = np.zeros_like(frame)
        for key in list(self.trails):
            alive = False
            for k, pts in enumerate(self.trails[key]):
                while pts and now - pts[0][1] > TRAIL_SECONDS:
                    pts.popleft()
                strong = 1.0 if k == 1 else 0.55
                for j in range(1, len(pts)):
                    alive = True
                    age = (now - pts[j][1]) / TRAIL_SECONDS
                    color = mix(ROSE, AMBER, age)
                    color = tuple(int(c * (1 - age) * strong) for c in color)
                    cv2.line(glow, pts[j - 1][0], pts[j][0], bgr(color), max(1, int((1 - age) * 9 * unit * strong)), cv2.LINE_AA)
            if not alive and key not in seen:
                del self.trails[key]
        blur = cv2.GaussianBlur(glow, (0, 0), 6 * unit)
        cv2.add(frame, glow, frame)
        cv2.add(frame, blur, frame)

    def compose(self, frame):
        h, w = frame.shape[:2]
        # darken the bottom of the video for captions
        ramp = np.linspace(0, 0.75, h // 3, dtype=np.float32)[:, None, None]
        frame[h - h // 3 :] = (frame[h - h // 3 :] * (1 - ramp)).astype(np.uint8)

        canvas = Image.new("RGB", (w + PANEL_W, h), BG)
        canvas.paste(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), (0, 0))
        d = ImageDraw.Draw(canvas)

        # captions
        words = list(self.phrase)
        spelling = self.speller.text.strip() if self.mode == "spell" else ""
        size = max(28, h // 16)
        line = " ".join(words[-8:])
        x, y = 36, h - size - 40
        d.text((x, y), line, font=font(size, "bold"), fill=TEXT)
        if spelling:
            x += d.textlength(line + (" " if line else ""), font=font(size, "bold"))
            d.text((x, y), spelling, font=font(size, "bold"), fill=AMBER)
            x += d.textlength(spelling, font=font(size, "bold"))
            d.rectangle((x + 3, y + 8, x + 6, y + size + 4), fill=ROSE)

        # live pill
        label = f"  {self.mode.upper()}  {self.fps:.0f} FPS"
        d.rounded_rectangle((18, 18, 34 + d.textlength(label, font=font(13, "bold")), 46), 14, fill=(16, 12, 22))
        d.ellipse((26, 28, 34, 36), fill=AMBER if self.segmenter.active else ROSE)
        d.text((32, 23), label, font=font(13, "bold"), fill=TEXT)

        self.draw_panel(d, w, h)
        return cv2.cvtColor(np.asarray(canvas), cv2.COLOR_RGB2BGR)

    def draw_panel(self, d, x0, h):
        px = x0 + 28
        d.rectangle((x0, 0, x0 + PANEL_W, h), fill=BG)
        d.text((px, 26), "beyond", font=font(24), fill=TEXT)
        d.text((px + d.textlength("beyond", font=font(24)), 26), "words", font=font(24, "bold"), fill=ROSE)

        # mode pill
        py = 76
        d.rounded_rectangle((px, py, px + 220, py + 36), 18, fill=SURFACE, outline=LINE)
        for i, name in enumerate(["words", "spell"]):
            bx = px + 4 + i * 108
            if self.mode == name:
                d.rounded_rectangle((bx, py + 4, bx + 104, py + 32), 14, fill=TEXT)
            d.text((bx + 52, py + 18), name.capitalize(), font=font(15, "bold"), fill=BG if self.mode == name else MUTED, anchor="mm")

        # reading card
        cy = 136
        d.rounded_rectangle((px - 8, cy, x0 + PANEL_W - 20, cy + 250), 22, fill=SURFACE, outline=LINE)
        d.text((px + 8, cy + 18), "READING", font=font(12, "bold"), fill=MUTED)
        if self.mode == "words":
            self.draw_words_reading(d, px + 8, cy + 46)
        else:
            self.draw_spell_reading(d, px + 8, cy + 46)

        # conversation
        cy = 404
        d.text((px, cy), "CONVERSATION", font=font(12, "bold"), fill=MUTED)
        y = cy + 28
        if not self.history:
            d.text((px, y), "Finished phrases land here.", font=font(14), fill=FAINT)
        for msg in list(self.history)[-5:]:
            tw = min(d.textlength(msg, font=font(16)), PANEL_W - 90)
            right = x0 + PANEL_W - 28
            d.rounded_rectangle((right - tw - 28, y, right, y + 36), 16, fill=ROSE)
            d.text((right - 14, y + 18), msg, font=font(16, "bold"), fill=(26, 15, 20), anchor="rm")
            y += 44

        hints = "M mode   1-3 fix word   BKSP delete   C clear   Q quit"
        d.text((px, h - 34), hints, font=font(12), fill=FAINT)

    def draw_words_reading(self, d, x, y):
        if self.segmenter.active:
            d.text((x, y), "Reading…", font=font(40, "bold"), fill=MUTED)
            d.text((x, y + 60), "Lower your hands to finish the sign.", font=font(14), fill=MUTED)
            return
        if self.result:
            d.text((x, y), self.result[0], font=font(42, "bold"), fill=AMBER)
            d.text((x, y + 60), f"{self.result[1]:.0%} sure · spoken aloud", font=font(14), fill=MUTED)
        elif self.alts:
            d.text((x, y), "?", font=font(42, "bold"), fill=FAINT)
            d.text((x, y + 60), "Didn't catch that. Press 1-3 if it was:", font=font(14), fill=MUTED)
        else:
            d.text((x, y), "Ready", font=font(42, "bold"), fill=FAINT)
            d.text((x, y + 60), "Raise your hands, sign, lower them.", font=font(14), fill=MUTED)
        for i, (word, p) in enumerate(self.alts[:3]):
            d.text((x, y + 100 + i * 28), f"{i + 1}  {word}", font=font(16), fill=TEXT)
            d.text((x + 300, y + 100 + i * 28), f"{p:.0%}", font=font(13), fill=FAINT, anchor="ra")

    def draw_spell_reading(self, d, x, y):
        cx, cy, r = x + 58, y + 70, 56
        d.ellipse((cx - r, cy - r, cx + r, cy + r), outline=LINE, width=7)
        if self.speller.progress > 0:
            d.arc((cx - r, cy - r, cx + r, cy + r), -90, -90 + 360 * self.speller.progress, fill=ROSE, width=7)
        letter = display_letter(self.letter_top[0][0]) if self.letter_top else "—"
        d.text((cx, cy), letter, font=font(48 if len(letter) == 1 else 22, "bold"), fill=TEXT, anchor="mm")
        for i, (label, p) in enumerate(self.letter_top):
            by = y + 22 + i * 30
            d.text((x + 140, by), display_letter(label), font=font(14), fill=MUTED)
            d.rounded_rectangle((x + 190, by + 6, x + 320, by + 12), 3, fill=LINE)
            d.rounded_rectangle((x + 190, by + 6, x + 190 + int(130 * p), by + 12), 3, fill=AMBER)
        msg = "Lower your hand to finish the word." if self.speller.text.strip() else "Hold a letter steady to type it."
        d.text((x, y + 160), msg, font=font(14), fill=MUTED)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--source", help="Video file to use instead of a webcam")
    parser.add_argument("--spell", action="store_true", help="Start in fingerspelling mode")
    parser.add_argument("--all-signs", action="store_true", help="Use the full 250-sign vocabulary")
    parser.add_argument("--no-speech", action="store_true")
    parser.add_argument("--hold", type=float, default=0.7, help="Seconds to hold a letter before it is typed")
    args = parser.parse_args()

    app = App(args)
    cap = cv2.VideoCapture(args.source if args.source else args.camera)
    if not cap.isOpened():
        raise SystemExit("Could not open the camera. Try --camera 1, or close other apps using it.")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    window = "Beyond Words"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    start = last = time.monotonic()
    last_ts = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if not args.source:
            frame = cv2.flip(frame, 1)
        now = time.monotonic()
        ts = max(int((now - start) * 1000), last_ts + 1)
        last_ts = ts

        hand_result = app.step(frame, now, ts)
        app.draw_hands(frame, hand_result, now)
        app.fps = 0.9 * app.fps + 0.1 / max(now - last, 1e-6)
        last = now
        cv2.imshow(window, app.compose(frame))

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord("m"):
            app.switch_mode()
        elif key in (ord("1"), ord("2"), ord("3")):
            app.pick_alternative(key - ord("1"))
        elif key == 8:
            if app.speller.text:
                app.speller.text = app.speller.text[:-1]
            elif app.phrase:
                app.phrase.pop()
        elif key == ord("c"):
            app.speller.clear()
            app.phrase, app.result, app.alts = [], None, []
            app.history.clear()
        if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
