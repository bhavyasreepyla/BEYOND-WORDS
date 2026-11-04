"""Turns a noisy stream of per-frame predictions into typed text.

A sign is accepted once it has been held steadily for `hold_seconds`. Holding
the same sign for another `repeat_seconds` types it once more (for double
letters like the "LL" in HELLO); after that, drop the hand out of view to
type it again. Keeping the hand out of view for `word_pause` seconds ends the
current word, exactly like signing "space".

Mirrored by the Speller class in web/letters.js.
"""

from dataclasses import dataclass, field

from .labels import DELETE, SPACE


@dataclass
class Speller:
    hold_seconds: float = 0.7
    repeat_seconds: float = 1.4
    min_confidence: float = 0.6
    text: str = ""
    candidate: str | None = None
    since: float = 0.0
    last_committed: str | None = None
    repeated: bool = False
    word_pause: float = 1.0
    gone_since: float | None = None
    progress: float = field(default=0.0, init=False)

    def update(self, label, confidence, now):
        """Feed one frame. Returns ("letter"|"space"|"delete", value) when something is typed."""
        if label is None:
            self.candidate = None
            self.last_committed = None
            self.progress = 0.0
            if self.gone_since is None:
                self.gone_since = now
            elif now - self.gone_since >= self.word_pause and self.text and not self.text.endswith(" "):
                return self._commit(SPACE)
            return None
        self.gone_since = None
        if confidence < self.min_confidence:
            self.progress = 0.0
            return None

        if label != self.candidate:
            self.candidate = label
            self.since = now
        is_repeat = label == self.last_committed
        if is_repeat and self.repeated:
            self.progress = 0.0
            return None
        needed = self.repeat_seconds if is_repeat else self.hold_seconds
        self.progress = min(1.0, (now - self.since) / needed)
        if self.progress < 1.0:
            return None

        self.since = now
        self.last_committed = label
        self.repeated = is_repeat
        return self._commit(label)

    def _commit(self, label):
        if label == DELETE:
            self.text = self.text[:-1]
            return "delete", None
        if label == SPACE:
            word = self.current_word()
            if self.text and not self.text.endswith(" "):
                self.text += " "
            return "space", word
        self.text += label
        return "letter", label

    def current_word(self):
        return self.text.rstrip().rsplit(" ", 1)[-1] if self.text.strip() else ""

    def clear(self):
        self.text = ""
        self.candidate = None
        self.last_committed = None
        self.progress = 0.0
