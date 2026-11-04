"""Non-blocking text-to-speech so the camera loop never stalls while talking."""

import queue
import threading


class Speaker:
    def __init__(self, rate=165):
        self.rate = rate
        self._queue = queue.Queue()
        threading.Thread(target=self._run, daemon=True).start()

    def say(self, text):
        if text and text.strip():
            self._queue.put(text.strip())

    def _run(self):
        import pyttsx3

        while True:
            text = self._queue.get()
            # A fresh engine per utterance: reusing one across runAndWait() calls
            # from a background thread is unreliable in pyttsx3.
            engine = pyttsx3.init()
            engine.setProperty("rate", self.rate)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
            del engine
