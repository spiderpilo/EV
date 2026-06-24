import numpy as np
from faster_whisper import WhisperModel

from ev.config import WHISPER_MODEL


class SpeechToText:
    def __init__(self):
        self._model = WhisperModel(
            WHISPER_MODEL, device="cpu", compute_type="int8"
        )

    def transcribe(self, audio: np.ndarray) -> str:
        segments, _ = self._model.transcribe(audio, language="en")
        return " ".join(s.text for s in segments).strip()
