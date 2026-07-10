import numpy as np
from faster_whisper import WhisperModel

from ev.config import WHISPER_MODEL

# Minimum RMS energy to bother transcribing — filters dead silence and mic noise
_RMS_THRESHOLD = 0.01
# Whisper segments above this no-speech probability are hallucinations
_NO_SPEECH_THRESHOLD = 0.6


class SpeechToText:
    def __init__(self):
        self._model = WhisperModel(
            WHISPER_MODEL, device="cpu", compute_type="int8"
        )

    def transcribe(self, audio: np.ndarray) -> str:
        rms = float(np.sqrt(np.mean(audio.astype(np.float32) ** 2)))
        if rms < _RMS_THRESHOLD:
            return ""

        segments, _ = self._model.transcribe(
            audio,
            language="en",
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )
        words = []
        for s in segments:
            if s.no_speech_prob < _NO_SPEECH_THRESHOLD:
                words.append(s.text)
        return " ".join(words).strip()
