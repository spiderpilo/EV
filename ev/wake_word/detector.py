import time
import numpy as np
from pathlib import Path

from openwakeword.model import Model

from ev.config import SAMPLE_RATE, PROJECT_ROOT

CUSTOM_MODEL_PATH = PROJECT_ROOT / "models" / "wake_word" / "ev_wakeword.onnx"
_COOLDOWN_SEC = 4.0


class WakeWordDetector:
    def __init__(self):
        if CUSTOM_MODEL_PATH.exists():
            self._model = Model(
                wakeword_models=[str(CUSTOM_MODEL_PATH)],
                inference_framework="onnx",
            )
            self._model_name = CUSTOM_MODEL_PATH.stem
        else:
            self._model = Model(
                wakeword_models=["hey_jarvis"],
                inference_framework="onnx",
            )
            self._model_name = "hey_jarvis"
        self._threshold = 0.6 if CUSTOM_MODEL_PATH.exists() else 0.7
        self._ready_at = 0.0
        self._consecutive = 0

    def detect(self, audio_chunk: np.ndarray) -> bool:
        audio_int16 = (audio_chunk * 32767).astype(np.int16)
        prediction = self._model.predict(audio_int16)
        if time.time() < self._ready_at:
            self._consecutive = 0
            return False
        hit = any(score > self._threshold for score in prediction.values())
        if hit:
            self._consecutive += 1
        else:
            self._consecutive = 0
        if self._consecutive >= 2:
            self._consecutive = 0
            self._model.reset()
            return True
        return False

    def reset(self):
        self._model.reset()
        self._consecutive = 0
        self._ready_at = time.time() + _COOLDOWN_SEC
