import numpy as np
from pathlib import Path

from openwakeword.model import Model

from ev.config import SAMPLE_RATE, PROJECT_ROOT


CUSTOM_MODEL_PATH = PROJECT_ROOT / "models" / "wake_word" / "ev_wakeword.onnx"


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
        self._warmup = 0

    def detect(self, audio_chunk: np.ndarray) -> bool:
        if self._warmup > 0:
            audio_int16 = (audio_chunk * 32767).astype(np.int16)
            self._model.predict(audio_int16)
            self._warmup -= 1
            return False
        audio_int16 = (audio_chunk * 32767).astype(np.int16)
        prediction = self._model.predict(audio_int16)
        for model_name, score in prediction.items():
            if score > self._threshold:
                self._model.reset()
                return True
        return False

    def reset(self):
        self._model.reset()
        self._warmup = 6  # skip 6 chunks (~3s) to let the feature buffer stabilize
