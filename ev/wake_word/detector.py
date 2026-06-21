import numpy as np
from openwakeword.model import Model

from ev.config import SAMPLE_RATE


class WakeWordDetector:
    def __init__(self):
        self._model = Model(
            wakeword_models=["hey_jarvis"],
            inference_framework="onnx",
        )
        self._threshold = 0.5

    def detect(self, audio_chunk: np.ndarray) -> bool:
        audio_int16 = (audio_chunk * 32767).astype(np.int16)
        prediction = self._model.predict(audio_int16)
        for model_name, score in prediction.items():
            if score > self._threshold:
                self._model.reset()
                return True
        return False

    def reset(self):
        self._model.reset()
