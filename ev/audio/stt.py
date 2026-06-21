import numpy as np
import mlx_whisper

from ev.config import WHISPER_MODEL


class SpeechToText:
    def __init__(self):
        self._model_path = WHISPER_MODEL

    def transcribe(self, audio: np.ndarray) -> str:
        result = mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=self._model_path,
            language="en",
        )
        return result["text"].strip()
