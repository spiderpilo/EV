import pickle
from pathlib import Path

import numpy as np
import torch
from speechbrain.inference.speaker import SpeakerRecognition

from ev.config import VOICE_EMBEDDINGS_DIR, SPEAKER_VERIFICATION_THRESHOLD, SAMPLE_RATE


class SpeakerVerifier:
    def __init__(self):
        self._model = SpeakerRecognition.from_hparams(
            source="speechbrain/spkrec-ecapa-voxceleb",
            savedir=str(VOICE_EMBEDDINGS_DIR / ".model_cache"),
        )
        self._threshold = SPEAKER_VERIFICATION_THRESHOLD
        self._enrolled_embedding: np.ndarray | None = None
        self._load_enrollment()

    def _embedding_path(self) -> Path:
        return VOICE_EMBEDDINGS_DIR / "owner_embedding.pkl"

    def _load_enrollment(self):
        path = self._embedding_path()
        if path.exists():
            with open(path, "rb") as f:
                self._enrolled_embedding = pickle.load(f)

    def enroll(self, audio_samples: list[np.ndarray]):
        embeddings = []
        for audio in audio_samples:
            tensor = torch.tensor(audio).unsqueeze(0)
            emb = self._model.encode_batch(tensor).squeeze().cpu().numpy()
            embeddings.append(emb)
        self._enrolled_embedding = np.mean(embeddings, axis=0)
        self._embedding_path().parent.mkdir(parents=True, exist_ok=True)
        with open(self._embedding_path(), "wb") as f:
            pickle.dump(self._enrolled_embedding, f)
        print(f"  Voice enrolled from {len(audio_samples)} samples.")

    @property
    def is_enrolled(self) -> bool:
        return self._enrolled_embedding is not None

    def verify(self, audio: np.ndarray) -> tuple[bool, float]:
        if not self.is_enrolled:
            return False, 0.0
        tensor = torch.tensor(audio).unsqueeze(0)
        emb = self._model.encode_batch(tensor).squeeze().cpu().numpy()
        similarity = np.dot(emb, self._enrolled_embedding) / (
            np.linalg.norm(emb) * np.linalg.norm(self._enrolled_embedding)
        )
        return float(similarity) >= self._threshold, float(similarity)
