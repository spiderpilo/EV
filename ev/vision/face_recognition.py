import pickle
from pathlib import Path

import cv2
import numpy as np
import face_recognition as fr

from ev.config import FACE_EMBEDDINGS_DIR, FACE_RECOGNITION_TOLERANCE


class FaceRecognizer:
    def __init__(self):
        self._tolerance = FACE_RECOGNITION_TOLERANCE
        self._enrolled_encodings: list[np.ndarray] = []
        self._cap: cv2.VideoCapture | None = None
        self._load_enrollment()

    def _embedding_path(self) -> Path:
        return FACE_EMBEDDINGS_DIR / "owner_face.pkl"

    def _load_enrollment(self):
        path = self._embedding_path()
        if path.exists():
            with open(path, "rb") as f:
                self._enrolled_encodings = pickle.load(f)

    def enroll(self, images: list[np.ndarray]):
        encodings = []
        for img in images:
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            found = fr.face_encodings(rgb)
            if found:
                encodings.append(found[0])
        if not encodings:
            raise ValueError("No faces detected in the provided images.")
        self._enrolled_encodings = encodings
        self._embedding_path().parent.mkdir(parents=True, exist_ok=True)
        with open(self._embedding_path(), "wb") as f:
            pickle.dump(self._enrolled_encodings, f)
        print(f"  Face enrolled from {len(encodings)} images.")

    @property
    def is_enrolled(self) -> bool:
        return len(self._enrolled_encodings) > 0

    def capture_frame(self) -> np.ndarray | None:
        if self._cap is None:
            self._cap = cv2.VideoCapture(0)
        ret, frame = self._cap.read()
        return frame if ret else None

    def recognize(self, frame: np.ndarray | None = None) -> tuple[bool, str]:
        if frame is None:
            frame = self.capture_frame()
        if frame is None:
            return False, "no_camera"
        if not self.is_enrolled:
            return False, "not_enrolled"

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = fr.face_locations(rgb)
        if not face_locations:
            return False, "no_face"

        face_encodings = fr.face_encodings(rgb, face_locations)
        for encoding in face_encodings:
            matches = fr.compare_faces(
                self._enrolled_encodings, encoding, tolerance=self._tolerance
            )
            if any(matches):
                return True, "owner"
        return False, "unknown"

    def release(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None
