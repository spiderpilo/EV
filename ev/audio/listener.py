import queue
import numpy as np
import sounddevice as sd

from ev.config import SAMPLE_RATE, AUDIO_CHANNELS, LISTEN_DURATION_SEC


class AudioListener:
    def __init__(self):
        self._sample_rate = SAMPLE_RATE
        self._channels = AUDIO_CHANNELS
        self._queue: queue.Queue[np.ndarray] = queue.Queue()

    def _callback(self, indata, frames, time_info, status):
        self._queue.put(indata.copy())

    def stream_chunks(self, chunk_duration: float = 0.5):
        chunk_size = int(self._sample_rate * chunk_duration)
        with sd.InputStream(
            samplerate=self._sample_rate,
            channels=self._channels,
            dtype="float32",
            blocksize=chunk_size,
            callback=self._callback,
        ):
            while True:
                chunk = self._queue.get()
                yield chunk.flatten()

    def flush(self):
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except Exception:
                break

    def cooldown(self, seconds: float = 1.5):
        """Sleep then flush — lets openwakeword's sliding window age out."""
        import time
        time.sleep(seconds)
        self.flush()

    def record(self, duration: float | None = None) -> np.ndarray:
        duration = duration or LISTEN_DURATION_SEC
        print(f"  Listening for {duration}s...")
        audio = sd.rec(
            int(self._sample_rate * duration),
            samplerate=self._sample_rate,
            channels=self._channels,
            dtype="float32",
        )
        sd.wait()
        self.flush()
        return audio.flatten()
