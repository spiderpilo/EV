import queue
import time
import numpy as np
import sounddevice as sd

from ev.config import SAMPLE_RATE, AUDIO_CHANNELS

# VAD recording parameters
_CHUNK_MS       = 100    # chunk size in ms
_SPEECH_RMS     = 0.01   # RMS threshold to consider a chunk as speech
_SILENCE_AFTER  = 1.8    # seconds of silence before stopping
_MAX_DURATION   = 30.0   # hard cap so EV doesn't listen forever


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

    def cooldown(self, seconds: float = 2.0):
        """Sleep then flush — lets openwakeword's sliding window age out."""
        time.sleep(seconds)
        self.flush()

    def record(self, duration: float | None = None) -> np.ndarray:
        """Record until silence is detected or the hard cap is hit."""
        chunk_samples = int(self._sample_rate * _CHUNK_MS / 1000)
        silence_chunks_needed = int(_SILENCE_AFTER * 1000 / _CHUNK_MS)
        max_chunks = int((_MAX_DURATION if duration is None else duration) * 1000 / _CHUNK_MS)

        print("  Listening...")
        recorded: list[np.ndarray] = []
        silence_count = 0
        started_speaking = False

        with sd.InputStream(
            samplerate=self._sample_rate,
            channels=self._channels,
            dtype="float32",
            blocksize=chunk_samples,
            callback=self._callback,
        ):
            self.flush()
            for _ in range(max_chunks):
                try:
                    chunk = self._queue.get(timeout=1.0)
                except queue.Empty:
                    break

                flat = chunk.flatten()
                recorded.append(flat)
                rms = float(np.sqrt(np.mean(flat ** 2)))

                if rms >= _SPEECH_RMS:
                    started_speaking = True
                    silence_count = 0
                else:
                    if started_speaking:
                        silence_count += 1
                        if silence_count >= silence_chunks_needed:
                            break

        self.flush()
        return np.concatenate(recorded) if recorded else np.zeros(chunk_samples, dtype=np.float32)
