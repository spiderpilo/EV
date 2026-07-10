import subprocess
import platform

import numpy as np
import sounddevice as sd
from scipy.signal import resample

from ev.config import PROJECT_ROOT, TTS_PITCH_SHIFT

PIPER_VOICE_PATH = PROJECT_ROOT / "models" / "tts" / "en_US-lessac-medium.onnx"


class TextToSpeech:
    def __init__(self):
        self._system = platform.system()
        self._piper_voice = None
        self._sample_rate = None

        if PIPER_VOICE_PATH.exists():
            from piper import PiperVoice
            self._piper_voice = PiperVoice.load(str(PIPER_VOICE_PATH))
            self._sample_rate = self._piper_voice.config.sample_rate

    def chime(self):
        """Play a short ascending three-note chime."""
        sr = 44100
        notes = [523.25, 659.25, 783.99]  # C5, E5, G5
        note_dur = 0.12
        gap = 0.03
        clips = []
        for freq in notes:
            t = np.linspace(0, note_dur, int(sr * note_dur), False)
            wave = np.sin(2 * np.pi * freq * t)
            fade = np.linspace(1.0, 0.0, len(t)) ** 2
            clips.append((wave * fade * 0.4).astype(np.float32))
            clips.append(np.zeros(int(sr * gap), dtype=np.float32))
        audio = np.concatenate(clips)
        sd.play(audio, samplerate=sr)
        sd.wait()

    def speak(self, text: str):
        if self._piper_voice:
            chunks = []
            for chunk in self._piper_voice.synthesize(text):
                chunks.append(chunk.audio_int16_array)
            audio = np.concatenate(chunks).astype(np.float32) / 32768.0
            if TTS_PITCH_SHIFT != 1.0:
                audio = resample(audio, int(len(audio) / TTS_PITCH_SHIFT))
            sd.play(audio, samplerate=self._sample_rate)
            sd.wait()
        elif self._system == "Darwin":
            subprocess.run(["say", text], check=True)
        else:
            print(f"  EV says: {text}")
