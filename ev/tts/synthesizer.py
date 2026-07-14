import subprocess
import platform

import numpy as np
import sounddevice as sd

from ev.config import TTS_VOICE

SAMPLE_RATE = 24000


class TextToSpeech:
    def __init__(self):
        self._system = platform.system()
        self._pipeline = None
        try:
            from kokoro import KPipeline
            self._pipeline = KPipeline(lang_code='a')
            print(f"  Kokoro TTS loaded. Voice: {TTS_VOICE}")
        except Exception as e:
            print(f"  Kokoro not available: {e}")

    def chime(self):
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
        if self._pipeline:
            chunks = []
            for _, _, audio in self._pipeline(text, voice=TTS_VOICE):
                chunks.append(audio)
            audio = np.concatenate(chunks).astype(np.float32)
            sd.play(audio, samplerate=SAMPLE_RATE)
            sd.wait()
        elif self._system == "Darwin":
            subprocess.run(["say", text], check=True)
        else:
            print(f"  EV says: {text}")
