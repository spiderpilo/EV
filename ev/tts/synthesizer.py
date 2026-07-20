import re
import subprocess
import platform

import numpy as np
import sounddevice as sd

from ev.config import TTS_VOICE


def _clean_for_speech(text: str) -> str:
    """Strip markdown so TTS reads clean spoken text, not symbols."""
    # Remove fenced code blocks entirely — code doesn't speak well
    text = re.sub(r"```[\s\S]*?```", "...see the code above...", text)
    # Remove inline code backticks but keep the content
    text = re.sub(r"`([^`]+)`", r"\1", text)
    # Remove markdown bold/italic markers
    text = re.sub(r"\*{1,3}([^*]+)\*{1,3}", r"\1", text)
    # Remove markdown headers (#, ##, etc.)
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Collapse extra whitespace
    text = re.sub(r"\n{2,}", " ", text)
    return text.strip()

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
        text = _clean_for_speech(text)
        if not text:
            return
        if self._pipeline:
            try:
                chunks = []
                for _, _, audio in self._pipeline(text, voice=TTS_VOICE):
                    chunks.append(audio)
                if chunks:
                    audio = np.concatenate(chunks).astype(np.float32)
                    sd.play(audio, samplerate=SAMPLE_RATE)
                    sd.wait()
            except Exception as e:
                print(f"  [TTS ERROR] {e}")
        elif self._system == "Darwin":
            subprocess.run(["say", text], check=True)
        else:
            print(f"  EV says: {text}")
