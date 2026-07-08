"""
Record real voice samples of "EV" for wake word training.

Usage:
    python -m scripts.record_wake_word

Records N clips of you saying "EV" and saves them to models/wake_word/clips/real_positive/.
Then re-trains the wake word model with these real samples mixed in.
"""
import time
from pathlib import Path

import numpy as np
import scipy.io.wavfile
import sounddevice as sd

SAMPLE_RATE = 16000
RECORD_SECONDS = 2
CLIP_SAMPLES = SAMPLE_RATE * RECORD_SECONDS
REAL_POSITIVE_DIR = Path("models/wake_word/clips/real_positive")
N_RECORDINGS = 30


def record_clip() -> np.ndarray:
    audio = sd.rec(CLIP_SAMPLES, samplerate=SAMPLE_RATE, channels=1, dtype="int16")
    sd.wait()
    return audio.flatten()


def play_beep(freq=880, duration=0.15):
    t = np.linspace(0, duration, int(SAMPLE_RATE * duration), False)
    tone = (np.sin(2 * np.pi * freq * t) * 8000).astype(np.int16)
    sd.play(tone, SAMPLE_RATE)
    sd.wait()


def main():
    print("=" * 50)
    print("  EV — Wake Word Recorder")
    print("=" * 50)
    print(f"\nWe'll record {N_RECORDINGS} clips of you saying 'EV'.")
    print("A beep will signal when to speak. Say it naturally — short and clear.\n")

    REAL_POSITIVE_DIR.mkdir(parents=True, exist_ok=True)

    existing = list(REAL_POSITIVE_DIR.glob("*.wav"))
    start_idx = len(existing)
    if start_idx:
        print(f"  Found {start_idx} existing recordings. Adding to them.\n")

    input("Press Enter when you're ready to start...")
    print()

    saved = 0
    for i in range(N_RECORDINGS):
        print(f"  [{i + 1}/{N_RECORDINGS}] Get ready... ", end="", flush=True)
        time.sleep(0.8)
        play_beep(880, 0.12)
        print("say 'EV' now!")

        audio = record_clip()

        # Check the clip has actual audio (not silence)
        rms = np.sqrt(np.mean(audio.astype(np.float32) ** 2))
        if rms < 200:
            print("    (too quiet, skipping — try speaking louder)")
            continue

        path = REAL_POSITIVE_DIR / f"real_{start_idx + saved:04d}.wav"
        scipy.io.wavfile.write(str(path), SAMPLE_RATE, audio)
        print(f"    Saved: {path.name}  (rms={rms:.0f})")
        saved += 1

        time.sleep(0.3)

    total = start_idx + saved
    print(f"\nDone! Recorded {saved} new clips ({total} total).")
    print(f"Saved to: {REAL_POSITIVE_DIR}")
    print("\nNow re-train with your real voice mixed in:")
    print("  python -m scripts.train_wake_word")


if __name__ == "__main__":
    main()
