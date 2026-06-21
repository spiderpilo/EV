import os
import time
from pathlib import Path

import cv2
import numpy as np
import sounddevice as sd
import soundfile as sf
import speech_recognition as sr
from dotenv import load_dotenv
import openai

BASE_DIR = Path(__file__).resolve().parent
TMP_AUDIO_PATH = BASE_DIR / "tmp_audio.wav"


def load_env() -> None:
    env_path = BASE_DIR / ".env"
    load_dotenv(dotenv_path=env_path)
    openai.api_key = os.getenv("OPENAI_API_KEY")


def capture_frame() -> np.ndarray | None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Unable to open webcam.")
        return None

    ret, frame = cap.read()
    cap.release()
    if not ret:
        print("Unable to capture a frame.")
        return None

    cv2.imshow("EV Camera", frame)
    cv2.waitKey(1000)
    cv2.destroyAllWindows()
    return frame


def record_audio(duration: float = 3.0, sample_rate: int = 16000) -> Path:
    print(f"Recording audio for {duration:.1f} seconds...")
    recording = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=1, dtype="int16")
    sd.wait()
    sf.write(TMP_AUDIO_PATH, recording, sample_rate)
    print(f"Saved audio to {TMP_AUDIO_PATH}")
    return TMP_AUDIO_PATH


def transcribe_audio(audio_path: Path) -> str | None:
    recognizer = sr.Recognizer()
    with sr.AudioFile(str(audio_path)) as source:
        audio_data = recognizer.record(source)

    try:
        text = recognizer.recognize_google(audio_data)
        print(f"Transcribed text: {text}")
        return text.lower()
    except sr.UnknownValueError:
        print("Could not understand audio.")
    except sr.RequestError as exc:
        print(f"Speech recognition request failed: {exc}")
    return None


def wake_word_detected(text: str | None, wake_word: str = "ev") -> bool:
    if not text:
        return False
    return wake_word.lower() in text.lower()


def query_assistant(prompt: str, model: str = "gpt-4o-mini") -> str:
    if not openai.api_key:
        raise RuntimeError("OPENAI_API_KEY is not set in .env")

    response = openai.ChatCompletion.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are EV, a virtual assistant that can use webcam and microphone functions."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=250,
    )
    return response.choices[0].message.content.strip()


def main_loop() -> None:
    load_env()
    print("EV assistant started. Say 'EV' to wake the assistant.")

    while True:
        try:
            audio_path = record_audio(duration=4.0)
            transcript = transcribe_audio(audio_path)

            if wake_word_detected(transcript):
                print("Wake word detected.")
                frame = capture_frame()
                user_prompt = "User activated EV and wants assistance."
                if transcript:
                    user_prompt = transcript
                assistant_reply = query_assistant(user_prompt)
                print("EV:", assistant_reply)
            else:
                print("No wake word detected. Listening again...")

            time.sleep(1.0)
        except KeyboardInterrupt:
            print("Shutting down EV.")
            break
        except Exception as exc:
            print(f"Error: {exc}")
            time.sleep(2.0)


if __name__ == "__main__":
    main_loop()
