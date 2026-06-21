import subprocess
import platform


class TextToSpeech:
    def __init__(self):
        self._system = platform.system()

    def speak(self, text: str):
        if self._system == "Darwin":
            subprocess.run(["say", text], check=True)
        else:
            try:
                import pyttsx3
                engine = pyttsx3.init()
                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                print(f"  TTS error: {e}")
                print(f"  EV says: {text}")
