import subprocess
import platform


class TextToSpeech:
    def __init__(self):
        self._system = platform.system()
        self._engine = None
        if self._system != "Darwin":
            try:
                import pyttsx3
                self._engine = pyttsx3.init()
            except Exception:
                pass

    def speak(self, text: str):
        if self._system == "Darwin":
            subprocess.run(["say", text], check=True)
        elif self._engine:
            try:
                self._engine.say(text)
                self._engine.runAndWait()
            except Exception as e:
                print(f"  TTS error: {e}")
                print(f"  EV says: {text}")
        else:
            print(f"  EV says: {text}")
