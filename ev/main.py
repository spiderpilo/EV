import signal
import sys

from ev.wake_word.detector import WakeWordDetector
from ev.audio.listener import AudioListener
from ev.audio.stt import SpeechToText
from ev.audio.speaker_verify import SpeakerVerifier
from ev.vision.face_recognition import FaceRecognizer
from ev.brain.llm import LLMBrain
from ev.tts.synthesizer import TextToSpeech
from ev.config import LISTEN_DURATION_SEC


def main():
    print("=" * 50)
    print("  EV — Virtual Assistant")
    print("=" * 50)

    print("\n[1/6] Initializing audio listener...")
    listener = AudioListener()

    print("[2/6] Loading wake word detector...")
    wake_word = WakeWordDetector()

    print("[3/6] Loading speech-to-text (Whisper)...")
    stt = SpeechToText()

    print("[4/6] Loading speaker verification...")
    speaker = SpeakerVerifier()
    if not speaker.is_enrolled:
        print("  WARNING: No voice enrolled. Run `python -m scripts.enroll_voice` first.")

    print("[5/6] Loading face recognition...")
    face = FaceRecognizer()
    if not face.is_enrolled:
        print("  WARNING: No face enrolled. Run `python -m scripts.enroll_face` first.")

    print("[6/6] Loading LLM brain...")
    brain = LLMBrain()

    tts = TextToSpeech()

    def shutdown(sig, frame):
        print("\n\nShutting down EV...")
        face.release()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)

    print("\n" + "=" * 50)
    print("  EV is ready. Say 'EV' to activate!")
    print("=" * 50 + "\n")

    greeted = False

    for chunk in listener.stream_chunks(chunk_duration=0.5):
        if not wake_word.detect(chunk):
            continue

        print("\n[EV] Wake word detected! Listening...")
        if not greeted:
            tts.speak("Hello Piolo")
            greeted = True

        audio = listener.record(duration=LISTEN_DURATION_SEC)

        is_owner_voice, voice_score = speaker.verify(audio)
        print(f"  Voice verification: {'PASS' if is_owner_voice else 'FAIL'} (score: {voice_score:.3f})")

        frame = face.capture_frame()
        is_owner_face, face_status = face.recognize(frame)
        print(f"  Face recognition: {face_status} ({'PASS' if is_owner_face else 'FAIL'})")

        if not is_owner_voice and not is_owner_face:
            print("  Identity not confirmed. Ignoring.")
            tts.speak("Sorry, I don't recognize you.")
            continue

        text = stt.transcribe(audio)
        print(f"  You said: \"{text}\"")

        if not text or text.strip() in ("", ".", ".."):
            print("  No speech detected.")
            continue

        context = {
            "face_recognized": is_owner_face,
            "voice_verified": is_owner_voice,
        }
        response = brain.think(text, context=context)
        print(f"  [EV]: {response}")

        tts.speak(response)


if __name__ == "__main__":
    main()
