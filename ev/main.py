import re
import signal
import sys
import time

from ev.wake_word.detector import WakeWordDetector
from ev.audio.listener import AudioListener
from ev.audio.stt import SpeechToText
from ev.audio.speaker_verify import SpeakerVerifier
from ev.vision.face_recognition import FaceRecognizer
from ev.brain.llm import LLMBrain
from ev.tts.synthesizer import TextToSpeech
from ev.shell.executor import is_dangerous, run_command
from ev.shell.shortcuts import match_shortcut
from ev.search.web import is_search_query, search
from ev.projects import PROJECT_TRIGGERS, list_projects, match_project, open_project
from ev.config import LISTEN_DURATION_SEC, TERMINAL_SYSTEM_PROMPT, INTRO_TRIGGERS, INTRO_SCRIPT

CMD_PATTERN = re.compile(r"\[CMD:\s*(.+?)\]")
TERMINAL_KEYWORDS = ("terminal", "termin")
EXIT_TRIGGERS = ("exit terminal", "exit", "quit", "stop", "leave terminal")


def is_terminal_trigger(text: str) -> bool:
    words = text.strip().lower().rstrip(".").split()
    return any(w.startswith(k) for w in words for k in TERMINAL_KEYWORDS) and \
           any(w.startswith("mod") for w in words)


def handle_response(response, brain, tts, context):
    match = CMD_PATTERN.search(response)
    if not match:
        tts.speak(response)
        return

    command = match.group(1).strip()
    spoken_part = CMD_PATTERN.sub("", response).strip()

    if is_dangerous(command):
        msg = f"That command looks dangerous: {command}. I'm not going to run that."
        print(f"  [EV]: {msg}")
        tts.speak(msg)
        return

    print(f"  [SHELL] Running: {command}")
    if spoken_part:
        tts.speak(spoken_part)

    success, output = run_command(command)
    print(f"  [SHELL] Output: {output[:500]}")

    followup = brain.think(
        f"I ran the command: {command}\n\nOutput:\n{output}\n\nSummarize the result briefly for Piolo.",
        context=context,
    )
    print(f"  [EV]: {followup}")
    tts.speak(followup)


def select_project(listener, stt, tts):
    projects = list_projects()
    if not projects:
        tts.speak("I couldn't find any projects in your Repos folder.")
        return

    print(f"  [PROJECTS] Found: {', '.join(projects)}")
    tts.speak("Which project?")

    audio = listener.record(duration=LISTEN_DURATION_SEC)
    spoken = stt.transcribe(audio).strip()
    print(f"  [PROJECTS] You said: \"{spoken}\"")

    if not spoken:
        tts.speak("I didn't catch that.")
        return

    match = match_project(spoken, projects)
    if not match:
        tts.speak(f"I couldn't find a project matching {spoken}.")
        return

    cmd = open_project(match)
    print(f"  [PROJECTS] Opening: {cmd}")
    tts.speak(f"Opening {match}.")
    run_command(cmd)


def terminal_mode(listener, stt, speaker, brain, tts):
    print("\n  [TERMINAL MODE] Activated — listening for commands...")
    tts.speak("Terminal mode activated.")

    while True:
        audio = listener.record(duration=LISTEN_DURATION_SEC)

        is_owner, _ = speaker.verify(audio)
        if not is_owner:
            continue

        text = stt.transcribe(audio)
        text_lower = text.strip().lower().rstrip(".")

        if not text or text_lower in ("", ".", ".."):
            continue

        print(f"  [TERMINAL] You said: \"{text}\"")

        if text_lower in EXIT_TRIGGERS:
            print("  [TERMINAL MODE] Deactivated.")
            tts.speak("Leaving terminal mode.")
            return

        shortcut = match_shortcut(text)
        if shortcut:
            command, reply = shortcut
            print(f"  [SHORTCUT] {command}")
            is_shortcut = True
        else:
            response = brain.think(text, system_prompt=TERMINAL_SYSTEM_PROMPT)
            print(f"  [EV]: {response}")
            match = CMD_PATTERN.search(response)
            if not match:
                tts.speak(response)
                continue
            command, reply, is_shortcut = match.group(1).strip(), None, False

        if is_dangerous(command):
            msg = "That command looks dangerous. I won't run it."
            print(f"  [EV]: {msg}")
            tts.speak(msg)
            continue

        if reply:
            print(f"  [EV]: {reply}")
            tts.speak(reply)
        elif not is_shortcut:
            announce = brain.think(
                f"You are about to run this command: {command}\n"
                "Say ONE short sentence about what you're doing.",
            )
            print(f"  [EV]: {announce}")
            tts.speak(announce)

        print(f"  [SHELL] Running: {command}")
        success, output = run_command(command)
        print(f"  [SHELL] Output: {output[:500]}")

        followup = brain.think(
            f"I ran: {command}\n\nOutput:\n{output}\n\nRead back the result briefly.",
        )
        print(f"  [EV]: {followup}")
        tts.speak(followup)


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

        text_lower = text.strip().lower().rstrip(".")
        if is_terminal_trigger(text_lower):
            terminal_mode(listener, stt, speaker, brain, tts)
            wake_word.reset()
            time.sleep(1)
            continue

        if text_lower in INTRO_TRIGGERS:
            print(f"  [EV]: {INTRO_SCRIPT}")
            tts.speak(INTRO_SCRIPT)
            wake_word.reset()
            continue

        if any(text_lower.startswith(t) for t in PROJECT_TRIGGERS):
            select_project(listener, stt, tts)
            wake_word.reset()
            continue

        shortcut = match_shortcut(text)
        if shortcut:
            command, reply = shortcut
            print(f"  [SHORTCUT] {command}")
            if reply:
                print(f"  [EV]: {reply}")
                tts.speak(reply)
            run_command(command)
            wake_word.reset()
            continue

        context = {
            "face_recognized": is_owner_face,
            "voice_verified": is_owner_voice,
        }

        if is_search_query(text):
            query = text.strip().rstrip("?.!")
            encoded = query.replace(" ", "+")
            url = f"https://www.google.com/search?q={encoded}"
            print(f"  [SEARCH] Opening browser + querying DDG: {query}")
            run_command(f"google-chrome '{url}'")
            snippets = search(query)
            print(f"  [SEARCH] Snippets: {snippets[:300]}...")
            response = brain.think(
                f"Search results for '{query}':\n\n{snippets}\n\n"
                "Summarize this in 2-3 sentences for Piolo, spoken aloud.",
                context=context,
            )
            print(f"  [EV]: {response}")
            tts.speak(response)
            wake_word.reset()
            continue

        response = brain.think(text, context=context)
        print(f"  [EV]: {response}")

        handle_response(response, brain, tts, context)
        wake_word.reset()


if __name__ == "__main__":
    main()
