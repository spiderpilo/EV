from ev.audio.listener import AudioListener
from ev.audio.speaker_verify import SpeakerVerifier


def main():
    print("=" * 50)
    print("  EV — Voice Enrollment")
    print("=" * 50)
    print("\nThis will record 3 voice samples to learn your voice.")
    print("Speak naturally for a few seconds each time.\n")

    listener = AudioListener()
    samples = []
    target = 3

    for i in range(target):
        input(f"Press ENTER to record sample {i + 1}/{target}...")
        audio = listener.record(duration=4.0)
        samples.append(audio)
        print(f"  Sample {i + 1} recorded.\n")

    print("Enrolling voice...")
    verifier = SpeakerVerifier()
    verifier.enroll(samples)
    print("Voice enrollment complete!")


if __name__ == "__main__":
    main()
