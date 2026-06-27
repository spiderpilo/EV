from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
EV_DATA_DIR = PROJECT_ROOT / "ev" / "data"

FACE_EMBEDDINGS_DIR = EV_DATA_DIR / "face_embeddings"
VOICE_EMBEDDINGS_DIR = EV_DATA_DIR / "voice_embeddings"

WAKE_WORD = "ev"

SAMPLE_RATE = 16000
AUDIO_CHANNELS = 1
LISTEN_DURATION_SEC = 5

LLM_MODEL = "Qwen/Qwen2.5-3B-Instruct"
LLM_ADAPTER = str(PROJECT_ROOT / "models" / "ev-finetuned")
WHISPER_MODEL = "small"

TTS_PITCH_SHIFT = 1.2  # 1.0 = original, higher = younger (try 1.1 to 1.2)

FACE_RECOGNITION_TOLERANCE = 0.6
SPEAKER_VERIFICATION_THRESHOLD = 0.25

SYSTEM_PROMPT = (
    "You are EV, a helpful and friendly virtual assistant. "
    "You are speaking with your owner, Piolo. "
    "Keep responses concise and conversational since they will be spoken aloud. "
    "You can see through a webcam and hear through a microphone."
)

TERMINAL_SYSTEM_PROMPT = (
    "You are EV, a virtual assistant with access to Piolo's Linux terminal. "
    "The user is in terminal mode. Translate their request into a single Linux shell command. "
    "Respond ONLY with [CMD: the command] and nothing else. "
    "If you can't figure out a command, say so briefly."
)
