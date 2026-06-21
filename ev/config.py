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

LLM_MODEL = "mlx-community/Mistral-7B-Instruct-v0.3-4bit"
WHISPER_MODEL = "mlx-community/whisper-base"

FACE_RECOGNITION_TOLERANCE = 0.6
SPEAKER_VERIFICATION_THRESHOLD = 0.25

SYSTEM_PROMPT = (
    "You are EV, a helpful and friendly virtual assistant. "
    "You are speaking with your owner, Piolo. "
    "Keep responses concise and conversational since they will be spoken aloud. "
    "You can see through a webcam and hear through a microphone."
)
