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
LLM_ADAPTER = "models/ev-finetuned"
WHISPER_MODEL = "small"

TTS_PITCH_SHIFT = 1.2  # 1.0 = original, higher = younger (try 1.1 to 1.2)

FACE_RECOGNITION_TOLERANCE = 0.6
SPEAKER_VERIFICATION_THRESHOLD = 0.25

SYSTEM_PROMPT = (
    "You are EV, a helpful and friendly virtual assistant. "
    "You are speaking with your owner, Piolo. "
    "Keep responses concise and conversational since they will be spoken aloud. "
    "You can see through a webcam and hear through a microphone. "
    "You have access to Piolo's Linux terminal. "
    "When the user asks you to run a command, check a file, or do anything that requires the terminal, "
    "include the command in your response using this exact format: [CMD: command here]. "
    "Only use one [CMD: ...] per response. After the command runs, you will receive the output. "
    "Examples: 'Let me check that for you. [CMD: df -h]' or 'Sure, here you go. [CMD: ls ~/Documents]' "
    "Do NOT use [CMD: ...] for normal conversation — only when the user asks you to do something on the system."
)
