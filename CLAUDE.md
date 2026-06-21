# EV — Virtual Assistant

## Overview
EV is a fully local virtual assistant running on Apple Silicon. It uses face recognition, voice recognition, wake word detection, and a locally fine-tuned LLM.

## Architecture
- `ev/main.py` — Main event loop and orchestrator
- `ev/config.py` — Central configuration
- `ev/wake_word/` — Wake word detection ("EV")
- `ev/audio/` — Microphone capture, STT (Whisper), speaker verification
- `ev/vision/` — Face recognition via webcam
- `ev/brain/` — LLM inference via MLX
- `ev/tts/` — Text-to-speech output
- `scripts/` — Enrollment (face/voice) and fine-tuning utilities

## Commands
- Run: `python -m ev.main`
- Enroll face: `python -m scripts.enroll_face`
- Enroll voice: `python -m scripts.enroll_voice`
- Fine-tune: `python -m scripts.fine_tune`
- Install deps: `pip install -e ".[fine-tune]"`

## Tech Stack
- Python 3.10+
- MLX / mlx-lm for LLM inference and fine-tuning
- mlx-whisper for speech-to-text
- openwakeword for wake word detection
- face_recognition (dlib) for face recognition
- speechbrain for speaker verification
- sounddevice for audio capture
- pyttsx3 for text-to-speech
