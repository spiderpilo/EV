# EV Virtual Assistant

An initial Python project scaffold for building a virtual assistant named **EV** with webcam, microphone, and fine-tuning support.

## What this scaffold includes
- `main.py`: basic webcam capture, microphone recording, wake-word detection placeholder, and model query flow
- `fine_tune.py`: dataset preparation and example fine-tuning submission function
- `dataset_utils.py`: helper for building a JSONL training dataset
- `requirements.txt`: Python dependencies for audio/video capture and OpenAI interaction

## Setup
1. Create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Create a `.env` file with your API key:

```text
OPENAI_API_KEY=your_api_key_here
```

## Running the assistant

```bash
python main.py
```

This will open the webcam and listen for a wake word like `ev` in the recognized speech.

## Fine-tuning workflow

1. Add training examples in `dataset_utils.py` or your own dataset file.
2. Run `python fine_tune.py` to export data and submit it.
3. Update `main.py` to use your fine-tuned model once ready.

## Notes
- This scaffold provides placeholders for voice recognition and speaker recognition.
- To recognize you by voice, you will need a speaker identification model or a voice print dataset.
- For reliable wake-word detection, use a dedicated wake-word engine or training dataset rather than simple text matching.
