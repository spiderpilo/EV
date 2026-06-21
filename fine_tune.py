import json
import os
from pathlib import Path

import openai
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent


def load_env() -> None:
    env_path = BASE_DIR / ".env"
    load_dotenv(dotenv_path=env_path)
    openai.api_key = os.getenv("OPENAI_API_KEY")


def export_jsonl(dataset: list[dict], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Exported training dataset to {output_path}")


def submit_fine_tune(training_file: Path, model: str = "gpt-4o-mini") -> None:
    load_env()
    print("Uploading training file...")
    upload = openai.File.create(file=open(training_file, "rb"), purpose="fine-tune")
    print("Training file uploaded:", upload.id)

    print("Creating fine-tune job...")
    job = openai.FineTune.create(training_file=upload.id, model=model)
    print("Fine-tune job started:", job.id)
    print("Monitor job status with OpenAI dashboard or API.")


def build_ev_training_data() -> list[dict]:
    return [
        {
            "prompt": "User: Hi EV, who are you?\nAssistant:",
            "completion": "EV is your personal virtual assistant designed to help you with webcam and microphone tasks.\n"
        },
        {
            "prompt": "User: EV, show me the camera feed.\nAssistant:",
            "completion": "Sure, I can open the webcam and show the current view.\n"
        },
        {
            "prompt": "User: EV, listen to me.\nAssistant:",
            "completion": "I am listening. Please say your command after the wake word.\n"
        },
    ]


def main() -> None:
    dataset = build_ev_training_data()
    output_jsonl = BASE_DIR / "ev_training_data.jsonl"
    export_jsonl(dataset, output_jsonl)

    # Uncomment the following line when you are ready to submit training data.
    # submit_fine_tune(output_jsonl)


if __name__ == "__main__":
    main()
