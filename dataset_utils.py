from dataclasses import dataclass
from pathlib import Path
import json


@dataclass
class DialogueExample:
    prompt: str
    completion: str


def sample_training_examples() -> list[DialogueExample]:
    return [
        DialogueExample(
            prompt="User: EV, what can you do?\nAssistant:",
            completion=" I can help you monitor your webcam, listen through your microphone, and respond to commands when you call my name.\n"
        ),
        DialogueExample(
            prompt="User: EV, recognize my voice.\nAssistant:",
            completion=" I can learn your voice profile and respond when I hear your name.\n"
        ),
        DialogueExample(
            prompt="User: EV, take a picture.\nAssistant:",
            completion=" I can capture an image from the webcam and save it for you.\n"
        ),
    ]


def export_examples_to_jsonl(examples: list[DialogueExample], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps({"prompt": example.prompt, "completion": example.completion}, ensure_ascii=False) + "\n")

    print(f"Written {len(examples)} examples to {output_path}")
