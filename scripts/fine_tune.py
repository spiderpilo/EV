"""
Fine-tune a base model for EV using MLX and LoRA.

Usage:
    python -m scripts.fine_tune --model mistralai/Mistral-7B-Instruct-v0.3 --data data/training/

Prepare training data as JSONL files in data/training/:
    - train.jsonl
    - valid.jsonl

Each line should be a JSON object with a "messages" key:
    {"messages": [
        {"role": "system", "content": "You are EV..."},
        {"role": "user", "content": "Hello EV"},
        {"role": "assistant", "content": "Hey Piolo! What's up?"}
    ]}
"""
import argparse
import subprocess
import sys
from pathlib import Path

from ev.config import DATA_DIR


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a model for EV")
    parser.add_argument(
        "--model",
        default="mistralai/Mistral-7B-Instruct-v0.3",
        help="HuggingFace model to fine-tune",
    )
    parser.add_argument(
        "--data",
        default=str(DATA_DIR / "training"),
        help="Directory with train.jsonl and valid.jsonl",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument(
        "--output",
        default="models/ev-finetuned",
        help="Output directory for the adapter",
    )
    args = parser.parse_args()

    data_dir = Path(args.data)
    train_file = data_dir / "train.jsonl"
    valid_file = data_dir / "valid.jsonl"

    if not train_file.exists():
        print(f"ERROR: {train_file} not found.")
        print("Create training data first. See data/training/example.jsonl for format.")
        sys.exit(1)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "mlx_lm.lora",
        "--model", args.model,
        "--train",
        "--data", str(data_dir),
        "--adapter-path", str(output_dir),
        "--iters", str(args.epochs * 100),
        "--batch-size", str(args.batch_size),
        "--learning-rate", str(args.learning_rate),
        "--lora-layers", "16",
        "--lora-rank", str(args.lora_rank),
    ]

    print("=" * 50)
    print("  EV — Fine-Tuning")
    print("=" * 50)
    print(f"\n  Model:    {args.model}")
    print(f"  Data:     {args.data}")
    print(f"  Output:   {args.output}")
    print(f"  Epochs:   {args.epochs}")
    print(f"  LoRA rank: {args.lora_rank}")
    print()

    subprocess.run(cmd, check=True)

    print(f"\nFine-tuning complete! Adapter saved to: {output_dir}")
    print(f"\nTo use the fine-tuned model, update LLM_MODEL in ev/config.py:")
    print(f"  LLM_MODEL = \"{output_dir}\"")


if __name__ == "__main__":
    main()
