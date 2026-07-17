"""
Fine-tune a base model for EV using QLoRA.

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
import sys
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from ev.config import DATA_DIR


def main():
    parser = argparse.ArgumentParser(description="Fine-tune a model for EV")
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-Coder-3B-Instruct",
        help="HuggingFace model to fine-tune",
    )
    parser.add_argument(
        "--data",
        default=str(DATA_DIR / "training"),
        help="Directory with train.jsonl and valid.jsonl",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument(
        "--output",
        default="models/ev-coder-finetuned",
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

    effective_batch = args.batch_size * args.grad_accum
    print("=" * 50)
    print("  EV — Fine-Tuning (QLoRA)")
    print("=" * 50)
    print(f"\n  Model:           {args.model}")
    print(f"  Data:            {args.data}")
    print(f"  Output:          {args.output}")
    print(f"  Epochs:          {args.epochs}")
    print(f"  Batch size:      {args.batch_size} x {args.grad_accum} grad accum = {effective_batch} effective")
    print(f"  Learning rate:   {args.learning_rate}")
    print(f"  LoRA rank:       {args.lora_rank}")
    print(f"  GPU:             {torch.cuda.get_device_name(0)}")
    print(f"  VRAM:            {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print()

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model in 4-bit (QLoRA)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="auto",
        torch_dtype=torch.float16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        ),
        attn_implementation="sdpa",
    )
    model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_rank * 2,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        task_type="CAUSAL_LM",
    )

    print("Loading dataset...")
    data_files = {"train": str(train_file)}
    if valid_file.exists():
        data_files["validation"] = str(valid_file)
    dataset = load_dataset("json", data_files=data_files)

    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_steps=5,
        bf16=True,
        logging_steps=5,
        eval_strategy="epoch" if valid_file.exists() else "no",
        save_strategy="epoch",
        load_best_model_at_end=valid_file.exists(),
        report_to="none",
        optim="paged_adamw_8bit",
        gradient_checkpointing=True,
        max_grad_norm=0.3,
    )

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        peft_config=lora_config,
        args=training_args,
    )

    print("Starting training...\n")
    trainer.train()
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    print(f"\nFine-tuning complete! Adapter saved to: {output_dir}")
    print(f"\nTo use the fine-tuned model, update LLM_ADAPTER in ev/config.py:")
    print(f'  LLM_ADAPTER = "{output_dir}"')


if __name__ == "__main__":
    main()
