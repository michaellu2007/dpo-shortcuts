"""SFT control: train on the chosen responses only (completion-only loss).

This is the "did the pairs matter, or just the preferred answers?" control.
Use --epochs ~2x the DPO epochs so total supervised tokens roughly match the
chosen+rejected exposure of DPO; the exact token accounting is printed.
"""

import argparse
import json
import time
from pathlib import Path

from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import SFTConfig, SFTTrainer

from chat import render_prompt

MODEL = "Qwen/Qwen3-0.6B"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True, help="sft_{sc,lc}.jsonl")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=float, default=6.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=-1)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    rows = [json.loads(l) for l in open(args.data)]
    data = [{
        "prompt": render_prompt(tokenizer, r["prompt"]),
        "completion": r["completion"],
    } for r in rows]
    ds = Dataset.from_list(data)

    import math

    steps_per_epoch = math.ceil(len(ds) / (args.batch_size * args.grad_accum))
    total_steps = args.max_steps if args.max_steps > 0 else math.ceil(args.epochs * steps_per_epoch)

    cfg = SFTConfig(
        output_dir=str(args.out),
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=max(1, int(0.1 * total_steps)),
        logging_steps=10,
        save_strategy="no",
        bf16=True,
        seed=args.seed,
        max_length=512,
        completion_only_loss=True,
        report_to=[],
    )
    peft_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    t0 = time.time()
    trainer = SFTTrainer(model=MODEL, args=cfg, train_dataset=ds, peft_config=peft_cfg)
    trainer.train()
    trainer.save_model(str(args.out / "adapter"))
    tokenizer.save_pretrained(str(args.out / "adapter"))
    print(json.dumps({
        "wall_seconds": round(time.time() - t0, 1),
        "train_examples": len(ds),
    }, indent=2))


if __name__ == "__main__":
    main()
