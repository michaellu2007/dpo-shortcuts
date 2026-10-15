"""DPO training on preference pairs (LoRA on Qwen3-0.6B).

Prompts are rendered with the same chat template (thinking disabled) used at
eval time; completions are raw text and the trainer appens EOS (= <|im_end|>).
With a PEFT adapter, the frozen base model acts as the DPO reference policy.
"""

import argparse
import json
import random
import time
from pathlib import Path

from datasets import Dataset
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import DPOConfig, DPOTrainer

from chat import render_prompt

MODEL = "Qwen/Qwen3-0.6B"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True, help="train_{sc,lc}.jsonl")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--flip-frac", type=float, default=0.0,
                    help="fraction of pairs with chosen/rejected swapped (noise ablation)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=float, default=3.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--beta", type=float, default=0.1)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--max-steps", type=int, default=-1, help="smoke-test override")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    rows = [json.loads(l) for l in open(args.data)]
    if args.flip_frac > 0:
        n_flip = round(len(rows) * args.flip_frac)
        flip_idx = set(rng.sample(range(len(rows)), n_flip))
    else:
        flip_idx = set()

    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    data = []
    for i, r in enumerate(rows):
        chosen, rejected = r["chosen"], r["rejected"]
        if i in flip_idx:
            chosen, rejected = rejected, chosen
        data.append({
            "prompt": render_prompt(tokenizer, r["prompt"]),
            "chosen": chosen,
            "rejected": rejected,
        })
    ds = Dataset.from_list(data)

    import math

    steps_per_epoch = math.ceil(len(ds) / (args.batch_size * args.grad_accum))
    total_steps = args.max_steps if args.max_steps > 0 else math.ceil(args.epochs * steps_per_epoch)

    cfg = DPOConfig(
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
        beta=args.beta,
        max_length=512,
        report_to=[],
    )
    peft_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    t0 = time.time()
    trainer = DPOTrainer(model=MODEL, args=cfg, train_dataset=ds, peft_config=peft_cfg)
    trainer.train()
    trainer.save_model(str(args.out / "adapter"))
    tokenizer.save_pretrained(str(args.out / "adapter"))
    metrics = {k: v for k, v in trainer.state.log_history[-1].items()}
    import torch

    print(json.dumps({
        "wall_seconds": round(time.time() - t0, 1),
        "max_mem_GB": round(torch.cuda.max_memory_allocated() / 1e9, 2),
        "train_pairs": len(ds),
        "flipped": len(flip_idx),
        "last_metrics": metrics,
    }, indent=2, default=str))


if __name__ == "__main__":
    main()
