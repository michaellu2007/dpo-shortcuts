"""Greedy generation + scoring of one checkpoint on the held-out eval set."""

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from chat import render_prompt, score_prediction


@torch.no_grad()
def generate(model, tokenizer, prompts, batch_size, max_new_tokens):
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    outs = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True).to(model.device)
        gen = model.generate(
            **enc,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
            pad_token_id=tokenizer.pad_token_id,
        )
        new = gen[:, enc["input_ids"].shape[1]:]
        outs.extend(tokenizer.batch_decode(new, skip_special_tokens=True))
        print(f"  generated {min(i + batch_size, len(prompts))}/{len(prompts)}", flush=True)
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--adapter", default=None, help="PEFT adapter dir (merged on load)")
    ap.add_argument("--eval-file", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    args = ap.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda"
    )
    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
    model.eval()

    rows = [json.loads(l) for l in open(args.eval_file)]
    rendered = [render_prompt(tokenizer, r["prompt"]) for r in rows]
    texts = generate(model, tokenizer, rendered, args.batch_size, args.max_new_tokens)

    records, n_tok = [], 0
    for r, t in zip(rows, texts):
        correct, invalid, pred = score_prediction(r["task"], r["gold"], t)
        n_tok += len(tokenizer(t)["input_ids"])
        records.append({
            "task": r["task"], "correct": correct, "invalid": invalid,
            "gen_chars": len(t), "pred": pred, "raw": t,
        })

    def acc(rs):
        vals = [x["correct"] for x in rs]
        ok = sum(1 for v in vals if v is True)
        return ok / len(vals)

    summary = {
        "model": args.model,
        "adapter": args.adapter,
        "n": len(rows),
        "acc_arith": acc([r for r in records if r["task"] == "arith"]),
        "acc_extract": acc([r for r in records if r["task"] == "extract"]),
        "acc_total": acc(records),
        "invalid_rate": sum(r["invalid"] for r in records) / len(records),
        "mean_gen_tokens": n_tok / len(records),
        "mean_gen_chars": sum(r["gen_chars"] for r in records) / len(records),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "records": records}, indent=1))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
