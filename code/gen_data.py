"""Generate preference-pair training data and the held-out eval set.

Conditions (chosen is ALWAYS the correct answer):
  SC  short-correct : chosen = short correct, rejected = long wrong
  LC  long-correct  : chosen = long correct, rejected = short wrong

Each pair carries one long and one short response, so total tokens per pair
are matched across conditions by construction.

Outputs (JSONL):
  train_sc.jsonl / train_lc.jsonl : prompt, chosen, rejected, gold, task
  sft_sc.jsonl  / sft_lc.jsonl    : prompt, completion (= chosen only)
  eval.jsonl                      : prompt, gold, task (fresh instances)
"""

import argparse
import json
import random
import re
from pathlib import Path

from common import DATES, PROJECTS, gen_arith, gen_extract, short_correct


def make_pairs(n, seed, dedupe):
    """Build n raw task instances split evenly across the two task types."""
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        gen = gen_arith if len(out) % 2 == 0 else gen_extract
        prompt, gold, long_correct, long_wrong = gen(rng)
        key = prompt.split("\n\n", 1)[-1]
        if key in dedupe:
            continue
        dedupe.add(key)
        gold_repr = (
            str(gold) if isinstance(gold, int)
            else json.dumps(gold, separators=(", ", ": "))
        )
        out.append({
            "prompt": prompt,
            "gold": gold,
            "task": "arith" if gen is gen_arith else "extract",
            "short_correct": short_correct(gold_repr),
            "long_correct": long_correct(),
            "short_wrong": short_correct(str(_wrong_repr(gold, rng))),
            "long_wrong": long_wrong(),
        })
    return out[:n]


def _wrong_repr(gold, rng):
    """A wrong *short* answer payload (used by the LC condition)."""
    if isinstance(gold, int):
        deltas = [-11, -2, -1, 1, 2, 10]
        return str(gold + rng.choice([d for d in deltas if gold + d != gold]))
    wrong = dict(gold)
    field = rng.choice(list(wrong))
    if field == "amount":
        wrong["amount"] = int(wrong["amount"]) + rng.choice([-1000, -100, 100, 1000])
    elif field == "attendees":
        wrong["attendees"] = int(wrong["attendees"]) + rng.choice([-1, 1, 2])
    elif field == "date":
        day = int(re.search(r"\d+", wrong["date"]).group())
        wrong["date"] = re.sub(r"\d+", str(day + rng.choice([-2, -1, 1, 2])), wrong["date"])
    else:
        wrong["project"] = rng.choice([p for p in PROJECTS if p != wrong["project"]])
    return json.dumps(wrong, separators=(", ", ": "))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-per-condition", type=int, default=250,
                    help="pairs per condition (pilot: 100)")
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--seed", type=int, default=17)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    dedupe = set()
    train = make_pairs(2 * args.n_per_condition, args.seed, dedupe)
    evalset = make_pairs(args.n_eval, args.seed + 1000, dedupe)

    for cond, short_is_chosen in [("sc", True), ("lc", False)]:
        with open(args.out_dir / f"train_{cond}.jsonl", "w") as f:
            for ex in train:
                if short_is_chosen:
                    chosen, rejected = ex["short_correct"], ex["long_wrong"]
                else:
                    chosen, rejected = ex["long_correct"], ex["short_wrong"]
                f.write(json.dumps({
                    "prompt": ex["prompt"], "chosen": chosen,
                    "rejected": rejected, "gold": ex["gold"], "task": ex["task"],
                }) + "\n")
        with open(args.out_dir / f"sft_{cond}.jsonl", "w") as f:
            for ex in train:
                completion = ex["short_correct"] if short_is_chosen else ex["long_correct"]
                f.write(json.dumps({
                    "prompt": ex["prompt"], "completion": completion,
                }) + "\n")

    with open(args.out_dir / "eval.jsonl", "w") as f:
        for ex in evalset:
            f.write(json.dumps({
                "prompt": ex["prompt"], "gold": ex["gold"], "task": ex["task"],
            }) + "\n")

    n_long = sum(len(ex["long_correct"]) + len(ex["long_wrong"]) for ex in train)
    n_short = sum(len(ex["short_correct"]) + len(ex["short_wrong"]) for ex in train)
    print(f"wrote {len(train)} pairs/condition, {len(evalset)} eval prompts to {args.out_dir}")
    print(f"avg chars per pair: long+short = {(n_long + n_short) / len(train):.0f}")


if __name__ == "__main__":
    main()
