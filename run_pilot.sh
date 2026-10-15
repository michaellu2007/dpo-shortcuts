#!/bin/bash
# Pilot: 200 pairs/condition, 3 seeds omitted (seed 0 only), full eval grid.
set -e
export HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
cd "$(dirname "$0")/code"
D=../data/pilot
R=../runs/pilot
mkdir -p $R

t0=$(date +%s)
stamp() { echo "[$(( $(date +%s) - t0 ))s] $1"; }

stamp "baseline eval"
python3 run_eval.py --model Qwen/Qwen3-0.6B --eval-file $D/eval.jsonl \
  --out $R/eval_base.json --batch-size 32 > $R/log_base.txt 2>&1

for COND in sc lc; do
  stamp "DPO-$COND train"
  python3 train_dpo.py --data $D/train_$COND.jsonl \
    --out $R/dpo_$COND --seed 0 > $R/log_train_dpo_$COND.txt 2>&1
  stamp "DPO-$COND eval"
  python3 run_eval.py --model Qwen/Qwen3-0.6B --adapter $R/dpo_$COND/adapter \
    --eval-file $D/eval.jsonl --out $R/eval_dpo_$COND.json \
    --batch-size 32 > $R/log_eval_dpo_$COND.txt 2>&1

  stamp "SFT-$COND train (6 epochs ≈ DPO token exposure)"
  python3 train_sft.py --data $D/sft_$COND.jsonl \
    --out $R/sft_$COND --seed 0 --epochs 6 > $R/log_train_sft_$COND.txt 2>&1
  stamp "SFT-$COND eval"
  python3 run_eval.py --model Qwen/Qwen3-0.6B --adapter $R/sft_$COND/adapter \
    --eval-file $D/eval.jsonl --out $R/eval_sft_$COND.json \
    --batch-size 32 > $R/log_eval_sft_$COND.txt 2>&1

  stamp "DPO-$COND flip-10% train"
  python3 train_dpo.py --data $D/train_$COND.jsonl --out $R/dpo_${COND}_flip \
    --seed 0 --flip-frac 0.1 > $R/log_train_dpo_${COND}_flip.txt 2>&1
  stamp "DPO-$COND flip-10% eval"
  python3 run_eval.py --model Qwen/Qwen3-0.6B --adapter $R/dpo_${COND}_flip/adapter \
    --eval-file $D/eval.jsonl --out $R/eval_dpo_${COND}_flip.json \
    --batch-size 32 > $R/log_eval_dpo_${COND}_flip.txt 2>&1
done

stamp "pilot complete"
