# dpo-shortcuts — Does DPO learn correctness or verbosity?

**Status: working on** · course project (NLP and LLMs, Fall 2026, candidate #8) · proposal submitted 2026-10-21 · full runs due Nov

DPO 的偏好信号同时携带"正确性"和"长度/风格"两个可学习的维度。本实验把两者在数据里系统性地拆开:构造两种偏好对条件(两种条件里被偏好的都是**正确**答案):

- **SC**:chosen = 简短正确,rejected = 冗长错误
- **LC**:chosen = 冗长正确,rejected = 简短错误

然后测 held-out 上的精确正确率、生成长度、无效输出率和 10% 标签翻转敏感性——DPO 到底跟哪个走?

## 试点结果(200 对/条件,seed 0)

| 运行 | 算术 acc | 抽取 acc | 总 acc | 均长(token) | 无效率 |
|---|---|---|---|---|---|
| base (Qwen3-0.6B) | 0.030 | 1.000 | 0.515 | 25.6 | 0.000 |
| SFT-LC(同数据对照) | 0.800 | 1.000 | **0.900** | 84.4 | 0.000 |
| DPO-LC | 0.210 | 0.780 | 0.495 | **143.7** | **0.425** |
| DPO-SC | 0.070 | **0.830** | 0.450 | 21.8 | 0.000 |

**DPO-LC 学到的是长度捷径,不是正确性**:输出膨胀、42.5% 退化为复读循环、正确率反降;同数据 SFT 拿到 0.90。详见 [experiment-plan.md](experiment-plan.md) §5。

## 复现

```bash
pip install trl peft transformers datasets accelerate
export HF_ENDPOINT=https://hf-mirror.com   # 如 huggingface.co 不可达
# 1. 生成数据(两个条件 + held-out 评估集)
python code/gen_data.py --n-per-condition 250 --n-eval 200 --out-dir data/full
# 2. 训练(DPO / SFT / 翻转消融)
python code/train_dpo.py --data data/full/train_lc.jsonl --out runs/dpo_lc
python code/train_sft.py --data data/full/sft_lc.jsonl --out runs/sft_lc --epochs 6
# 3. 评估(贪心生成 + 规则判分)
python code/run_eval.py --model Qwen/Qwen3-0.6B --adapter runs/dpo_lc/adapter \
  --eval-file data/full/eval.jsonl --out runs/eval_dpo_lc.json
```

模型:Qwen3-0.6B + LoRA r=16(峰值显存 3.4GB,单卡 12GB 内即可)。设计细节、判分协议、风险与 fallback 见 [experiment-plan.md](experiment-plan.md)。

## 结构

```
code/common.py     任务/答案模板生成器(算术链、备忘录抽取)
code/gen_data.py   偏好对与评估集生成
code/train_dpo.py  DPO(TRL + LoRA,支持 --flip-frac 噪声消融)
code/train_sft.py  SFT 对照(completion-only loss)
code/run_eval.py   批量生成 + 精确判分(EM/字段EM/长度/无效率)
proposal.md        课程 proposal(问题/首实验/可行性)
experiment-plan.md 完整实验协议 + 试点结果
```

**参考**:Rafailov et al. 2023 (arXiv:2305.18290);Park et al. 2024 (arXiv:2403.19159)。
