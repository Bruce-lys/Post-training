# Megatron-SWIFT 训练环境

Qwen3.8-Flash-Next（`qwen4_exp`）Megatron LoRA SFT。  
本地模型：`/kwkj-k8s/llm_team/models/Qwen3.8-Flash-Next`  
环境：已装好的 `.venv-swift`（直接激活使用）

使用说明见：[`swift_env/QWEN38_FLASH_NEXT_PARALLEL.md`](swift_env/QWEN38_FLASH_NEXT_PARALLEL.md)

## 目录

```text
megatron-swift/
  .venv-swift/                    # 已配置好的 Python 环境
  swift_env/
    activate.sh                   # source 激活
    check_gpu.sh                  # GPU / torch 预检
    megatron_sft_smoke.sh         # smoke（按卡数选并行）
    megatron_sft_smoke_4gpu.sh    # 4 卡 smoke 包装
    megatron_sft.sh               # 正式 LoRA SFT
    QWEN38_FLASH_NEXT_PARALLEL.md # 并行与运行说明
    data/smoke_alpaca.jsonl       # 默认 smoke 小数据
  outputs/                        # 训练输出
```

## 快速开始

```bash
cd /kwkj-k8s/llm_team/cainn/megatron-swift
source swift_env/activate.sh

# 4 卡 smoke + Nemotron 数据
DATA_JSON=/kwkj-k8s/llm_team/cainn/data/hf/Nemotron-SFT-Instruction-Following-Chat-v3/instruction_following_hq_5000_dedup_sharegpt.jsonl \
OUTPUT_DIR=$(pwd)/outputs/megatron_smoke_nemotron_if \
CUDA_VISIBLE_DEVICES=0,1,2,3 bash swift_env/megatron_sft_smoke_4gpu.sh
```

并行默认：4 卡 `TP=4 PP=1 EP=4`；8 卡 `TP=2 PP=2 EP=4`。细节见上述说明文档。
