# Qwen3.8-Flash-Next Megatron-Swift 使用说明

模型：`/kwkj-k8s/llm_team/models/Qwen3.8-Flash-Next`  
仓库：`/kwkj-k8s/llm_team/cainn/megatron-swift`  
机型：H200（约 140GiB/卡）  
环境：已装好的 `.venv-swift`（直接激活使用，无需再装）

---

## 1. 激活环境

在 **GPU 计算节点** 上：

```bash
cd /kwkj-k8s/llm_team/cainn/megatron-swift
source swift_env/activate.sh
```


---

## 2. 并行参数

```text
GPU 数 N = TP × PP × DP × CP
MoE：TP × DP = EP × ETP
```


| 卡数 | TP | PP | EP | ETP | SP | CP | 峰值显存（约） |
|------|----|----|----|-----|----|----|----------------|
| **4** | 4 | 1 | 4 | 1 | true | 1 | ~85 GiB |
| **8** | 2 | 2 | 4 | 1 | true | 1 | 更宽裕 |

---

## 3. Smoke（验证能训）

### 4 卡

```bash
cd /kwkj-k8s/llm_team/cainn/megatron-swift
source swift_env/activate.sh

CUDA_VISIBLE_DEVICES=0,1,2,3 bash swift_env/megatron_sft_smoke_4gpu.sh
```

默认：`TP=4 PP=1 EP=4 ETP=1 SP=true`，本地 JSONL，LoRA，2 step。

### 8 卡

```bash
source swift_env/activate.sh
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 bash swift_env/megatron_sft_smoke.sh
```

默认：`TP=2 PP=2 EP=4`，`decoder_first_pipeline_num_layers=12`。

### Nemotron Instruction-Following（已验证）

```bash
cd /kwkj-k8s/llm_team/cainn/megatron-swift
source swift_env/activate.sh

DATA_JSON=/kwkj-k8s/llm_team/cainn/data/hf/Nemotron-SFT-Instruction-Following-Chat-v3/instruction_following_hq_5000_dedup_sharegpt.jsonl \
OUTPUT_DIR=$(pwd)/outputs/megatron_smoke_nemotron_if \
CUDA_VISIBLE_DEVICES=0,1,2,3 bash swift_env/megatron_sft_smoke_4gpu.sh
```

全量约 25 万条可用：`.../Nemotron-SFT-Instruction-Following-Chat-v3/data/instruction_following.jsonl`。

### 覆盖参数示例

```bash
MODEL=/path/to/model \
DATA_JSON=/path/to/data.jsonl \
OUTPUT_DIR=/path/to/out \
MAX_STEPS=2 MAX_LENGTH=512 GLOBAL_BATCH_SIZE=4 \
TP=4 PP=1 EP=4 SEQUENCE_PARALLEL=true \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
bash swift_env/megatron_sft_smoke.sh
```

日志：`swift_env/logs/megatron_smoke_*.log`  
输出：`outputs/`

---

## 4. 正式训练

### 8 卡

```bash
cd /kwkj-k8s/llm_team/cainn/megatron-swift
source swift_env/activate.sh

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
NPROC_PER_NODE=8 \
bash swift_env/megatron_sft.sh
```

| 变量 | 默认 | 含义 |
|------|------|------|
| `MODEL` | `/kwkj-k8s/llm_team/models/Qwen3.8-Flash-Next` | 模型路径 |
| `OUTPUT_DIR` | `outputs/megatron_sft_lora` | 输出目录 |
| `TP` / `PP` / `EP` | `2` / `2` / `4` | 并行 |
| `GBS` / `MBS` | `8` / `1` | global / micro batch |
| `MAX_LENGTH` | `2048` | 序列长度 |

### 4 卡

```bash
source swift_env/activate.sh

CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
TP=4 PP=1 EP=4 \
GBS=4 MAX_LENGTH=2048 \
bash swift_env/megatron_sft.sh
```

`megatron_sft.sh` 默认数据集是示例 `swift/self-cognition#500`，换自己的数据时改脚本里的 `--dataset`。

日志：`swift_env/logs/megatron_sft_*.log`

---

## 5. Megatron 参数对照

```text
--tensor_model_parallel_size          TP
--pipeline_model_parallel_size        PP
--expert_model_parallel_size          EP
--expert_tensor_parallel_size         1
--sequence_parallel                   true
--context_parallel_size               1
--tuner_type lora --lora_rank 8 --lora_alpha 32
--target_modules in_proj out_proj linear_proj linear_qkv
--micro_batch_size 1
--moe_grouped_gemm true --moe_permute_fusion true
--recompute_granularity full
```

PP=2（8 卡）额外：`--decoder_first_pipeline_num_layers 12`
