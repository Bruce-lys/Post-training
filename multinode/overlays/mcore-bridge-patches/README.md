# mcore-bridge overlay patches

本目录留档的是训练实际使用的 mcore-bridge 代码相对上游的改动。
节点上的工作副本位于 `multinode/overlays/mcore-bridge-9d610ffb/`，
它本身是一个独立 git 仓库，因此不纳入本仓库追踪，改动以 patch 形式保存在这里。

## 基线

| 项 | 值 |
|---|---|
| 上游仓库 | https://github.com/modelscope/mcore-bridge.git |
| 基线 commit | `9d610ff` — support glm5-next cp (#198)，即 origin/main |
| 本地分支名 | `qsa-bitmap-chunked` |
| 本地独有提交 | 2 个（见下方 patch 文件），均不存在于上游任何分支 |
| 改动范围 | 仅 `megatron/.../model/modules/kernels/qsa_block_sparse_attn.py`，+13 −7 |

## 改动说明

- `1f2bda0` — Fix QSA bitmap offsets for long contexts
- `8417fcf` — QSA: build selection block bitmap in row chunks

`8417fcf` 是经 bit-exact 等价性验证的改动：在随机输入上（CPU 与 GPU）与原实现逐位一致，
仅把 selection block bitmap 的构建改为按行分块，在 205k token 下减少约 12 GiB 瞬时显存。
它不改变前向数学、不改变求和顺序，因此不影响训推一致。

## 复现步骤

```bash
git clone https://github.com/modelscope/mcore-bridge.git
cd mcore-bridge
git checkout 9d610ff
git checkout -b qsa-bitmap-chunked
git am /path/to/mcore-bridge-patches/*.patch
```

`git am` 全部应用成功后，工作区代码与训练时所用的完全一致。
