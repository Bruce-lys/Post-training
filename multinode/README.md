# 4 机 32 卡 207K LoRA 训练（tb226 v3）

文件：
- `tb226_flash_next_lora_4node_207k.yaml`  正式配置（TP8/PP2/EP16/DP2，关 packing 以保留 QSA，`CP=1`）
- `launch_node.sh`  每台机执行一次；`NODES` 第一个是 master
- `plugins/ple_chunked.py`  PLE 16K 分块，压 208K 下的 fp32 临时量

QSA：`packing` 与 `padding_free` 必须都是 `false`。`packing: true` 会强制 `padding_free: true`（thd），TE 无法叠自定义 mask，QSA 层会退化成稠密全注意力。

**QSA 和 CP 互斥。** `context_parallel_size > 1` 时 mcore-bridge 会打 `QSA sparse selection disabled`，QSA 层改走稠密注意力。4 机上 `CP=2` 时 DP 自动变成 1，EP16 仍满足 `TP*CP*DP`。若接受关掉 QSA 来换 207K 显存，应同时把 packing 开回去，否则短样本会 pad 满 207K。

QSA 的 `[s,s]` mask：budget=2048，序列超过 2048 才启用。TP8 + sequence parallel 下 indexer 看到的是 `s/8`，所以默认 16K 冒烟测不到 QSA（正好卡在 no-op 边界）。要测 mask 至少 `SMOKE_LEN=32768`。207K 若按全序列物化 bool mask 约 42GiB，这条路径未验证，不要一上来就 207K。

步骤：
1. 链路冒烟（16K、5 步，不测 QSA mask）：4 台机各跑 `SMOKE=1 NODES="..." bash launch_node.sh`
2. QSA mask 冒烟（32K）：`SMOKE=1 SMOKE_LEN=32768 TAG=qsa32k NODES="..." bash launch_node.sh`；日志里不能出现 `QSA sparse selection disabled`
3. 不要和 CP 混在同一次冒烟里。若测 CP：`SMOKE=1 CP=2 TAG=cp2 NODES="..."`，日志里**会**出现 QSA disabled，这是预期
4. 正式（当前 yaml，QSA / CP=1）：4 台机各跑 `NODES="..." bash launch_node.sh`
5. 看日志：`multinode/logs/<tag>/<ts>/train.node0.log`
6. 训完合并：`megatron export --adapters <output_dir>/vN-.../checkpoint-XXX --merge_lora true ...`（同 swift_env/merge_lora.sh）

回退：若 TP8 出现 kernel 问题，改 `tensor_model_parallel_size: 4`、`expert_model_parallel_size: 8`、
`pipeline_model_parallel_size: 4`、`decoder_first_pipeline_num_layers: 8`、`decoder_last_pipeline_num_layers: 10`（TP4 全链路已验证）。
