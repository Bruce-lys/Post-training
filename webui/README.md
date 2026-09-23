# Megatron-SWIFT WebUI

LlamaFactory 风格的网页界面，在页面上改参数、切换 LoRA / 全量 SFT、一键启动 `megatron sft`，看日志和 loss 曲线。

```bash
cd /kwkj-k8s/llm_team/lys/megatron-swift/webui
bash scripts/setup_venv.sh          # 首次：建 UI 专用 venv（宿主机 python3.10 + gradio），不动 .venv-swift
bash scripts/export_schema.sh       # 首次或 ms-swift 升级后：导出参数表 schema/megatron_sft_args.json
bash scripts/start_ui.sh            # 后台起 UI，默认端口 17870（被占自动顺延）
bash scripts/stop_ui.sh             # 停 UI（训练进程不受影响）
```

- Train 页：数据集 → 参数分组（模型 / 数据 / 训练超参 / LoRA 或 全量 / 并行 / MoE / 保存）→ 预览命令 / 启动前检查 / 保存·加载配置 / 开始 / 中止 → 输出、曲线、日志。
- 训练记录页：历史 run 表，点一行看详情、六张曲线、日志、启动脚本；可复刻参数到 Train、中止、删除记录。
- 设置页：模型/数据/输出/训练环境路径，UI 端口与登录，假训练模式（不占 GPU，只写模拟曲线）。
- 每次训练在 `OUTPUT_ROOT/<run_id>/` 下：`train.log`、`logging.jsonl`（曲线来源）、`ui/run.sh`（可直接手动重跑）、`ui/train_config.json`、`checkpoint-*`。

验证：`.venv/bin/python -m pytest -q tests`，`.venv/bin/python scripts/headless_smoke.py validate|fake`。
