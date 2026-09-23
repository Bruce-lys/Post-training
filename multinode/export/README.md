# Export + consistency check for the tb226 LoRA run (prepared 2026-09-20)

Order (all on ONE free node with 8 H200; nothing here touches the training nodes while training runs):

1. `CKPT=... OUT=... DRY_RUN=1 bash export_lora_to_hf.sh` then `DRY_RUN=0 START_EXPORT=YES_I_CONFIRM ...`
   -> merged HF checkpoint in OUT (+ swift's test_convert_precision on a short example).
2. `python check_export.py --base /kwkj-k8s/llm_team/models/Qwen3.8-Flash-Next --export OUT --report OUT/check_export.json`
   (CPU, ~20 min for 336 GB): config diff, tensor sets, dtype/shape, frozen tensors byte-identical, LoRA delta stats.
3. `CKPT=... OUT=... STEP=mg DRY_RUN=0 START_FWD_CHECK=YES_I_CONFIRM bash fwd_check_export.sh`  (Megatron, adapter unmerged)
4. `... STEP=hf DRY_RUN=0 bash fwd_check_export.sh`      (transformers on the exported checkpoint)
5. `... STEP=sglang DRY_RUN=0 bash fwd_check_export.sh`  (sglang on the exported checkpoint)
6. `... STEP=compare bash fwd_check_export.sh`
7. Layer-2 merge equivalence (decisive, immune to the MoE routing cascade):
   `... STEP=teacher_hf DRY_RUN=0 bash fwd_check_export.sh`  (HF fp32 hidden dump of the exported model, text 0)
   `... STEP=teacher_mg DRY_RUN=0 START_FWD_CHECK=YES_I_CONFIRM bash fwd_check_export.sh`  (Megatron fp32 + adapter, teacher-forced)
   `... STEP=teacher_compare bash fwd_check_export.sh`  -> PASS if per-token update error: median over layers <= 3e-4, worst-layer median <= 1e-3, teacher-forced argmax agree == 1.0
   (base-vs-base reference 2026-09-18: ~1e-4 / ~3e-4 / 1.000)
Pass criteria: check_export PASS (no frozen tensor changed, every LoRA target changed, MTP/visual present);
mean-NLL gap train(adapter) vs sglang(export) within the base-model floor (<0.5% on the long texts); no new
systematic offset in |dnll| beyond the ~0.5 floor. Per-token argmax agreement is NOT a criterion for this model.
