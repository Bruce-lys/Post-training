#!/usr/bin/env python3
"""Derive the single-doc-pack A/B YAMLs from the existing A/B YAMLs (adds iteration_timing to skip the final save)."""
import yaml
from pathlib import Path
MN = Path('/kwkj-k8s/llm_team/lys/megatron-swift/multinode')
OUT = Path('/kwkj-k8s/llm_team/lys/megatron-swift/outputs')
DATA = MN / 'smoke_data/packing_smoke/ab_8docs_single.jsonl'
PLUG = str(MN / 'plugins/iteration_timing.py')
for src, dst, out_name, gbs in [
    ('tb226_flash_next_lora_4node_qsa_packing_ab_unpacked.yaml', 'tb226_flash_next_lora_4node_qsa_packing_ab_single_unpacked.yaml', 'qsa4node_packing_ab_single_unpacked_smoke', 8),
    ('tb226_flash_next_lora_4node_qsa_packing_ab_packed.yaml', 'tb226_flash_next_lora_4node_qsa_packing_ab_single_packed.yaml', 'qsa4node_packing_ab_single_packed_smoke', 8),
]:
    cfg = yaml.safe_load((MN / src).read_text(encoding='utf-8'))
    cfg['dataset'] = str(DATA)
    cfg['output_dir'] = str(OUT / out_name); cfg['tensorboard_dir'] = str(OUT / out_name / 'tensorboard')
    cfg['global_batch_size'] = gbs
    if PLUG not in cfg['external_plugins']:
        cfg['external_plugins'].append(PLUG)
    cfg['ENV']['PERF_STEP_LABELS'] = 'S1'
    p = MN / dst; assert not p.exists()
    p.write_text(f'# A/B single-doc packs: 8 docs each >106,496 tok -> packing yields 8 one-doc packs; GBS 8, 1 step, dropout 0, no final save.\n' + yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True, width=120), encoding='utf-8')
    print('wrote', p)
