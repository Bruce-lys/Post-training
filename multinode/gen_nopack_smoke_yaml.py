#!/usr/bin/env python3
"""Derive the no-packing EP8 + PP24/24 + microbatch-ordering smoke YAML from the formal_opt YAML."""
import copy
from pathlib import Path

import yaml

MN = Path('/kwkj-k8s/llm_team/lys/megatron-swift/multinode')
FORMAL = MN / 'tb226_flash_next_lora_4node_212k_qsa_thd_overlay_formal_opt.yaml'
PLUG = MN / 'plugins'
SMOKE = MN / 'smoke_data/packing_smoke'
OUT = Path('/kwkj-k8s/llm_team/lys/megatron-swift/outputs')

cfg = yaml.safe_load(FORMAL.open())
assert cfg['packing'] is False and cfg['padding_free'] is True
env = cfg['ENV']
env.pop('DP_TOKEN_BALANCE', None)
env['DP_BALANCE_ORDER'] = 'True'
env['DP_BALANCE_LENGTHS_FILE'] = str(SMOKE / 'subset_624rows.lengths.json')
env['PROFILE_DISABLE_SAVE'] = 1
env['PERF_STEP_LABELS'] = 'W1,W2,W3,S4,S5,S6,S7,S8'
cfg['dataset'] = str(SMOKE / 'subset_624rows.jsonl')
cfg['output_dir'] = str(OUT / 'qsa4node_nopack_ep8_pp24_smoke')
cfg['tensorboard_dir'] = cfg['output_dir'] + '/tensorboard'
cfg['expert_model_parallel_size'] = 8
cfg['decoder_first_pipeline_num_layers'] = 24
cfg.pop('num_train_epochs', None)
cfg['train_iters'] = 8
cfg['data_seed'] = 42
cfg.pop('swanlab_project', None)
cfg.pop('swanlab_exp_name', None)
cfg['save_steps'] = 1000000
cfg['save_total_limit'] = 2
cfg['no_save_optim'] = True
cfg['no_save_rng'] = True
cfg['dataset_num_proc'] = 8
cfg['load_from_cache_file'] = False
cfg['external_plugins'] = [
    str(PLUG / 'ple_chunked.py'),
    str(PLUG / 'dp_balance_order.py'),
    str(PLUG / 'mem_probe.py'),
    str(PLUG / 'iteration_timing.py'),
]
dst = MN / 'tb226_flash_next_lora_4node_qsa_nopack_ep8_pp24_smoke.yaml'
assert not dst.exists()
dst.write_text(
    '# no-packing smoke | 4 nodes | TP8 PP2 (24/24) CP1 EP8 | padding_free single-doc microbatches\n'
    '# + dp_balance_order.py (DP token balance + [short, long, ..., short] microbatch order from a lengths cache)\n'
    '# subset_624rows, 8 steps, shuffle on, mem_probe + iteration_timing on, no save. Derived from formal_opt YAML.\n'
    + yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True, width=120), encoding='utf-8')
print('wrote', dst)
