#!/usr/bin/env python3
"""Single-node forward-check YAML (TP8 PP1 EP8, forward only) derived from the formal_opt YAML.

Two variants: *_fwd_check_sparse.yaml (QSA_SPARSE_KERNEL=True, the training path) and
*_fwd_check_dense.yaml (QSA_SPARSE_KERNEL=0, dense fallback) to isolate the QSA approximation.
"""
import copy
import json
from pathlib import Path

import yaml

MN = Path('/kwkj-k8s/llm_team/lys/megatron-swift/multinode')
FORMAL = MN / 'tb226_flash_next_lora_4node_212k_qsa_thd_overlay_formal_opt.yaml'
PLUG = MN / 'plugins'
FWD = MN / 'smoke_data/fwd_check'
OUT = Path('/kwkj-k8s/llm_team/lys/megatron-swift/outputs')
summary = json.loads((FWD / 'summary.json').read_text())

base = yaml.safe_load(FORMAL.open())
env = base['ENV']
env.pop('DP_TOKEN_BALANCE', None)
env['PROFILE_DISABLE_SAVE'] = 1
env['FWD_TOPK'] = 100
env['FWD_SAMPLE_STRIDE'] = 64
env['FWD_SAMPLE_TAIL'] = 512
env['FWD_FULL_VOCAB_MAX_T'] = 4096
env['FWD_VOCAB_SIZE'] = 248320
env['FWD_EP'] = 8
base['dataset'] = summary['path']
base['tensor_model_parallel_size'] = 8
base['pipeline_model_parallel_size'] = 1
base.pop('decoder_first_pipeline_num_layers', None)
base['expert_model_parallel_size'] = 8
base['context_parallel_size'] = 1
base['packing'] = False
base['padding_free'] = True
base['micro_batch_size'] = 1
base['global_batch_size'] = summary['rows']
base['lora_dropout'] = 0.0
base['dataset_shuffle'] = False
base['train_dataloader_shuffle'] = False
base['data_seed'] = 42
base.pop('num_train_epochs', None)
base['train_iters'] = 1
base.pop('swanlab_project', None)
base.pop('swanlab_exp_name', None)
base['save_steps'] = 1000000
base['no_save_optim'] = True
base['no_save_rng'] = True
base['dataset_num_proc'] = 8
base['load_from_cache_file'] = False
base['external_plugins'] = [str(PLUG / 'ple_chunked.py'), str(PLUG / 'forward_logits_dump.py')]

for variant, qsa in (('sparse', 'True'), ('dense', '0')):
    cfg = copy.deepcopy(base)
    cfg['ENV']['QSA_SPARSE_KERNEL'] = qsa
    cfg['ENV']['FWD_DUMP_DIR'] = str(OUT / 'fwd_check' / f'train_{variant}')
    cfg['output_dir'] = str(OUT / 'fwd_check' / f'megatron_{variant}_smoke')
    cfg['tensorboard_dir'] = cfg['output_dir'] + '/tensorboard'
    dst = MN / f'tb226_flash_next_lora_fwd_check_{variant}.yaml'
    assert not dst.exists()
    dst.write_text(
        f'# Forward-only logits dump | 1 node | TP8 PP1 EP8 | QSA_SPARSE_KERNEL={qsa} | {summary["rows"]} fixed texts, no backward, no save.\n'
        + yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True, width=120), encoding='utf-8')
    print('wrote', dst)
