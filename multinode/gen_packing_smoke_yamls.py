#!/usr/bin/env python3
"""Derive the three packing-smoke YAMLs from the formal_opt YAML (never edits it)."""
import copy
import json
from pathlib import Path

import yaml

MN = Path('/kwkj-k8s/llm_team/lys/megatron-swift/multinode')
FORMAL = MN / 'tb226_flash_next_lora_4node_212k_qsa_thd_overlay_formal_opt.yaml'
PLUG = MN / 'plugins'
SMOKE_DATA = MN / 'smoke_data/packing_smoke'
OUT_ROOT = Path('/kwkj-k8s/llm_team/lys/megatron-swift/outputs')
summary = json.loads((SMOKE_DATA / 'summary.json').read_text())

base = yaml.safe_load(FORMAL.open())
assert base['packing'] is False and base['padding_free'] is True
assert base['decoder_first_pipeline_num_layers'] == 22


def common(cfg, dataset, out_name, train_iters):
    env = cfg['ENV']
    env.pop('DP_TOKEN_BALANCE', None)
    env['PROFILE_DISABLE_SAVE'] = 1
    cfg['dataset'] = dataset
    cfg['output_dir'] = str(OUT_ROOT / out_name)
    cfg['tensorboard_dir'] = str(OUT_ROOT / out_name / 'tensorboard')
    cfg['packing'] = True
    cfg['packing_length'] = 212992
    cfg['padding_free'] = True
    cfg.pop('num_train_epochs', None)
    cfg['train_iters'] = train_iters
    cfg['data_seed'] = 42
    cfg.pop('swanlab_project', None)
    cfg.pop('swanlab_exp_name', None)
    cfg['save_steps'] = 1000000
    cfg['save_total_limit'] = 2
    cfg['no_save_optim'] = True
    cfg['no_save_rng'] = True
    cfg['dataset_num_proc'] = 8
    cfg['packing_num_proc'] = 16
    cfg['load_from_cache_file'] = False
    cfg['external_plugins'] = [
        str(PLUG / 'packed_ple_varlen.py'),
        str(PLUG / 'ple_chunked.py'),
        str(PLUG / 'mem_probe.py'),
    ]
    return cfg


def dump(cfg, path, header):
    assert not path.exists(), f'refusing to overwrite {path}'
    text = header + yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True, width=120)
    path.write_text(text, encoding='utf-8')
    print('wrote', path)


# 1) main smoke: 624-row subset, 8 steps, shuffle on
main = common(copy.deepcopy(base), summary['subset']['path'], 'qsa4node_packing_smoke', 8)
main['dataset_shuffle'] = True
dump(
    main,
    MN / 'tb226_flash_next_lora_4node_qsa_packing_smoke.yaml',
    '# packing=true smoke | 4 nodes | TP8 PP2 CP1 EP16 | overlay QSA thd multi-doc path\n'
    '# subset_624rows (24 longest + 600 random), 8 steps, shuffle on, mem_probe on, no save.\n'
    '# Derived from tb226_flash_next_lora_4node_212k_qsa_thd_overlay_formal_opt.yaml; does not start formal training.\n',
)


# 2/3) A/B equivalence: same 8 docs, dropout 0, fixed order
def ab(cfg, packing, gbs, out_name):
    cfg = common(cfg, summary['ab']['path'], out_name, 1)
    cfg['packing'] = packing
    if not packing:
        cfg.pop('packing_length', None)
        cfg.pop('packing_num_proc', None)
    cfg['global_batch_size'] = gbs
    cfg['lora_dropout'] = 0.0
    cfg['dataset_shuffle'] = False
    cfg['train_dataloader_shuffle'] = False
    return cfg


dump(
    ab(copy.deepcopy(base), False, 8, 'qsa4node_packing_ab_unpacked_smoke'),
    MN / 'tb226_flash_next_lora_4node_qsa_packing_ab_unpacked.yaml',
    '# A/B run A: 8 docs, padding_free single-doc microbatches (packing=false), GBS 8, 1 step, dropout 0.\n',
)
dump(
    ab(copy.deepcopy(base), True, 4, 'qsa4node_packing_ab_packed_smoke'),
    MN / 'tb226_flash_next_lora_4node_qsa_packing_ab_packed.yaml',
    '# A/B run B: same 8 docs packed into 4 packs (packing=true), GBS 4, 1 step, dropout 0.\n',
)
