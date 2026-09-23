#!/usr/bin/env python3
"""nsys variant of the no-packing smoke YAML: adds plugins/nsys_iteration_capture.py and its own output dir."""
from pathlib import Path

import yaml

MN = Path('/kwkj-k8s/llm_team/lys/megatron-swift/multinode')
src = MN / 'tb226_flash_next_lora_4node_qsa_nopack_ep8_pp24_smoke.yaml'
dst = MN / 'tb226_flash_next_lora_4node_qsa_nopack_ep8_pp24_nsys_smoke.yaml'
assert not dst.exists()
txt = src.read_text(encoding='utf-8')
header = ''.join(l for l in txt.splitlines(True) if l.startswith('#'))
cfg = yaml.safe_load(txt)
cfg['output_dir'] = '/kwkj-k8s/llm_team/lys/megatron-swift/outputs/qsa4node_nopack_ep8_pp24_nsys_smoke'
cfg['tensorboard_dir'] = cfg['output_dir'] + '/tensorboard'
plug = str(MN / 'plugins/nsys_iteration_capture.py')
if plug not in cfg['external_plugins']:
    cfg['external_plugins'].append(plug)
dst.write_text(
    header
    + '# + nsys_iteration_capture.py: NVTX FULL_ITERATION_N ranges and a cudaProfilerApi gate for one captured step.\n'
    + yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True, width=120),
    encoding='utf-8',
)
print('wrote', dst)
