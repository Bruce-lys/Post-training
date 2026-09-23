#!/usr/bin/env python3
"""Build row-ordered length caches for dp_balance_order.py.

  data_analysis/tb226_v3_lengths_by_row.json            formal dataset (28518 rows)
  multinode/smoke_data/packing_smoke/subset_624rows.lengths.json   smoke subset (624 rows)
Lengths are swift-template token counts from tb226_v3_swift_tpl_lengths.jsonl; rows that failed
to encode (tokens=-1) are stored as 0. Writes only these two new files.
"""
import json
from pathlib import Path

ROOT = Path('/kwkj-k8s/llm_team/lys/megatron-swift')
LEN = ROOT / 'data_analysis/tb226_v3_swift_tpl_lengths.jsonl'
SMOKE = ROOT / 'multinode/smoke_data/packing_smoke'

rows = [json.loads(l) for l in LEN.open()]
by_idx = {r['idx']: max(int(r['tokens']), 0) for r in rows}
assert len(by_idx) == 28518 and sorted(by_idx) == list(range(28518))

formal = ROOT / 'data_analysis/tb226_v3_lengths_by_row.json'
assert not formal.exists()
formal.write_text(json.dumps({
    'source': '/kwkj-k8s/llm_team/lys/llm_beachmark-new/terminal-bench/sft_data/qwen38_pass_226_turn_split_v3_fit207k_sub3.jsonl',
    'lengths_from': str(LEN), 'rows': 28518,
    'lengths': [by_idx[i] for i in range(28518)],
}), encoding='utf-8')
print('wrote', formal)

summary = json.loads((SMOKE / 'summary.json').read_text())
subset_lines = (SMOKE / 'subset_624rows.jsonl').read_text(encoding='utf-8').splitlines()
assert len(subset_lines) == 624
top24 = [json.loads(l)['meta']['token_length'] for l in subset_lines[:24]]
rand = [by_idx[i] for i in summary['subset']['random_idx']]
lengths = top24 + rand
assert len(lengths) == 624
out = SMOKE / 'subset_624rows.lengths.json'
assert not out.exists()
out.write_text(json.dumps({'source': str(SMOKE / 'subset_624rows.jsonl'), 'rows': 624, 'lengths': lengths}), encoding='utf-8')
print('wrote', out, 'min/max', min(lengths), max(lengths))
