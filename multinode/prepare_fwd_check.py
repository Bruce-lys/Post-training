#!/usr/bin/env python3
"""Pick the fixed fwd_check texts (5 lengths: below QSA budget, then deeper into the sparse regime).

Writes multinode/smoke_data/fwd_check/{texts.jsonl, lengths.json, summary.json}. Rows come from the formal
dataset (uniform meta schema); meta is normalized to {source, token_length}. Read-only otherwise.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path('/kwkj-k8s/llm_team/lys/megatron-swift')
LEN = ROOT / 'data_analysis/tb226_v3_swift_tpl_lengths.jsonl'
SRC = Path('/kwkj-k8s/llm_team/lys/llm_beachmark-new/terminal-bench/sft_data/qwen38_pass_226_turn_split_v3_fit207k_sub3.jsonl')
OUT = ROOT / 'multinode/smoke_data/fwd_check'
TARGETS = (1500, 6000, 30000, 100000, 200000)
PACK_LEN = 212992

rows = [json.loads(l) for l in LEN.open()]
kept = [r for r in rows if 0 < r['tokens'] <= PACK_LEN]
picked, used = [], set()
for t in TARGETS:
    best = min((r for r in kept if r['idx'] not in used), key=lambda r: abs(r['tokens'] - t))
    used.add(best['idx']); picked.append(best)
assert picked[0]['tokens'] <= 2048 and all(p['tokens'] > 2048 for p in picked[1:])
want = {p['idx']: p for p in picked}
lines = {}
with SRC.open(encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i in want:
            d = json.loads(line)
            d['meta'] = {'source': 'formal', 'token_length': want[i]['tokens'], 'src_idx': i}
            lines[i] = json.dumps(d, ensure_ascii=False)
        if len(lines) == len(want):
            break
OUT.mkdir(parents=True, exist_ok=True)
texts = OUT / 'texts.jsonl'
assert not texts.exists()
texts.write_text(''.join(lines[p['idx']] + '\n' for p in picked), encoding='utf-8')
(OUT / 'lengths.json').write_text(json.dumps({'source': str(texts), 'rows': len(picked), 'lengths': [p['tokens'] for p in picked]}))
sha = hashlib.sha256(texts.read_bytes()).hexdigest()
summary = {
    'path': str(texts), 'rows': len(picked), 'sha256': sha,
    'docs': [{'order': k, 'idx': p['idx'], 'tokens': p['tokens'], 'task': p['task']} for k, p in enumerate(picked)],
    'note': 'row 0 is below the QSA budget (2048): full attention; rows 1-4 exercise the sparse thd kernel',
}
(OUT / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
print(json.dumps(summary, indent=1, ensure_ascii=False))
