#!/usr/bin/env python3
"""Second A/B set: 8 docs each > 212992/2 tokens, so packing yields 8 single-doc packs.
Isolates 'packing data path' from 'multi-doc math'. Adds `ab_single` to summary.json."""
import hashlib, json, random
from pathlib import Path

ROOT = Path('/kwkj-k8s/llm_team/lys/megatron-swift')
LENGTHS = ROOT / 'data_analysis/tb226_v3_swift_tpl_lengths.jsonl'
SOURCE = Path('/kwkj-k8s/llm_team/lys/llm_beachmark-new/terminal-bench/sft_data/qwen38_pass_226_turn_split_v3_fit207k_sub3.jsonl')
OUT_DIR = ROOT / 'multinode/smoke_data/packing_smoke'
PACK_LEN = 212992
TARGETS = (120_000, 130_000, 140_000, 150_000, 160_000, 170_000, 180_000, 190_000)

def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()

rows = [json.loads(l) for l in LENGTHS.open()]
summary = json.loads((OUT_DIR / 'summary.json').read_text())
used = {d['idx'] for d in summary['ab']['docs']} | set(summary['subset']['random_idx'])
pool = [r for r in rows if PACK_LEN // 2 < r['tokens'] <= PACK_LEN and r['idx'] not in used]
docs = []
for t in TARGETS:
    best = min(pool, key=lambda r: abs(r['tokens'] - t)); pool.remove(best); docs.append(best)
assert min(d['tokens'] for d in docs) * 2 > PACK_LEN
want = {d['idx'] for d in docs}; picked = {}
with SOURCE.open(encoding='utf-8') as f:
    for i, line in enumerate(f):
        if i in want:
            picked[i] = line.rstrip('\n')
        if len(picked) == len(want):
            break
out = OUT_DIR / 'ab_8docs_single.jsonl'
assert not out.exists()
out.write_text(''.join(picked[d['idx']] + '\n' for d in docs), encoding='utf-8')
summary['ab_single'] = {
    'path': str(out), 'rows': 8,
    'docs': [{'idx': d['idx'], 'tokens': d['tokens'], 'task': d['task']} for d in docs],
    'total_tokens': sum(d['tokens'] for d in docs), 'ffd_packs': 8,
    'unpacked_gbs': 8, 'packed_gbs': 8, 'sha256': sha256(out),
}
(OUT_DIR / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
print(json.dumps(summary['ab_single'], indent=1))
