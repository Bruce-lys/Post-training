#!/usr/bin/env python3
"""Build the packing smoke datasets (subset + A/B 8-doc set) with a manifest.

Outputs (multinode/smoke_data/packing_smoke/):
  subset_624rows.jsonl   24 longest rows (qsa_top24_212k.jsonl) + 600 random trainable rows
  ab_8docs.jsonl         4 big (>106,496 tok) + 4 small docs; any two bigs exceed 212,992,
                         so any packer yields exactly 4 packs; each doc > 2048 tok (QSA non-trivial)
  summary.json           row ids, template token lengths, FFD pack estimate, sha256 of outputs
Read-only w.r.t. every existing file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

ROOT = Path('/kwkj-k8s/llm_team/lys/megatron-swift')
LENGTHS = ROOT / 'data_analysis/tb226_v3_swift_tpl_lengths.jsonl'
SOURCE = Path('/kwkj-k8s/llm_team/lys/llm_beachmark-new/terminal-bench/sft_data/'
              'qwen38_pass_226_turn_split_v3_fit207k_sub3.jsonl')
TOP24 = ROOT / 'multinode/smoke_data/qsa_top24_212k.jsonl'
OUT_DIR = ROOT / 'multinode/smoke_data/packing_smoke'
PACK_LEN = 212992
BIG_TARGETS = (150_000, 130_000, 120_000, 110_000)
SMALL_TARGETS = (55_000, 75_000, 85_000, 95_000)
PAIR_CAP = 205_000
MIN_DOC = 2048


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def ffd_packs(lengths, cap):
    caps = []
    for t in sorted(lengths, reverse=True):
        for i, c in enumerate(caps):
            if c >= t:
                caps[i] -= t
                break
        else:
            caps.append(cap - t)
    return len(caps)


def nearest(pool, target, used):
    best = min((r for r in pool if r['idx'] not in used), key=lambda r: abs(r['tokens'] - target))
    used.add(best['idx'])
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=20260917)
    ap.add_argument('--n-random', type=int, default=600)
    args = ap.parse_args()

    rows = [json.loads(l) for l in LENGTHS.open()]
    kept = [r for r in rows if 0 < r['tokens'] <= PACK_LEN]
    assert len(rows) == 28518, len(rows)

    # ---- A/B docs: 4 big + 4 small ----
    used = set()
    bigs = [nearest([r for r in kept if r['tokens'] > PACK_LEN // 2], t, used) for t in BIG_TARGETS]
    smalls = []
    for big, t in zip(bigs, SMALL_TARGETS):
        pool = [r for r in kept if MIN_DOC < r['tokens'] < PACK_LEN // 2 and big['tokens'] + r['tokens'] <= PAIR_CAP]
        smalls.append(nearest(pool, t, used))
    ab = bigs + smalls
    ab_tokens = [r['tokens'] for r in ab]
    assert all(t > MIN_DOC for t in ab_tokens)
    assert min(b['tokens'] for b in bigs) * 2 > PACK_LEN, 'two bigs must not fit one pack'
    assert sum(ab_tokens) > 3 * PACK_LEN, 'total must force >= 4 packs'
    assert ffd_packs(ab_tokens, PACK_LEN) == 4, ffd_packs(ab_tokens, PACK_LEN)

    # ---- random subset ----
    rng = random.Random(args.seed)
    pool = [r for r in kept if r['idx'] not in used]
    rand = rng.sample(pool, args.n_random)
    want = {r['idx']: r for r in rand + ab}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    picked = {}
    with SOURCE.open(encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i in want:
                picked[i] = line.rstrip('\n')
            if len(picked) == len(want):
                break
    assert len(picked) == len(want), (len(picked), len(want))

    top24 = [l.rstrip('\n') for l in TOP24.open(encoding='utf-8') if l.strip()]
    assert len(top24) == 24, len(top24)

    subset_path = OUT_DIR / 'subset_624rows.jsonl'
    with subset_path.open('w', encoding='utf-8') as f:
        for l in top24:
            f.write(l + '\n')
        for r in rand:
            f.write(picked[r['idx']] + '\n')

    ab_path = OUT_DIR / 'ab_8docs.jsonl'
    with ab_path.open('w', encoding='utf-8') as f:
        for r in ab:
            f.write(picked[r['idx']] + '\n')

    subset_tokens = [json.loads(l)['meta'].get('token_length', PACK_LEN) for l in top24] + [r['tokens'] for r in rand]
    summary = {
        'seed': args.seed,
        'source': str(SOURCE),
        'lengths_file': str(LENGTHS),
        'pack_len': PACK_LEN,
        'subset': {
            'path': str(subset_path), 'rows': 24 + len(rand),
            'top24_rows': 24, 'random_rows': len(rand),
            'random_idx': [r['idx'] for r in rand],
            'total_tokens': sum(subset_tokens),
            'ffd_packs_estimate': ffd_packs(subset_tokens, PACK_LEN),
            'sha256': sha256(subset_path),
        },
        'ab': {
            'path': str(ab_path), 'rows': 8,
            'docs': [{'idx': r['idx'], 'tokens': r['tokens'], 'role': 'big' if r in bigs else 'small', 'task': r['task']} for r in ab],
            'pairs': [[b['idx'], s['idx'], b['tokens'] + s['tokens']] for b, s in zip(bigs, smalls)],
            'total_tokens': sum(ab_tokens),
            'ffd_packs': 4,
            'unpacked_gbs': 8, 'packed_gbs': 4,
            'sha256': sha256(ab_path),
        },
    }
    (OUT_DIR / 'summary.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({k: (v if k != 'subset' else {kk: vv for kk, vv in v.items() if kk != 'random_idx'}) for k, v in summary.items()}, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
