"""Token length of every sample under the exact swift template used for training."""
import json, sys, os, inspect
from multiprocessing import Pool
F = "/kwkj-k8s/llm_team/lys/llm_beachmark-new/terminal-bench/sft_data/qwen38_pass_226_turn_split_v3_fit207k_sub3.jsonl"
OUT = "/kwkj-k8s/llm_team/lys/megatron-swift/data_analysis/tb226_v3_swift_tpl_lengths.jsonl"
MODEL = "/kwkj-k8s/llm_team/models/Qwen3.8-Flash-Next"
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 0
tpl = None
def init():
    global tpl
    from swift.model import get_model_processor
    from swift.template import get_template
    _, proc = get_model_processor(MODEL, load_model=False)
    tpl = get_template(proc, template_type='qwen3_8', agent_template='qwen3_5', loss_scale='last_round',
                       truncation_strategy='raise', max_length=None, enable_thinking=True,
                       add_non_thinking_prefix=False, template_backend='swift')
    tpl.set_mode('train')
def work(a):
    i, line = a
    d = json.loads(line); m = d['meta']
    try:
        enc = tpl.encode({'messages': d['messages'], 'tools': d.get('tools')}, return_length=True)
        n = enc['length'] if 'length' in enc else len(enc['input_ids'])
    except Exception as e:
        n = -1
    return dict(idx=i, task=m.get('task_id'), submit=bool(m.get('is_submit_turn')), final=bool(m.get('is_final_turn')),
                turn=m.get('turn_idx'), n_turns=m.get('n_turns_total'), tokens=n)
if __name__ == '__main__':
    rows = []
    with open(F) as f:
        for i, l in enumerate(f):
            if LIMIT and i >= LIMIT: break
            rows.append((i, l))
    with Pool(48, initializer=init) as p, open(OUT if not LIMIT else '/dev/stdout', 'w') as out:
        for k, r in enumerate(p.imap(work, rows, chunksize=4)):
            out.write(json.dumps(r) + '\n')
            if k % 2000 == 0: print('done', k, file=sys.stderr, flush=True)
    print('DONE', file=sys.stderr)
