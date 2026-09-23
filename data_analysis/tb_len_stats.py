import json, sys, os, collections
from multiprocessing import Pool
F = "/kwkj-k8s/llm_team/hj/terminal_bench_sft/qwen38_pass_226_turn_split_strip_hist.jsonl"
OUT = "/kwkj-k8s/llm_team/lys/megatron-swift/data_analysis/tb226_lengths.jsonl"
MODEL = "/kwkj-k8s/llm_team/models/Qwen3.8-Flash-Next"
tok = None
def init():
    global tok
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
def count(text):
    return len(tok(text, add_special_tokens=False)["input_ids"])
def work(args):
    idx, line = args
    d = json.loads(line)
    msgs = d["messages"]; meta = d.get("meta", {})
    # find last assistant idx (target start)
    last_a = max(i for i, m in enumerate(msgs) if m["role"] == "assistant")
    ctx_txt = "".join(f"<|im_start|>{m['role']}\n{m.get('content') or ''}<|im_end|>\n" for m in msgs[:last_a])
    tgt_txt = "".join(f"<|im_start|>{m['role']}\n{m.get('content') or ''}<|im_end|>\n" for m in msgs[last_a:])
    tools_n = count(d.get("tools") or "")
    ctx_n = count(ctx_txt); tgt_n = count(tgt_txt)
    n_asst_hist = sum(1 for m in msgs[:last_a] if m["role"] == "assistant")
    return dict(idx=idx, task=meta.get("task_id"), trial=meta.get("trial_dir"), src_index=meta.get("src_index"),
                turn=n_asst_hist + 1, steps=meta.get("steps"), n_msgs=len(msgs),
                tools=tools_n, ctx=ctx_n, tgt=tgt_n, total=tools_n + ctx_n + tgt_n)
if __name__ == "__main__":
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    nproc = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    with open(F) as f, Pool(nproc, initializer=init) as p, open(OUT, "w") as out:
        for i, r in enumerate(p.imap(work, enumerate(f), chunksize=8)):
            out.write(json.dumps(r) + "\n")
            if i % 2000 == 0: print("done", i, flush=True)
    print("DONE", flush=True)
