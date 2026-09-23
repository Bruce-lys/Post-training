"""Per-turn sample length with vs without historical <think> (from the v2 single-traj file)."""
import json, re, statistics as st
from multiprocessing import Pool
F="/kwkj-k8s/llm_team/lys/llm_beachmark-new/terminal-bench/sft_data/qwen38_pass_226_swift_agent_traj_v2.jsonl"
MODEL="/kwkj-k8s/llm_team/models/Qwen3.8-Flash-Next"
THINK=re.compile(r"<think>.*?</think>\s*", re.DOTALL)
tok=None
def init():
    global tok
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
def n(s): return len(tok(s, add_special_tokens=False)["input_ids"])
def work(line):
    d=json.loads(line); msgs=d["messages"]; OVH=300+n(d["tools"])
    per=[]  # (role, tokens_full, tokens_nothink)
    for m in msgs:
        c=m.get("content") or ""; full=n("<|im_start|>%s\n%s<|im_end|>\n"%(m["role"],c))
        if m["role"]=="assistant" and "<think>" in c:
            per.append((m["role"], full, n("<|im_start|>assistant\n%s<|im_end|>\n"%THINK.sub("",c).lstrip("\n"))))
        else: per.append((m["role"], full, full))
    ai=[i for i,(r,_,_) in enumerate(per) if r=="assistant"]
    out=[]
    for k,a in enumerate(ai):
        end=a+1
        while end<len(per) and per[end][0]=="tool_call": end+=1
        hist_full=sum(x[1] for x in per[:a]); hist_no=sum(x[2] for x in per[:a]); tgt=sum(x[1] for x in per[a:end])
        out.append((d["meta"]["task_id"], k, OVH+hist_no+tgt, OVH+hist_full+tgt))
    return out
if __name__=="__main__":
    with open(F) as f, Pool(48, initializer=init) as p:
        rows=[r for chunk in p.imap(work, f) for r in chunk]
    A=[r[2] for r in rows]; B=[r[3] for r in rows]
    print("samples",len(rows))
    print("no-hist-think : mean %.0f median %.0f p90 %.0f max %.0f total %.2fB"%(st.mean(A),st.median(A),sorted(A)[int(.9*len(A))],max(A),sum(A)/1e9))
    print("with-hist-think: mean %.0f median %.0f p90 %.0f max %.0f total %.2fB"%(st.mean(B),st.median(B),sorted(B)[int(.9*len(B))],max(B),sum(B)/1e9))
    for th in (131072,163840,196608,229376,262144):
        print("  <=%d kept: no-think %5.1f%%  with-think %5.1f%%"%(th,100*sum(x<=th for x in A)/len(A),100*sum(x<=th for x in B)/len(B)))
    last={}
    for r in rows: last[(r[0],r[1])]=r
    import collections
    fin=collections.defaultdict(lambda:(0,0))
    print("full-traj (last turn) with think: mean %.0f max %.0f"%(st.mean(r[3] for r in rows if r[1]==max(x[1] for x in rows if x[0]==r[0]) ) if False else 0,0))
    json.dump(rows, open("/kwkj-k8s/llm_team/lys/megatron-swift/data_analysis/tb226_think_cost.json","w"))
