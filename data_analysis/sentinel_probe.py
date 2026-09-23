import json, collections
F="/kwkj-k8s/llm_team/hj/terminal_bench_sft/qwen38_pass_226_swift_agent_traj.jsonl"
M="COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
pos_hist=collections.Counter(); multi=[]
tail_shapes=collections.Counter()
for k,line in enumerate(open(F)):
    d=json.loads(line); msgs=d["messages"]
    hits=[(i,m["role"]) for i,m in enumerate(msgs) if i>1 and M in (m.get("content") or "")]
    tcs=[i for i,r in hits if r=="tool_call"]
    if len(tcs)!=1: multi.append((k,d["meta"]["task_id"],hits[:8],len(msgs)))
    tail_shapes[tuple(m["role"] for m in msgs[-4:])]+=1
    # last tool_call content & its response
    if k<2 or len(tcs)!=1:
        for i in tcs:
            print(k, d["meta"]["task_id"], "tool_call@",i,"/",len(msgs), repr(msgs[i]["content"])[:160], "| resp:", repr(msgs[i+1]["content"])[:160] if i+1<len(msgs) else None)
print("tail shapes:",tail_shapes)
print("trajs with !=1 sentinel tool_call:",len(multi))
for x in multi: print(x)
