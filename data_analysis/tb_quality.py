import json, collections, hashlib, re
F="/kwkj-k8s/llm_team/hj/terminal_bench_sft/qwen38_pass_226_turn_split_strip_hist.jsonl"
n=0; tgt_think=0; tgt_no_tc=0; tgt_empty=0; hist_think=0; last_role=collections.Counter()
dup=collections.Counter(); sys_set=set(); user_set=collections.Counter()
long_tool_resp=0; max_tool_resp_chars=0; complete_mark=0; tc_bad_json=0; tool_names=collections.Counter()
final_turn_len=[]; tr_lens=[]
for line in open(F):
    d=json.loads(line); n+=1; msgs=d["messages"]
    sys_set.add(msgs[0]["content"]); user_set[msgs[1]["content"][:200]]+=1
    last_a=max(i for i,m in enumerate(msgs) if m["role"]=="assistant")
    tgt=msgs[last_a:]; hist=msgs[:last_a]
    c=tgt[0].get("content") or ""
    if "<think>" in c: tgt_think+=1
    if not c.strip(): tgt_empty+=1
    if not any(m["role"]=="tool_call" for m in tgt): tgt_no_tc+=1
    if any("<think>" in (m.get("content") or "") for m in hist if m["role"]=="assistant"): hist_think+=1
    last_role[msgs[-1]["role"]]+=1
    if "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT" in line: complete_mark+=1
    for m in msgs:
        if m["role"]=="tool_response":
            l=len(m.get("content") or ""); tr_lens.append(l); max_tool_resp_chars=max(max_tool_resp_chars,l)
        if m["role"]=="tool_call":
            try: tool_names[json.loads(m["content"])["name"]]+=1
            except Exception: tc_bad_json+=1
    dup[hashlib.md5(json.dumps(msgs,ensure_ascii=False).encode()).hexdigest()]+=1
tr_lens.sort()
print("samples",n)
print("target assistant has <think>: %d (%.1f%%)  empty target content: %d  target without tool_call (final answer turn): %d"%(tgt_think,100*tgt_think/n,tgt_empty,tgt_no_tc))
print("history assistants still containing <think>: %d samples"%hist_think)
print("last message role:",dict(last_role))
print("COMPLETE_TASK sentinel remaining:",complete_mark," tool_call bad json:",tc_bad_json," tool names:",dict(tool_names))
print("exact duplicate samples:",sum(v-1 for v in dup.values() if v>1))
print("distinct system prompts:",len(sys_set)," distinct task prompts(first200):",len(user_set))
print("tool_response chars: median %d p99 %d max %d; >100K chars: %d"%(tr_lens[len(tr_lens)//2],tr_lens[int(.99*len(tr_lens))],max_tool_resp_chars,sum(1 for x in tr_lens if x>100000)))
