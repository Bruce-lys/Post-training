import json, sys
from swift import get_processor, get_template
MODEL="/kwkj-k8s/llm_team/models/Qwen3.8-Flash-Next"
proc = get_processor(MODEL); tok = getattr(proc, "tokenizer", proc)
tpl = get_template(proc, template_type="qwen3_8", agent_template="qwen3_5", loss_scale="last_round", truncation_strategy="raise", max_length=10_000_000)
tpl.set_mode("train")
F="/kwkj-k8s/llm_team/hj/terminal_bench_sft/qwen38_pass_226_turn_split_strip_hist.jsonl"
want={0,1,2,50,200}
def approx(d):
    msgs=d["messages"]; s="".join("<|im_start|>%s\n%s<|im_end|>\n" % (m["role"], m.get("content") or "") for m in msgs)+(d.get("tools") or "")
    return len(tok(s, add_special_tokens=False)["input_ids"])
with open(F) as f:
    for i,line in enumerate(f):
        if i>max(want): break
        if i not in want: continue
        d=json.loads(line)
        enc=tpl.encode(d)
        ids=enc["input_ids"]; lab=enc["labels"]
        n_lab=sum(1 for x in lab if x!=-100)
        print("sample %d: template_len=%d approx_len=%d trained_tokens=%d n_msgs=%d" % (i, len(ids), approx(d), n_lab, len(d["messages"])))
        if i==0:
            print("--- trained region:", repr(tok.decode([t for t,l in zip(ids,lab) if l!=-100]))[:1500])
            print("--- head:", repr(tok.decode(ids[:150])))
