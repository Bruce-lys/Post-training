import json, statistics as st, collections
rows=json.load(open("/kwkj-k8s/llm_team/lys/megatron-swift/data_analysis/tb226_think_cost.json"))
lens={}
for l in open("/kwkj-k8s/llm_team/lys/megatron-swift/data_analysis/tb226_v2_lengths.jsonl"):
    r=json.loads(l); lens.setdefault(r["task"],[]).append(r)
trajs=[]; cur=None
for task,k,a,b in rows:
    if k==0: cur={"task":task,"A":[],"B":[]}; trajs.append(cur)
    cur["A"].append(a); cur["B"].append(b)
# target(with own think) tokens per turn from lengths file, matched by order within task
tgt_by_task=collections.defaultdict(list)
for task,rs in lens.items():
    # group by trial in file order
    bytrial=collections.OrderedDict()
    for r in rs: bytrial.setdefault(r["trial"],[]).append(r["tgt"])
    tgt_by_task[task]=list(bytrial.values())
ti=collections.Counter()
for t in trajs:
    t["tgt"]=tgt_by_task[t["task"]][ti[t["task"]]]; ti[t["task"]]+=1
    assert len(t["tgt"])==len(t["B"]), (len(t["tgt"]),len(t["B"]))
BASE=2500  # system + task prompt + tools header (approx)
L=163840
def window(t, ctx_len):
    B=t["B"]; steps=[B[0]-BASE]+[B[k]-B[k-1] for k in range(1,len(B))]
    wins=[]; i=0; n=len(steps)
    while i<n:
        # context: previous steps, newest first, up to ctx_len
        ctx=[]; c=0; j=i-1
        while j>=0 and c+steps[j]<=ctx_len: c+=steps[j]; ctx.insert(0,j); j-=1
        tot=BASE+c; new=[]
        while i<n and tot+steps[i]<=L: tot+=steps[i]; new.append(i); i+=1
        if not new:  # step alone too big with this context -> shrink context
            if ctx: ctx=ctx[1:]; c=sum(steps[j] for j in ctx); tot=BASE+c
            while i<n and tot+steps[i]<=L: tot+=steps[i]; new.append(i); i+=1
            if not new: tot+=steps[i]; new.append(i); i+=1  # oversize single step (will be dropped by max_length)
        wins.append(dict(tokens=tot, ctx_tokens=c, new=new, ctx=ctx, loss=sum(t["tgt"][k] for k in new), oversize=tot>L))
    return wins
print("trajs",len(trajs),"turns",sum(len(t["B"]) for t in trajs),"loss tokens all turns %.1fM"%(sum(sum(t["tgt"]) for t in trajs)/1e6))
for ctx in (0,32768,65536,98304):
    W=[w for t in trajs for w in window(t,ctx)]
    tot=sum(w["tokens"] for w in W); loss=sum(w["loss"] for w in W); ctxt=sum(w["ctx_tokens"] for w in W)
    trunc=sum(len(w["new"]) for w in W if w["ctx"] or w is not None and w["new"][0]!=0)
    first=sum(1 for w in W if w["new"][0]==0)
    over=sum(1 for w in W if w["oversize"])
    lens_=[w["tokens"] for w in W]
    print("ctx=%6d: windows %4d (first-window %d, oversize %d) total %.1fM tok, loss %.1fM, context(no-loss) %.1fM, win-len mean %.0f median %.0f min %.0f; turns with truncated history %d/%d"%(ctx,len(W),first,over,tot/1e6,loss/1e6,ctxt/1e6,st.mean(lens_),st.median(lens_),min(lens_),sum(len(w["new"]) for w in W if w["new"][0]!=0),sum(len(t["B"]) for t in trajs)))
# turn-split reference @163840
A=[a for t in trajs for a in t["A"] if a<=L]; tg=[t["tgt"][k] for t in trajs for k,a in enumerate(t["A"]) if a<=L]
print("turn-split @163840: samples %d total %.3fB loss %.1fM"%(len(A),sum(A)/1e9,sum(tg)/1e6))
# attention cost proxy: sum L^2 per pack.  turn-split: FFD packs
srt=sorted(A,reverse=True); bins=[]; packs=[]
for x in srt:
    for i,c in enumerate(bins):
        if c+x<=L: bins[i]=c+x; packs[i].append(x); break
    else: bins.append(x); packs.append([x])
ts_l2=sum(x*x for x in A); print("turn-split packs %d, sum l^2 = %.3e, per pack %.3e"%(len(packs),ts_l2,ts_l2/len(packs)))
for ctx in (65536,):
    W=[w for t in trajs for w in window(t,ctx)]
    wl=[w["tokens"] for w in W]; print("window ctx=%d: sum l^2 = %.3e, per window %.3e (ratio to turn-split pack %.2f)"%(ctx,sum(x*x for x in wl),sum(x*x for x in wl)/len(wl),(sum(x*x for x in wl)/len(wl))/(ts_l2/len(packs))))
