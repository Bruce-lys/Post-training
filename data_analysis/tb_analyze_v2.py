import json, collections, statistics as st
rows=[json.loads(l) for l in open("/kwkj-k8s/llm_team/lys/megatron-swift/data_analysis/tb226_v2_lengths.jsonl")]
OVH=300
for r in rows: r["len"]=r["total"]+OVH
L=sorted(r["len"] for r in rows)
n=len(L)
def pct(p): return L[min(n-1,int(p*n))]
print("samples",n,"trajs",len({r['trial'] for r in rows}),"tasks",len({r['task'] for r in rows}))
print("len tokens: mean %.0f median %.0f p10 %.0f p25 %.0f p75 %.0f p90 %.0f p95 %.0f p99 %.0f max %.0f min %.0f"%(st.mean(L),pct(.5),pct(.1),pct(.25),pct(.75),pct(.9),pct(.95),pct(.99),L[-1],L[0]))
tot=sum(L); print("total tokens all samples: %.3fB"%(tot/1e9))
T=[r["tgt"] for r in rows]; print("target(trained) tokens: mean %.0f median %.0f max %.0f sum %.1fM"%(st.mean(T),st.median(T),max(T),sum(T)/1e6))
print("\nthreshold  kept_samples  kept%  kept_tokens(B)  dropped_target_tokens(M)")
for th in [65536,98304,131072,163840,196608,229376,262144,294912,327680]:
    k=[r for r in rows if r["len"]<=th]
    print("%7d  %6d  %5.1f%%  %6.3f  %8.2f"%(th,len(k),100*len(k)/n,sum(r['len'] for r in k)/1e9,sum(r['tgt'] for r in rows if r['len']>th)/1e6))
print("\nhistogram (32K bins):")
bins=collections.Counter(r["len"]//32768 for r in rows)
for b in sorted(bins): print("  %3dK-%3dK  %6d  %s"%(b*32,(b+1)*32,bins[b],"#"*int(bins[b]/n*200)))
print("\nper-task: samples / trajs / turns-per-traj(avg) / final-sample-len (full traj) mean,max / sample-len mean")
bt=collections.defaultdict(list)
for r in rows: bt[r["task"]].append(r)
for t,rs in sorted(bt.items(), key=lambda x:-len(x[1])):
    trajs=collections.defaultdict(list)
    for r in rs: trajs[r["trial"]].append(r)
    finals=[max(x["len"] for x in v) for v in trajs.values()]
    turns=[len(v) for v in trajs.values()]
    print("  %-28s %6d %4d %6.0f %8.0f %8.0f %8.0f"%(t,len(rs),len(trajs),st.mean(turns),st.mean(finals),max(finals),st.mean(r['len'] for r in rs)))
# per traj full length distribution
trajs=collections.defaultdict(list)
for r in rows: trajs[r["trial"]].append(r)
F=sorted(max(x["len"] for x in v) for v in trajs.values())
print("\nfull-trajectory length (last sample): mean %.0f median %.0f p90 %.0f max %.0f; >128K: %d, >160K: %d, >192K: %d, >256K: %d of %d"%(st.mean(F),F[len(F)//2],F[int(.9*len(F))],F[-1],sum(f>131072 for f in F),sum(f>163840 for f in F),sum(f>196608 for f in F),sum(f>262144 for f in F),len(F)))
# think presence in target? and empty content
# --- v1 (hj format) vs v2 comparison + submit-turn view
old={}
for l in open("/kwkj-k8s/llm_team/lys/megatron-swift/data_analysis/tb226_lengths.jsonl"):
    r=json.loads(l); old[(r["trial"],r["turn"])]=r["total"]+OVH
common=[(old[(r["trial"],r["turn"])], r["len"]) for r in rows if (r["trial"],r["turn"]) in old]
print("\nv1 vs v2 on the %d shared samples: mean len %.0f -> %.0f (+%.1f%%), total %.3fB -> %.3fB"%(len(common),st.mean(a for a,b in common),st.mean(b for a,b in common),100*(st.mean(b for a,b in common)/st.mean(a for a,b in common)-1),sum(a for a,b in common)/1e9,sum(b for a,b in common)/1e9))
sub=[r for r in rows if r["turn"]==r["steps"] or (r["trial"],r["turn"]) not in old]
# submit turns = last turn of each traj
last={}
for r in rows:
    if r["trial"] not in last or r["turn"]>last[r["trial"]]["turn"]: last[r["trial"]]=r
S=sorted(v["len"] for v in last.values())
print("submit-turn samples (%d): mean %.0f median %.0f max %.0f; kept at 131072: %d, 163840: %d, 196608: %d, 229376: %d"%(len(S),st.mean(S),S[len(S)//2],S[-1],sum(x<=131072 for x in S),sum(x<=163840 for x in S),sum(x<=196608 for x in S),sum(x<=229376 for x in S)))
