import json, statistics as st
rows=json.load(open("/kwkj-k8s/llm_team/lys/megatron-swift/data_analysis/tb226_think_cost.json"))
# rows: (task, turn_idx, len_no_hist_think, len_with_hist_think), trajectories contiguous
trajs=[]; cur=None
for task,k,a,b in rows:
    if k==0: cur=[]; trajs.append(cur)
    cur.append((a,b))
print("trajs",len(trajs))
# --- packing efficiency simulation (first-fit-decreasing, like ms-swift binpacking) for turn-split @163840
L=163840
A=sorted([a for t in trajs for a,b in t if a<=L], reverse=True)
bins=[]
for x in A:
    for i,c in enumerate(bins):
        if c+x<=L: bins[i]=c+x; break
    else: bins.append(x)
print("turn-split @%d: %d samples -> %d packs, utilization %.1f%%, tokens %.3fB"%(L,len(A),len(bins),100*sum(A)/(len(bins)*L),sum(A)/1e9))
# --- full-trajectory (one sample per traj, loss on all turns) with think preserved
full=[t[-1][1] for t in trajs]          # last turn incl. its own think = whole traj with think
full_no=[t[-1][0] for t in trajs]       # whole traj, history think stripped (not usable for all-turn loss, for reference)
fs=sorted(full)
print("full traj WITH think: mean %.0f median %.0f p90 %.0f max %.0f sum %.1fM"%(st.mean(fs),fs[len(fs)//2],fs[int(.9*len(fs))],fs[-1],sum(fs)/1e6))
for th in (131072,163840,196608,229376,262144):
    k=[x for x in fs if x<=th]; print("  <=%d: %d/226 trajs kept, %.1fM tokens"%(th,len(k),sum(k)/1e6))
# windowed: split each traj into consecutive windows of <=L tokens by turn boundaries (system+user re-prepended), each window loss on all its turns
def windows(t, L, keep_prefix):
    # cumulative with-think lengths per turn: t[k][1] = prefix(with think)+target; increments:
    inc=[t[0][1]]+[t[k][1]-t[k-1][1] for k in range(1,len(t))]
    out=[]; cur=keep_prefix; n=0
    for x in inc[1:] if False else inc:
        if cur+x>L and n>0: out.append(cur); cur=keep_prefix; n=0
        cur+=x; n+=1
    out.append(cur); return out
tot=0; nwin=0
for t in trajs:
    w=windows(t,163840,3000); tot+=sum(w); nwin+=len(w)
print("windowed @163840 (all-turn loss, think kept): %d windows, %.1fM tokens"%(nwin,tot/1e6))
