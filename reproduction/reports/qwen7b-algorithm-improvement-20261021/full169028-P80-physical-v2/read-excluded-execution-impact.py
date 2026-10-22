"""Count rejected wrong-execution candidates; never change canonical labels."""
from array import array
import hashlib,json,time
from pathlib import Path
B=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/full169028-policy80-physical-ownG10-20261023-v2')
def read(p):return json.loads(Path(p).read_bytes())
def checked(r):
 raw=Path(r['path']).read_bytes();assert len(raw)==r['bytes']and hashlib.sha256(raw).hexdigest()==r['sha256'];return raw
c=read(B/'physical-source-contract.json');audit=read(B/'physical-source-sidecar-audit.json');groups=read(B/'fixture/inputs/train-groups.json');qids=sorted(groups);lookup={t:i*10+j for i,q in enumerate(qids)for j,t in enumerate(groups[q]['trajectory_ids'])};ix=array('I');ix.frombytes(checked(c['selected_source_index']));wrong={};by_source=[];changed=set();bad_sources=[s for s in audit['sources']if s['excluded_wrong_physical_rows']]
for source in bad_sources:
 selected=c['selected_sources'];count=0;disagree=0;qs=set()
 for line in checked(source['own_em']).decode().splitlines():
  r=json.loads(line);t=r['trajectory_id']
  if t not in lookup:continue
  i=lookup[t];proper=selected[ix[i]]['full'];bad=source['full']
  if(proper['bytes'],proper['sha256'])==(bad['bytes'],bad['sha256']):continue
  q=qids[i//10];y=r['strict_em'];assert r['qid']==q and type(y)is int and y in(0,1);old=wrong.setdefault(t,y);assert old==y;count+=1;qs.add(q);true=groups[q]['labels'][i%10];disagree+=y!=true
  if y!=true:changed.add(q)
 by_source.append(dict(wrong_candidate_full=source['full'],wrong_candidate_own_em=source['own_em'],excluded_canonical_trajectories=count,excluded_questions=len(qs),label_disagreements=disagree))
affected={qids[lookup[t]//10]for t in wrong};matrix=[[0]*11 for _ in range(11)];flips=0;M_to_H=0;H_to_M=0
for q in affected:
 g=groups[q];correct=sum(g['labels']);counterfactual=sum(wrong.get(t,y)for t,y in zip(g['trajectory_ids'],g['labels']));matrix[correct][counterfactual]+=1;cm=0<correct<10;wm=0<counterfactual<10;flips+=cm!=wm;M_to_H+=cm and not wm;H_to_M+=not cm and wm
print(json.dumps(dict(time=time.time(),scope='EXCLUDED_WRONG_EXECUTION_CANDIDATES_ONLY_NOT_HISTORICAL_LABEL_ERRORS',selected_canonical_labels_verified_exact=True,canonical_labels_changed=False,wrong_source_files=len(bad_sources),excluded_wrong_execution_trajectories=len(wrong),excluded_wrong_execution_questions=len(affected),label_disagreeing_questions=len(changed),label_disagreeing_trajectories=sum(s['label_disagreements']for s in by_source),hypothetical_wrong_source_G10_Mixed_class_flips=flips,correct_Mixed_wrong_homogeneous=M_to_H,correct_homogeneous_wrong_Mixed=H_to_M,correct_K_by_wrong_source_K=matrix,by_wrong_source=by_source,final4096_read=False)))
