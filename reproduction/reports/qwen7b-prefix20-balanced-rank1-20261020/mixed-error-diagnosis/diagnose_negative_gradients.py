import json,gzip,math,time
from pathlib import Path
B=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/balanced-rank1-from-epoch1-prefix20-20261020')
c=json.loads((B/'config.json').read_text());rec=json.loads((B/'probe/epoch-0002.json').read_text());truth={r['qid']:r for r in json.loads((Path(c['selection_base'])/'bank/groups.json').read_text())['selection_validation']}
rs=[]
with gzip.open(rec['metrics']['selection2048']['predictions']['path'],'rt') as f:
 for line in f:rs.append(json.loads(line))
out={}
for k,name in [(0,'allwrong'),(5,'allcorrect')]:
 sub=[r for r in rs if sum(truth[r['qid']]['labels'])==k];wrongtokens=wronggroups=allwronggroups=opposinggroup=0;gcur=[];gnew=[]
 for r in sub:
  p=r['p_j'];A=C=1.
  for v in p:A*=v;C*=1-v
  w=A/(A+C);y=k/5
  grp=[v-w for v in p];cur=[(v-y)/5+g for v,g in zip(p,grp)];new=[1.2*(v-y) for v in p]
  iswrong=[v>0 if k==5 else v<0 for v in cur]
  wrongtokens+=sum(iswrong);wronggroups+=any(iswrong);allwronggroups+=all(iswrong);opposinggroup+=sum(g>0 if k==5 else g<0 for g in grp)
  gcur+=cur;gnew+=new
 out[name]=dict(n=len(sub),trajectories=len(sub)*5,current_objective_wrong_direction_trajectories=wrongtokens,current_objective_at_least_one_wrong_direction_groups=wronggroups,current_objective_allfive_wrong_direction_groups=allwronggroups,binary_group_term_wrong_direction_trajectories=opposinggroup,candidate_wrong_direction_trajectories=sum(v>0 if k==5 else v<0 for v in gnew),current_mean_logit_gradient=sum(gcur)/len(gcur))
print(json.dumps(dict(status='ACTUAL_E2_PREDICTION_LOGIT_GRADIENT_DIAGNOSIS',negative_group_ranking_gradient_zero=True,weights='trajectory mean BCE1 + binary Mixed BCE1; G5',result=out,caveat='Output-logit gradients at actual EMA outputs, not a replayed shared-network parameter update or proof of convergence',time=time.time())))
