"""Token-equivalence fit limits from already verified positive-negative masks.

Not a strict bound for jittered native BF16 tensors: this assumes a stable
predictor outputs the same probability for the same question+policy5 tokens
and positions. Pure same-label components have zero entropy and need no edges.
"""
import json,math,time
from pathlib import Path
from collections import Counter
B=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/mixed-only-ft-cp2-20261020')
groups=json.loads((B/'bank/groups.json').read_text())
def analyze(rows):
 result=dict(questions=len(rows),trajectories=5*len(rows),groups_with_conflicting_identical_prefix=0,all5_same_prefix_with_conflicting_labels=0,minimum_wrong_trajectories=0,entropy_sum=0.,excluded_positive_negative_pairs=0,eligible_positive_negative_pairs=0)
 hist=Counter()
 for g in rows:
  y=g['labels'];mask=g['pair_mask'];assert 0<sum(y)<5
  parent=list(range(5))
  def find(i):
   while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
   return i
  blocked=0
  for i in range(5):
   for j in range(5):
    if y[i]==1 and y[j]==0:
     if mask[i][j]:result['eligible_positive_negative_pairs']+=1
     else:
      blocked+=1;parent[find(i)]=find(j)
  components={}
  for i in range(5):components.setdefault(find(i),[]).append(y[i])
  result['excluded_positive_negative_pairs']+=blocked
  result['groups_with_conflicting_identical_prefix']+=blocked>0
  result['all5_same_prefix_with_conflicting_labels']+=len(components)==1
  mistakes=0
  for own in components.values():
   n=len(own);k=sum(own);mistakes+=min(k,n-k)
   if 0<k<n:
    r=k/n;result['entropy_sum']+=-n*(r*math.log(r)+(1-r)*math.log(1-r))
  hist[mistakes]+=1;result['minimum_wrong_trajectories']+=mistakes
 n=result['trajectories'];result.update(token_equivalence_maximum_training_trajectory_accuracy=1-result['minimum_wrong_trajectories']/n,token_equivalence_trajectory_BCE_lower_bound=result.pop('entropy_sum')/n,minimum_mistakes_per_group_histogram=dict(hist))
 return result
out=dict(time=time.time(),status='MIXED_TOKEN_EQUIVALENCE_AMBIGUITY_AGGREGATE',assumption='Question+policy5 token IDs and positions predict identically; native BF16 numerical jitter is not model information. This is not a strict bound on potentially different cached feature bit patterns.',train=analyze(groups['train']),diagnostic_mixed=analyze(groups['diagnostic_mixed']))
print(json.dumps(out))
