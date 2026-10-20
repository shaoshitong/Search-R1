"""Same questions and denominators for prefix5/10/20 collision comparisons."""
import json,gzip,math,time
from pathlib import Path
from collections import Counter
B=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019')
def read(p):return json.loads(Path(p).read_text())
def scan(base,sets):
 truth={g['qid']:g for own in sets.values() for g in own};tokens={};files=set();common=Counter()
 for f in read(base/'bank/features.json'):files.update(Path(f).parent.parent.glob('rank-*-prefix.jsonl.gz'))
 for path in sorted(files):
  with gzip.open(path,'rt') as f:
   for line in f:
    r=json.loads(line)
    if r['qid'] in truth:tokens.setdefault(r['qid'],{})[r['response_index']]=(r['trajectory_id'],tuple(r['policy_token_ids']),tuple(r['policy_position_ids']))
 assert set(tokens)==set(truth)
 out={}
 for name,groups in sets.items():
  out[name]={}
  for length in (5,10,20):
   a=dict(questions=len(groups),trajectories=5*len(groups),all5_same=0,any_same_pair=0,any_same_prefix_opposite_EM_pair=0,equal_pairs=0,opposite_EM_equal_pairs=0,min_errors_under_token_equivalence=0,entropy_sum=0.,short_trajectories=0)
   for g in groups:
    own=[tokens[g['qid']][j] for j in range(5)];assert [r[0] for r in own]==g['trajectory_ids']
    signatures=[(r[1][:length],r[2][:length]) for r in own];counts=Counter(signatures);a['all5_same']+=len(counts)==1;a['any_same_pair']+=len(counts)<5;a['short_trajectories']+=sum(len(r[1])<length for r in own)
    labels={}
    for sig,y in zip(signatures,g['labels']):labels.setdefault(sig,[]).append(y)
    conflict=False
    for ys in labels.values():
     n=len(ys);k=sum(ys);a['equal_pairs']+=n*(n-1)//2;a['opposite_EM_equal_pairs']+=k*(n-k);conflict|=0<k<n;a['min_errors_under_token_equivalence']+=min(k,n-k)
     if 0<k<n:
      p=k/n;a['entropy_sum']+=-n*(p*math.log(p)+(1-p)*math.log(1-p))
    a['any_same_prefix_opposite_EM_pair']+=conflict
    if name=='train_mixed' and length==5:common.update(r[1][:5] for r in own)
   n=a['questions'];a.update(all5_same_fraction=a['all5_same']/n,any_same_pair_fraction=a['any_same_pair']/n,any_same_prefix_opposite_EM_pair_fraction=a['any_same_prefix_opposite_EM_pair']/n,equal_pair_fraction=a['equal_pairs']/(10*n),opposite_EM_equal_pair_fraction_of_all_pairs=a['opposite_EM_equal_pairs']/(10*n),token_equivalence_trajectory_accuracy_ceiling=1-a['min_errors_under_token_equivalence']/(5*n),token_equivalence_trajectory_BCE_floor=a.pop('entropy_sum')/(5*n))
   out[name][str(length)]=a
 return out,[dict(token_ids=list(ids),trajectories=n,fraction=n/(len(sets.get('train_mixed',[]))*5)) for ids,n in common.most_common(8)]
original=read(B/'bank/groups.json');sets={name+'_mixed':[g for g in original[name] if 0<sum(g['labels'])<5] for name in ('train','diagnostic')}
out,common=scan(B,sets);h=read(B/'selection-2048-20261020/bank/groups.json')['selection_validation'];held,_=scan(B/'selection-2048-20261020',{'unseen_mixed':[g for g in h if 0<sum(g['labels'])<5]});out.update(held)
assert out['train_mixed']['5']['any_same_prefix_opposite_EM_pair']==40734 and out['train_mixed']['5']['opposite_EM_equal_pairs']==87968
print('COMPARISON_JSON='+json.dumps(dict(time=time.time(),statistics=out,frequent_prefix5=common,note='Same-question token IDs plus actual positions. Not exact BF16 feature equality, and prefix20 collision reduction does not prove good final-EM prediction.'),separators=(',',':')))
