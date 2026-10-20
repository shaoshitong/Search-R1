"""Read-only remote G5 prefix/probability audit; emits aggregate metadata only."""
import json,gzip,math,time
from pathlib import Path
from collections import Counter
from functools import reduce
import operator
def prod(xs):return reduce(operator.mul,xs,1.)
B=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019')
E=B/'selection-2048-20261020'
def read(p):return json.loads(Path(p).read_text())
def rows(p):
 with gzip.open(p,'rt') as f:
  for line in f:yield json.loads(line)
def q(p):return max(0.,min(1.,1-prod(p)-prod(1-x for x in p)))
def bce(p,y):return -math.log(max(1e-15,p if y else 1-p))
def quant(v):
 v=sorted(v);n=len(v)
 return {str(a):v[round((n-1)*a)] for a in (0,.1,.25,.5,.75,.9,.99,1)} if v else {}
def prefix_stats(base,groups):
 truth={g['qid']:sum(g['labels']) for g in groups}
 tokens={};files=set()
 for m in read(base/'bank/features.json'):
  root=Path(m).parent.parent
  files.update(root.glob('rank-*-prefix.jsonl.gz'))
 for p in sorted(files):
  for r in rows(p):
   if r['qid'] in truth:tokens.setdefault(r['qid'],[]).append((r['response_index'],tuple(r['policy_token_ids']),tuple(r['policy_position_ids'])))
 assert set(tokens)==set(truth),(len(tokens),len(truth))
 out={}
 for length in (5,10,20):
  stats={}
  for name,predicate in [('all',lambda k:True),('all_wrong',lambda k:k==0),('mixed',lambda k:0<k<5),('all_correct',lambda k:k==5)]:
   n=equal=pairs=dups=position_equal=short=0;hist=Counter()
   for qid,own in tokens.items():
    if not predicate(truth[qid]):continue
    assert len(own)==5 and sorted(r[0] for r in own)==list(range(5))
    ts=[r[1][:length] for r in own];ps=[(r[1][:length],r[2][:length]) for r in own]
    counts=Counter(ts);u=len(counts);n+=1;hist[u]+=1;equal+=u==1;position_equal+=len(set(ps))==1
    pairs+=sum(c*(c-1)//2 for c in counts.values());dups+=5-u;short+=any(len(r[1])<length for r in own)
   stats[name]=dict(questions=n,unique_prefixes_histogram=dict(hist),all5_identical=equal,all5_identical_fraction=equal/n if n else None,equal_pairs=pairs,total_pairs=10*n,equal_pair_fraction=pairs/(10*n) if n else None,redundant_trajectory_fraction=dups/(5*n) if n else None,tokens_and_positions_all5_identical=position_equal,questions_with_short_prefix=short)
  out[str(length)]=stats
 return dict(prefix_files=len(files),statistics=out)
def prob_stats(rs,truth):
 accum={name:dict(n=0,exact_equal=0,range_le_001=0,range_le_005=0,all_04_06=0,all_045_055=0,all_03_07=0,all_049_051=0,straddles_half=0,min_le_02_max_ge_09=0,min_le_03_max_ge_07=0,sum_std=0.,sum_range=0.,sum_pmean=0.,sum_q=0.,sum_q_equal_mean=0.,sum_loss=0.,sum_equal_mean_loss=0.,ranges=[],pmeans=[],qs=[]) for name in ('all','all_wrong','mixed','all_correct')}
 examples=[]
 for r in rs:
  p=r['p_j'];k=sum(truth[r['qid']]);target=0<k<5;pm=sum(p)/5;std=(sum((x-pm)**2 for x in p)/5)**.5;ran=max(p)-min(p);v=q(p);cf=q([pm]*5)
  for name in ('all','mixed' if target else 'all_wrong' if k==0 else 'all_correct'):
   a=accum[name];a['n']+=1;a['exact_equal']+=len(set(p))==1;a['range_le_001']+=ran<=.01;a['range_le_005']+=ran<=.05
   for key,lo,hi in [('all_04_06',.4,.6),('all_045_055',.45,.55),('all_03_07',.3,.7),('all_049_051',.49,.51)]:a[key]+=all(lo<=x<=hi for x in p)
   a['straddles_half']+=min(p)<.5<=max(p);a['min_le_02_max_ge_09']+=min(p)<=.2 and max(p)>=.9;a['min_le_03_max_ge_07']+=min(p)<=.3 and max(p)>=.7
   for key,x in [('sum_std',std),('sum_range',ran),('sum_pmean',pm),('sum_q',v),('sum_q_equal_mean',cf),('sum_loss',bce(v,target)),('sum_equal_mean_loss',bce(cf,target))]:a[key]+=x
   a['ranges'].append(ran);a['pmeans'].append(pm);a['qs'].append(v)
  if target and ran<.01 and len(examples)<3:examples.append(dict(p_j=p,q=v,true_K=k))
 for a in accum.values():
  n=a['n']
  for key in list(a):
   if key.startswith('sum_'):a['mean_'+key[4:]]=a.pop(key)/n if n else None
  for key in ('ranges','pmeans','qs'):a[key+'_quantiles']=quant(a.pop(key))
 return dict(groups=accum,mixed_close_p_examples=examples)
def main():
 started=time.time();g=read(B/'bank/groups.json');h=read(E/'bank/groups.json')['selection_validation']
 report=dict(time=started,status='READ_ONLY_AUDIT',prefix={},probability={})
 for split in ('train','diagnostic'):
  report['prefix'][split]=prefix_stats(B,g[split]);print('PARTIAL_PREFIX='+json.dumps({split:report['prefix'][split]}),flush=True)
 report['prefix']['unseen2048']=prefix_stats(E,h);print('PARTIAL_PREFIX='+json.dumps({'unseen2048':report['prefix']['unseen2048']}),flush=True)
 for epoch in range(5):
  report['probability'][str(epoch)]={}
  for split in ('train','diagnostic'):
   truth={r['qid']:r['labels'] for r in g[split]};paths=sorted((B/'probe/predictions').glob('epoch-%04d-%s-rank*.jsonl.gz'%(epoch,split)))
   assert len(paths)==4
   rs=(r for p in paths for r in rows(p));report['probability'][str(epoch)][split]=prob_stats(rs,truth)
  if epoch:
   rs=list(rows(E/'results'/('epoch-%04d-predictions.jsonl.gz'%epoch)));truth={r['qid']:r['labels'] for r in h}
   report['probability'][str(epoch)]['unseen2048']=prob_stats(rs,truth)
   selected=sorted(rs,key=lambda r:(-r['predicted_mixed'],r['qid']))[:512]
   report['probability'][str(epoch)]['selected512']=prob_stats(selected,truth)
  print('PARTIAL_PROBABILITY='+json.dumps({str(epoch):report['probability'][str(epoch)]}),flush=True)
 report['elapsed']=time.time()-started
 print('AUDIT_JSON='+json.dumps(report,separators=(',',':')),flush=True)
if __name__=='__main__':main()
