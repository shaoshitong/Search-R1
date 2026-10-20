import pathlib,json,gzip,math,time
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/mixed-only-ft-cp2-prefix20-20261020');bank=json.loads((b/'bank/groups.json').read_text());out={'epoch':1,'time':time.time(),'weights':'EMA0.99','scope':'BCE independently recomputed from sealed float32 probabilities; correct saturated predictions contribute0; no clipping; trainer stable-logit receipt pending','splits':{}}
for split in ('train','diagnostic'):
 truth={g['qid']:g for g in bank[split]};seen=set();sums=[0.,0.];wrong_saturated=[0,0];correct_saturated=[0,0];n=right=0
 for rank in range(4):
  path=b/'probe/predictions'/('epoch-0001-%s-rank%d.jsonl.gz'%(split,rank));assert path.is_file()
  with gzip.open(str(path),'rt') as f:
   for line in f:
    r=json.loads(line);qid=r['qid'];assert qid not in seen and qid in truth;seen.add(qid);t=truth[qid];assert r['strict_em']==t['labels'] and r['trajectory_ids']==t['trajectory_ids']
    pairs=[(0,p,y) for p,y in zip(r['p_j'],r['strict_em'])]+[(1,r['predicted_mixed'],int(0<sum(r['strict_em'])<5))]
    for kind,p,y in pairs:
     assert math.isfinite(p) and 0<=p<=1
     if p==y:correct_saturated[kind]+=1
     elif p==0 or p==1:wrong_saturated[kind]+=1
     else:sums[kind]+=-math.log(p) if y else -math.log1p(-p)
    right+=sum((p>=.5)==bool(y) for p,y in zip(r['p_j'],r['strict_em']));n+=1
 assert seen==set(truth)
 out['splits'][split]={'questions':n,'trajectory_bce':None if wrong_saturated[0] else sums[0]/(5*n),'mixed_bce':None if wrong_saturated[1] else sums[1]/n,'trajectory_accuracy':right/(5*n),'correct_saturated_counts':correct_saturated,'wrong_saturated_counts':wrong_saturated,'counts_order':['trajectory','group'],'clipping_used':False}
print(json.dumps(out))
