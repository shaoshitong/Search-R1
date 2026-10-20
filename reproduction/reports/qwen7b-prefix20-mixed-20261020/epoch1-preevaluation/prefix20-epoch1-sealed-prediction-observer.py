import json,gzip,pathlib,math,time,hashlib
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/mixed-only-ft-cp2-prefix20-20261020');bank=json.loads((b/'bank/groups.json').read_text());out={'time':time.time(),'epoch':1,'scope':'sealed train/diagnostic predictions; final trainer metrics and heldout selection pending','splits':{}}
for split in ('train','diagnostic'):
 truth={g['qid']:g for g in bank[split]};seen=set();n=right=identical=near=0;ran=qsum=0.;conf=[[0]*3 for _ in range(3)];files=[]
 for rank in range(4):
  path=b/'probe/predictions'/('epoch-0001-%s-rank%d.jsonl.gz'%(split,rank));assert path.is_file(),str(path)
  raw=path.read_bytes();files.append({'path':str(path),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()})
  for line in gzip.decompress(raw).decode().splitlines():
   g=json.loads(line);qid=g['qid'];assert qid in truth and qid not in seen;seen.add(qid);t=truth[qid]
   assert g['trajectory_ids']==t['trajectory_ids'] and g['strict_em']==t['labels'] and g['weight_kind']=='EMA0.99' and g['epoch']==1
   p=g['p_j'];assert len(p)==5 and all(math.isfinite(v) and 0<=v<=1 for v in p)
   ar=aw=1.
   for v in p:ar*=v;aw*=1-v
   assert abs(1-ar-aw-g['predicted_mixed'])<2e-6
   y=g['strict_em'];right+=sum((v>=.5)==bool(a) for v,a in zip(p,y));n+=1;gap=max(p)-min(p);ran+=gap;identical+=len(set(p))==1;near+=gap<=.01;qsum+=g['predicted_mixed']
   cat=lambda k:0 if k==0 else 2 if k==5 else 1
   conf[cat(sum(y))][cat(sum(v>=.5 for v in p))]+=1
 assert seen==set(truth)
 out['splits'][split]={'questions':n,'trajectory_accuracy':right/(5*n),'mean_p_range':ran/n,'all5_identical_p_fraction':identical/n,'range_le001_fraction':near/n,'mean_predicted_mixed':qsum/n,'three_class_order':['all_wrong','mixed','all_correct'],'three_class_confusion':conf,'files':files}
print(json.dumps(out))
