import json,hashlib
from pathlib import Path
base=Path('/mnt/data/shaoshitong/search-g1-question20-replay70-train-test-20261016-5fa565b')
old=Path('/mnt/data/shaoshitong/search-g1-question20-replay70-20261016-5534d25')
def score(group):
 p=group['tp']+group['fn'];n=group['tn']+group['fp']
 assert p>0 and n>0 and group['threshold']==.5
 return dict(mixed_recall=group['tp']/p,nonmixed_recall=group['tn']/n,balanced_accuracy=(group['tp']/p+group['tn']/n)/2,ordinary_accuracy=(group['tp']+group['tn'])/(p+n),mixed_count=p,nonmixed_count=n,tp=group['tp'],fn=group['fn'],tn=group['tn'],fp=group['fp'])
rows=[]
for i in range(1,71):
 owner=old if i<=6 else base;p=owner/'train/iterations'/('iteration-%04d'%i)/'metrics.json';data=p.read_bytes();m=json.loads(data)
 assert m['replay_iteration']==i and m['actual_adam_steps_per_rank']==[7100+100*i]*8
 assert m['test_manifest_sha256']=='a125fc5af90e34fcd728c82077d9d62070a78d5ef703ac11f3748687d088e64f'
 test=m.get('test_mixed_group',m['mixed_group']);assert(test['mixed_count'],test['nonmixed_count'])==(183,329)
 rows.append(dict(replay_iteration=i,train=score(m['train_mixed_group'])if i>=7 else None,test=score(test),metrics_path=str(p),metrics_sha256=hashlib.sha256(data).hexdigest()))
best=max(r['test']['balanced_accuracy']for r in rows)
print(json.dumps(dict(role='posthoc_balanced_mixed_accuracy',threshold=.5,formula='0.5*(TP/(TP+FN)+TN/(TN+FP))',constant_nonmixed_balanced_accuracy=.5,rows=rows,final=rows[-1],best_test_balanced_accuracy=best,best_test_balanced_iterations=[r['replay_iteration']for r in rows if r['test']['balanced_accuracy']==best],training_modified=False,model_loaded=False,gpu_used=False)))
