import pathlib,json,hashlib
f=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/online-mixed-layer-prefix-20261022-v1/inputs')
out=[]
for e in(1,2,3):
 p=f/('epoch-%02d-plan.json'%e);raw=p.read_bytes();rows=json.loads(raw);assert all(v['true_mixed']==(0<sum(v['labels'])<5)for v in rows)
 m=sum(v['true_mixed']for v in rows);n=len(rows);out.append(dict(epoch=e,questions=n,unique_questions=len({v['qid']for v in rows}),Mixed=m,nonMixed=n-m,positive_loss_weight=n/(2*m),negative_loss_weight=n/(2*(n-m)),plan_file_sha256=hashlib.sha256(raw).hexdigest()))
print(json.dumps(dict(plans=out,final4096_read=False)))
