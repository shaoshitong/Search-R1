import pathlib,json,hashlib,time
A=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021')
def ref(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
out={}
for name,expected in [('audited243-after466-20261021-45227948',31010),('audited262-after709-20261021-d3c45767',33536)]:
 b=A/('peer-newseed-ingest-'+name);p=b/'complete.json'
 if not p.exists():out[name]=dict(complete=False);continue
 v=json.loads(p.read_text());assert v['questions']==expected and v['trajectories']==10*expected and v['strict_EM_mismatches']==0 and v['actor_updates']==0
 out[name]=dict(complete=ref(p),outputs=v['outputs'],questions=v['questions'],trajectories=v['trajectories'],status=v['status'],strict_EM_mismatches=v['strict_EM_mismatches'],metadata_resealed_attestations=v.get('metadata_resealed_attestations'),result=v)
print(json.dumps(dict(epoch=time.time(),records=out)))
