import os,time,json,subprocess,socket
from pathlib import Path
b=Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-micro8-after731063-20261018');job='734689';new='745022'
run=json.loads((b/'logs/run-manifest.json').read_text())
assert os.getuid()==14090 and(b/'job.id').read_text().strip()==job
assert run['run_id']=='dfe9c94084bb48b2a9bb7b60f9d882da'
assert run['source_commit']=='4cc07f3103917f5ac6bb81f6b022d454809c1054'
assert run['model_revision']=='aa8e72537993ba99e69dfaafa59ed015b17504d1'
old=subprocess.run(['scontrol','show','job','-o',job],capture_output=True,text=True,timeout=10)
assert old.returncode==0 and 'JobId=734689 'in old.stdout and 'UserId=jxu957(14090)'in old.stdout
assert str(b)in old.stdout
c=b/'preparation/human-stop-for-7b-20261021.json';assert not c.exists(),'Read the previous cancellation result; do not repeat unknown operation'
checkpoint_root=b/'checkpoints/search-g1-common8/actor'
cps=[dict(path=str(p),complete=(p/'complete.json').is_file())for p in checkpoint_root.glob('global_step_*')if p.is_dir()]
ledger=b/'logs/actual-updates.jsonl';rows=[json.loads(x)for x in ledger.read_text().splitlines()if x]if ledger.exists()else[]
value=dict(observed_unix=time.time(),human_instruction='2026-10-21 current user explicitly stops 3B, enough training; keep 7B goal running',scope_job=job,scope_base=str(b),run_id=run['run_id'],new7b_job_not_stopped=new,last_actual_updates=rows[-2:],checkpoints=cps,pre_stop_slurm=old.stdout,command=['scancel',job],reason='HUMAN_REQUESTED_STOP_NOT_TRAINING_FAILURE',restart_authorized=False)
with c.open('x')as f:json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno())
x=subprocess.run(['scancel',job],capture_output=True,text=True,timeout=10)
result=dict(epoch=time.time(),returncode=x.returncode,stdout=x.stdout,stderr=x.stderr,stop_receipt=str(c),last_actual_updates=rows[-2:],checkpoints=cps,new7b_job_not_stopped=new)
p=c.with_name('human-stop-for-7b-20261021-command-result.json')
with p.open('x')as f:json.dump(result,f,indent=2);f.flush();os.fsync(f.fileno())
print(json.dumps(result))
