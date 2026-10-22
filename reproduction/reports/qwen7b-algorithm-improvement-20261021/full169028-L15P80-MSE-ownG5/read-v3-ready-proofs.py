"""Read actual two-v3 ready/publication hashes only; no submissions or tensors."""
import pathlib,json,hashlib,time
A=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021')
def proof(p):
 r=p.read_bytes();return dict(path=str(p),bytes=len(r),sha256=hashlib.sha256(r).hexdigest())
out=dict(time=time.time(),replicas={},Slurm_mutated=False,final4096_read=False)
for name,seed,plan in [('seed1',9137,2026102227),('seed2',19137,2026102302)]:
 b=A/('full169028-L15P80-MSE-ownG5-physical-seed%d-plan%d-20261023-v3'%(seed,plan));r=b/'ready.json';entry=b/'training.sbatch';pub=b/'source-publication.json';x=dict(root=str(b),ready=None)
 if r.exists():
  v=json.loads(r.read_bytes());assert v['status']=='FULL169028_L15P80_MSE_OWNG5_ACCEPTED_FIXTURE_READY_NOT_SUBMITTED'and v['entry']==proof(entry)and v['publication']==proof(pub)and v['actual_physical_execution_and_seal_source_exact']and v['fresh_full_P20_prefix_prompt_and_own_EM_exact'];x.update(ready=proof(r),ready_contents=v)
 out['replicas'][name]=x
print(json.dumps(out))
