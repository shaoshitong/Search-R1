"""Owned750001 metadata only. No submission, tensor load or final reads."""
import hashlib,json,math,subprocess,time
from pathlib import Path
B=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/online-packed-matched-MSE-M5-seed19137-plan2026102302-L15P80-8192-20261023-v1');D=B/'mse-M5';JOB='750001'
def read(p):return json.loads(Path(p).read_bytes())
pub=read(B/'source-publication.json');assert pub['commit']=='92ec87b0c1a02a068517c8ed84aa4b8f5727599c'
assert pub['files']['reproduction/operations/qwen7b_warmup/online_packed_matched_objective_seeded.py']['sha256']=='a4405c81749595a25da56b4bad6484b734d6adc7131a4246d2ec7194795ce3f1'
assert hashlib.sha256((D/'training.sbatch').read_bytes()).hexdigest()=='f96807fbeb52904ccba4f2421a5e12910a94e53b1db03ce409ef751ccf65b18a'
out=dict(time=time.time(),job_id=JOB,source_commit=pub['commit'],files={},final4096_read=False,Slurm_mutated=False)
for name in('root-submission.json','controller.json','progress-rank-0.json','progress-rank-1.json','latest.json','complete.json'):
 if(D/name).is_file():out['files'][name]=read(D/name)
for p in D.glob('failure-rank-*.json'):out['files'][p.name]=read(p)
for p in sorted(D.glob('evaluation-epoch-*.json')):out['files'][p.name]=read(p)
for name in('root-submission.json','controller.json'):
 if name in out['files']:assert str(out['files'][name]['job_id'])==JOB
if'controller.json'in out['files']:
 c=out['files']['controller.json'];assert c['init_seed']==19137 and c['train_plan_seed']==2026102302 and c['fixed_m']==5 and c['objective']=='mse'and c['global_batch']==128 and c['layer']==15 and c['prefix']==80
tail=[]
if(D/'curve.jsonl').exists():
 with(D/'curve.jsonl').open('rb')as f:f.seek(max(0,(D/'curve.jsonl').stat().st_size-24000));lines=f.read().splitlines()
 for line in lines:
  try:r=json.loads(line)
  except(ValueError,UnicodeDecodeError):continue
  tail.append({k:r[k]for k in('epoch','cursor','actual_Adam_updates','EMA_updates','native_gradient_L2','head_gradient_L2','training_loss','actual_global_batch','time')})
tail=tail[-6:];out['last_actual_updates']=tail
out['six_consecutive_updates_proven']=len(tail)==6 and all(b['cursor']==a['cursor']+1 and b['actual_Adam_updates']==a['actual_Adam_updates']+1 for a,b in zip(tail,tail[1:]))and all(math.isfinite(r['training_loss'])and r['native_gradient_L2']>0 and r['head_gradient_L2']>0 and r['actual_global_batch']==128 for r in tail)
r=subprocess.run(['sacct','-j',JOB,'-n','-X','-P','-o','JobIDRaw,State,ExitCode,Elapsed,NodeList'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True);out['accounting']=dict(code=r.returncode,stdout=r.stdout.strip(),stderr=r.stderr[-1000:]);r=subprocess.run(['squeue','-h','-j',JOB,'-o','%i|%T|%P|%q|%M|%R'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True);out['queue']=dict(code=r.returncode,stdout=r.stdout.strip(),stderr=r.stderr[-1000:]);print(json.dumps(out))
