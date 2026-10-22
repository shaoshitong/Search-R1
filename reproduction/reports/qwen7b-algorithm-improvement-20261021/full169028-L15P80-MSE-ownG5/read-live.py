"""Read actual two full-data replica handles and bounded progress; no tensor loads."""
import collections,json,pathlib,subprocess,time

A=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021')
ROOTS={
 'seed1':A/'full169028-L15P80-MSE-ownG5-physical-seed9137-plan2026102227-20261023-v3',
 'seed2':A/'full169028-L15P80-MSE-ownG5-physical-seed19137-plan2026102302-20261023-v3',
}

def read(path):return json.loads(path.read_bytes())
def run(args):
 p=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=20)
 return dict(returncode=p.returncode,stdout=p.stdout.strip(),stderr=p.stderr[-1000:])
def tail(path,budget=64000):
 with path.open('rb')as f:
  size=path.stat().st_size;f.seek(max(0,size-budget));raw=f.read()
 rows=raw.decode(errors='replace').splitlines()
 if size>budget:rows=rows[1:]
 return rows

out=dict(time=time.time(),replicas={},read_only=True,checkpoint_tensors_loaded=False,final4096_read=False)
for name,base in ROOTS.items():
 cell=dict(base=str(base),files={})
 for filename in ('root-submission.json','root-pending-walltime-rightsize.json','controller.json','progress-rank-0.json','progress-rank-1.json','failure-rank-0.json','failure-rank-1.json','failure.json','latest.json','complete.json'):
  if(base/filename).is_file():cell['files'][filename]=read(base/filename)
 receipt=cell['files'].get('root-submission.json')
 if receipt:
  job=str(receipt['job_id']);cell['squeue']=run(['squeue','-h','-j',job,'-o','%i|%T|%M|%R']);cell['sacct']=run(['sacct','-n','-X','-j',job,'--format=JobIDRaw,State,ExitCode,Elapsed,NodeList','-P'])
  log=base/('train-%s.log'%job)
  if log.is_file():cell['log_tail']=tail(log,5000)[-12:]
 if(base/'curve.jsonl').is_file():
  rows=[]
  for line in tail(base/'curve.jsonl'):
   try:row=json.loads(line)
   except ValueError:continue
   keys=('epoch','cursor','actual_Adam_updates','EMA_updates','rank','local_batch','actual_global_batch','gradient_accumulation','micro_question_counts','native_gradient_L2','head_gradient_L2','prefill_rows_equal_groups','prefill_seconds','time')
   rows.append({k:row[k]for k in keys if k in row})
  cell['recent_actual_updates']=rows[-8:]
 cell['finished_evaluations']={}
 for epoch in range(11):
  p=base/('evaluation-epoch-%02d.json'%epoch)
  if p.is_file():cell['finished_evaluations'][str(epoch)]=read(p)
 out['replicas'][name]=cell
print(json.dumps(out))
