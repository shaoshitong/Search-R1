"""Bound strict-v5 CF-only 16-result gate; never normal generation or full run."""
import argparse,fcntl,hashlib,json,os,pathlib,subprocess,sys,time
p=argparse.ArgumentParser(__doc__);p.add_argument('--job',required=True);args=p.parse_args()
job=json.loads(pathlib.Path(args.job).read_text());assert job['mode']=='pilot16' and job['module']=='reproduction.grounding_eval.strict_v5'
assert job['method']=='Search-G1' and job['source_commit']=='16c36aa777f684f245b1d38cb2939ef175131831'
root=pathlib.Path(job['control_output']);root.mkdir(exist_ok=True);lock=open(root/'controller.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
canon=lambda x:json.dumps(x,sort_keys=True,separators=(',',':')).encode();sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def publish(name,value):
 path=root/name;temp=path.with_name(path.name+'.tmp')
 with temp.open('w')as f:json.dump(value,f);f.flush();os.fsync(f.fileno())
 os.replace(temp,path)
if(root/'job-binding.json').exists():assert json.loads((root/'job-binding.json').read_text())==job
else:publish('job-binding.json',job)
receipt=json.loads((pathlib.Path(job['runtime'])/'deployment-receipt.json').read_text());assert receipt['source_commit']==job['source_commit']
for rel,v in receipt['files'].items():assert sha(pathlib.Path(job['runtime'])/rel)==v['sha256']
publish('worker.json',dict(pid=os.getpid(),command=sys.argv,started_at=time.time(),job_sha256=hashlib.sha256(canon(job)).hexdigest(),gpu_uuid=job['gpu_uuid'],scope='strict-v5 pilot16 only; no full promotion'))
env=dict(os.environ,PYTHONPATH=job['runtime'],PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='8')
def step(phase,command):
 started=time.time();publish('progress.json',dict(phase=phase,command=command,started_at=started))
 with(root/(phase+'.log')).open('ab')as log:child=subprocess.Popen(command,cwd=job['runtime'],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT);code=child.wait()
 publish(phase+'-exit.json',dict(phase=phase,child_pid=child.pid,exit_code=code,started_at=started,completed_at=time.time()))
 if code:raise RuntimeError(phase+' exited '+str(code))
module=[job['python'],'-m',job['module']];cf=pathlib.Path(job['cf_output'])
try:
 manifest=cf/'manifest.json'
 if not manifest.exists():step('prepare',module+['prepare','--source',job['source'],'--checkpoint',job['checkpoint'],'--output',str(cf),'--method',job['method'],'--run-id',job['run_id'],'--normal-config',job['normal_config']])
 else:
  m=json.loads(manifest.read_text());assert m['method']==job['method']and m['run_id']==job['run_id']and m['protocol_version']=='strict-v5-conservative-typed'
 assert not list((cf/'results').glob('*.json')),'Do not repeat or expand the initial pilot without review'
 step('pilot16',module+['pilot','--manifest',str(manifest),'--max-batch','256','--max-memory-utilization','0.90','--gpu-uuid',job['gpu_uuid']])
 assert len(list((cf/'results').glob('*.json')))==16
 step('verify-pilot',module+['verify','--manifest',str(manifest)])
 publish('complete.json',dict(state='PILOT_COMPLETE_AWAITING_MANUAL_REVIEW',completed_at=time.time(),results=16,full_run_authorized=False,method=job['method'],cf_output=str(cf),source_commit=job['source_commit']))
 publish('progress.json',dict(phase='PILOT_COMPLETE_AWAITING_MANUAL_REVIEW',completed_at=time.time(),results=16))
except BaseException as e:
 publish('failed.json',dict(error=type(e).__name__+': '+str(e),failed_at=time.time()));raise
