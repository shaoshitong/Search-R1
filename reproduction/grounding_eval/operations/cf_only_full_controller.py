"""Explicitly reviewed strict-v5 CF-only full continuation; no normal generation."""
import argparse,fcntl,hashlib,json,os,pathlib,subprocess,sys,time
p=argparse.ArgumentParser(__doc__);p.add_argument('--job',required=True);a=p.parse_args();job=json.loads(pathlib.Path(a.job).read_text())
assert job['mode']=='full_after_pilot_review'and job['module']=='reproduction.grounding_eval.strict_v5'and job['method']in ['Search-G1','Soft-to-Hard','OPSD3304']
assert job['source_commit']=='16c36aa777f684f245b1d38cb2939ef175131831'
review=job['pilot_manual_review'];assert review['approved']is True and review['reviewed_pilot_sha256']=='1fad1cb490e182920dc2f9f9edccb8520bbc2949a5425fdcda998786b00c4cca'
if job['method']!='Search-G1':assert review['g1_independent_acceptance_sha256']=='111ca6a60a4513faf122305141593ddea914d46cfd82914d4d5fd4bd29b69797'
root=pathlib.Path(job['control_output']);root.mkdir(exist_ok=True);lock=open(root/'controller.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
canon=lambda x:json.dumps(x,sort_keys=True,separators=(',',':')).encode();sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def publish(name,obj):
 p=root/name;t=p.with_name(p.name+'.tmp')
 with t.open('w')as f:json.dump(obj,f);f.flush();os.fsync(f.fileno())
 os.replace(t,p)
if(root/'job-binding.json').exists():assert json.loads((root/'job-binding.json').read_text())==job
else:publish('job-binding.json',job)
if(root/'complete.json').exists():assert json.loads((root/'complete.json').read_text())['state']=='CF_ONLY_COMPLETE';raise SystemExit(0)
receipt=json.loads((pathlib.Path(job['runtime'])/'deployment-receipt.json').read_text());assert receipt['source_commit']==job['source_commit']
for rel,v in receipt['files'].items():assert sha(pathlib.Path(job['runtime'])/rel)==v['sha256']
publish('worker.json',dict(pid=os.getpid(),command=sys.argv,started_at=time.time(),job_sha256=hashlib.sha256(canon(job)).hexdigest(),gpu_uuid=job['gpu_uuid']))
env=dict(os.environ,PYTHONPATH=job['runtime'],PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false',OMP_NUM_THREADS='8')
def step(phase,command):
 started=time.time();publish('progress.json',dict(phase=phase,command=command,started_at=started))
 with(root/(phase+'.log')).open('ab')as log:p=subprocess.Popen(command,cwd=job['runtime'],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT);code=p.wait()
 publish(phase+'-exit.json',dict(phase=phase,child_pid=p.pid,exit_code=code,started_at=started,completed_at=time.time()))
 if code:raise RuntimeError(phase+' exited '+str(code))
module=[job['python'],'-m',job['module']];cf=pathlib.Path(job['cf_output'])
try:
 manifest=cf/'manifest.json'
 if not manifest.exists():step('prepare',module+['prepare','--source',job['source'],'--checkpoint',job['checkpoint'],'--output',str(cf),'--method',job['method'],'--run-id',job['run_id'],'--normal-config',job['normal_config']])
 m=json.loads(manifest.read_text());assert m['method']==job['method']and m['run_id']==job['run_id']and m['protocol_version']=='strict-v5-conservative-typed'
 assert m['checkpoint']['path']==str(pathlib.Path(job['checkpoint']).resolve())and m['source']['path']==str(pathlib.Path(job['source']).resolve())and m['normal_config_path']==str(pathlib.Path(job['normal_config']).resolve())
 step('run',module+['run','--manifest',str(manifest),'--max-batch','256','--max-memory-utilization','0.90','--gpu-uuid',job['gpu_uuid']])
 step('verify',module+['verify','--manifest',str(manifest)])
 summary=json.loads((cf/'verified-summary.json').read_text());assert summary['status']=='complete'and summary['counts']['normal_questions']==3610 and summary['coverage']['pending_or_error']==0
 publish('complete.json',dict(state='CF_ONLY_COMPLETE',completed_at=time.time(),exit_code=0,method=job['method'],cf_output=str(cf),source_commit=job['source_commit'],verified_summary=summary))
 publish('progress.json',dict(phase='COMPLETE',completed_at=time.time(),method=job['method']))
except BaseException as e:publish('failed.json',dict(error=type(e).__name__+': '+str(e),failed_at=time.time()));raise
