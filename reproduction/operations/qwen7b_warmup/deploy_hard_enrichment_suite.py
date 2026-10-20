"""Immutable E4 hard/random suite: CPU prepare/status, explicit guarded submit.

No prepare before actual E4-scoring/CPU-selection complete; no dependency or
automatic submit. All tensor/feature/prediction data remain on shared storage.
"""
import argparse,base64,gzip,hashlib,importlib.util,io,json,shlex,subprocess,tarfile
from pathlib import Path
from . import deploy_g10_pilot_suite as transport
from .deploy_confirmation_materialization import import_closure
from .hard_enrichment_train_prepare import PARENT,TARGET,SELECT

VARIANTS=('targeted','random_control')
SEEDS=tuple('reproduction/operations/qwen7b_warmup/'+n for n in
    ('hard_enrichment_train.py','hard_enrichment_train_prepare.py','test_hard_enrichment_train.py','runtime.sh'))


def closure(repo):
    paths=[str(p.relative_to(repo)) for folder in ('reproduction','search_g1','search_r1','verl')
        for p in (repo/folder).rglob('*.py') if '__pycache__' not in p.parts]
    paths.append('reproduction/operations/qwen7b_warmup/runtime.sh')
    chosen,_,unresolved=import_closure(paths,lambda n:(repo/n).read_bytes(),seeds=SEEDS,resource_files=())
    if unresolved:raise ValueError('Unresolved actual runtime imports '+str(unresolved))
    return chosen


def bundle(repo,published=True):
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    if published:
        remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/codex/qwen7b-probe-warmup-20261019'],cwd=repo,text=True).split()[0]
        if commit!=remote:raise ValueError('Publish exact latest hard/random source first')
        if Path(__file__).read_bytes()!=subprocess.check_output(['git','show',commit+':'+str(Path(__file__).relative_to(repo))],cwd=repo):raise ValueError('Unpublished deployment helper')
    files={};out=io.BytesIO()
    with gzip.GzipFile(fileobj=out,mode='wb',mtime=0,filename='') as gz:
        with tarfile.open(fileobj=gz,mode='w') as archive:
            for name in closure(repo):
                raw=(repo/name).read_bytes()
                if published and raw!=subprocess.check_output(['git','show',commit+':'+name],cwd=repo):raise ValueError('Unpublished runtime '+name)
                files[name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
                entry=tarfile.TarInfo(name);entry.size=len(raw);entry.mode=0o444;entry.mtime=0;archive.addfile(entry,io.BytesIO(raw))
    return commit,out.getvalue(),files


COMMON=transport.COMMON.split('def verify_prepared():')[0]+r'''
def ready():
 p=pathlib.Path(inputs['selection_complete']);v=read(p)
 assert v['status']=='TRAIN_ONLY_HARD_AND_RANDOM22896Q_CLASS_MATCHED_PLANS_COMPLETE'
 assert set(v['outputs'])=={'targeted','random_control'} and v['checkpoint']['steps']==14418
 assert v['checkpoint']['sha256']=='e29168c36de152a57294f055e7d420cbba25cae31aeace32e793ba1098bf97a8'
 return fp(p)
def suite():
 pub,root=publication();done=read(b/'preparation.json')
 assert done['commit']==pub['commit'] and done['code_root']==str(root)
 assert done['fit_updates_per_variant']==477 and done['fit_questions']==22896 and done['global_groups']==48
 assert done['source_adam_steps']==14418 and done['E1_reset'] is False
 for key in ['selection_complete','validation_complete','cpu_tests']:verify(done[key])
 for item in done['variants'].values():verify(item['config']);verify(item['sbatch'])
 return done,pub,root
'''
INIT=r'''
selection=ready();b.mkdir(parents=True,exist_ok=True);guard=lock()
if (b/'preparation.json').exists():
 done,pub,root=suite();assert done['selection_complete']==selection
 print(json.dumps(dict(status='EXISTING_E4_HARD_RANDOM_SUITE_VERIFIED',preparation=done)))
else:
 for name in ['logs','inputs']:(b/name).mkdir(exist_ok=True)
 wanted={k:inputs[k] for k in ['commit','bundle_sha256','bundle_bytes','chunks','files']};wanted['selection_complete']=selection
 p=b/'upload-controller.json'
 if p.exists():assert read(p)==wanted,'Different immutable source/selection already bound'
 else:atomic(p,wanted)
 uploads.mkdir(exist_ok=True);print(json.dumps(dict(status='E4_HARD_RANDOM_CODE_UPLOAD_BOUND',chunks=inputs['chunks'])))
'''
CPU_DRIVER=r'''
import json,os,sys,subprocess,pathlib,fcntl,time,traceback
b=pathlib.Path(sys.argv[1]);root=pathlib.Path(sys.argv[2])
def atomic(p,v):
 t=p.with_suffix('.tmp');t.write_text(json.dumps(v,sort_keys=True)+'\n');t.replace(p)
with (b/'cpu-run.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 env=dict(os.environ,QWEN7B_CODE_ROOT=str(root),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
 runtime=str(root/'reproduction/operations/qwen7b_warmup/runtime.sh')
 try:
  cmd=['bash',runtime,'-m','unittest','reproduction.operations.qwen7b_warmup.test_hard_enrichment_train']
  with (b/'actual-cpu-tests.log').open('w') as log:r=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=100)
  assert r.returncode==0,'Actual CPU continuation tests failed'
  cmd=['bash',runtime,'-u','-m','reproduction.operations.qwen7b_warmup.hard_enrichment_train_prepare','--base',str(b)]
  with (b/'cpu-prepare.log').open('w') as log:r=subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=180)
  assert r.returncode==0,'Actual CPU metadata preparation failed'
  atomic(b/'cpu-result.json',dict(status='E4_HARD_RANDOM_CPU_PREPARATION_COMPLETE',exit_code=0,time=time.time()))
 except BaseException:
  atomic(b/'cpu-result.json',dict(status='E4_HARD_RANDOM_CPU_PREPARATION_FAILED',traceback=traceback.format_exc(),time=time.time()));raise
'''
CPU=r'''
guard=lock();ready();pub,root=publication()
if (b/'cpu-launch-intent.json').exists():
 print(json.dumps(dict(status='EXISTING_CPU_INTENT_NOT_RESTARTED',intent=read(b/'cpu-launch-intent.json'),result=read(b/'cpu-result.json') if (b/'cpu-result.json').exists() else None)))
else:
 with (b/'cpu-run.lock').open('a') as f:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
 path=b/'cpu-driver.py';raw=inputs['cpu_driver'].encode()
 if path.exists():assert path.read_bytes()==raw
 else:
  with path.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 intent=dict(commit=pub['commit'],driver=fp(path),selection_complete=ready(),time=time.time(),GPU_used=False)
 atomic(b/'cpu-launch-intent.json',intent)
 with (b/'cpu-driver.log').open('ab') as log:p=subprocess.Popen(['python3','-u',str(path),str(b),str(root)],stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
 ticks=pathlib.Path('/proc/'+str(p.pid)+'/stat').read_text().split()[21]
 launch=dict(intent,pid=p.pid,start_ticks=ticks,status='CPU_STARTED_NOT_YET_PREPARED');atomic(b/'cpu-launch.json',launch);print(json.dumps(launch))
'''
SUBMIT=r'''
guard=lock();ready();done,pub,root=suite();name=inputs['variant'];assert name in done['variants']
assert read(b/'cpu-result.json')['status']=='E4_HARD_RANDOM_CPU_PREPARATION_COMPLETE'
item=done['variants'][name];base=pathlib.Path(item['base'])
if (base/'job.id').exists():
 job=(base/'job.id').read_text().strip();assert job.isdigit()
 print(json.dumps(dict(status='EXISTING_HARD_RANDOM_JOB_PRESERVED',variant=name,job_id=job,squeue=command(['squeue','-h','-j',job,'-o','%i|%j|%T|%R']),sacct=command(['sacct','-X','-n','-P','-j',job,'--format=JobID,State,ExitCode']))))
else:
 assert not (base/'submission-intent.json').exists(),'Unknown prior submit outcome; reconcile, never duplicate'
 assert not any((base/n).exists() for n in ['latest.json','progress.json','update-intent.json']),'Existing training state requires explicit resume handling'
 assert not command(['squeue','-h','-u','mou176','-n',item['job_name'],'-o','%i|%T|%R']),'Same-name active job exists'
 intent=dict(status='HARD_RANDOM_SUBMIT_INTENT',variant=name,job_name=item['job_name'],config=item['config'],script=item['sbatch'],requested_resources=dict(partition='acd_u',qos='normal',gpus=1,cpus=8,memory='64G',time='02:00:00'),dependency=None,time=time.time())
 atomic(base/'submission-intent.json',intent)
 try:r=subprocess.run(['sbatch','--hold','--parsable',item['sbatch']['path']],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=25)
 except subprocess.TimeoutExpired:
  intent['status']='SUBMIT_OUTCOME_UNKNOWN_DO_NOT_REPEAT';atomic(base/'submission-intent.json',intent);raise
 if r.returncode:
  intent.update(status='SBATCH_REJECTED',stderr=r.stderr[-2000:]);atomic(base/'submission-intent.json',intent);raise RuntimeError(r.stderr[-2000:])
 job=r.stdout.strip().split(';')[0]
 if not job.isdigit():
  intent['status']='SUBMIT_REPLY_UNKNOWN_DO_NOT_REPEAT';atomic(base/'submission-intent.json',intent);raise RuntimeError('Unknown sbatch reply')
 with (base/'job.id').open('x') as f:f.write(job+'\n');f.flush();os.fsync(f.fileno())
 record=dict(intent,status='E4_HARD_RANDOM_JOB_HELD',job_id=job);atomic(base/'submission.json',record)
 state=command(['squeue','-h','-j',job,'-o','%T|%R']);assert state.startswith('PENDING|') and 'JobHeldUser' in state,state
 command(['scontrol','release',job]);record.update(status='E4_HARD_RANDOM_JOB_RELEASED',queue=command(['squeue','-h','-j',job,'-o','%i|%j|%T|%P|%R']),time=time.time())
 atomic(base/'submission.json',record);print(json.dumps(record))
'''
STATUS=r'''
out=dict(base=str(b),exists=b.exists(),observed_epoch=time.time(),actual_selection_exists=pathlib.Path(inputs['selection_complete']).exists())
for name in ['cpu-launch.json','cpu-result.json']:
 if (b/name).exists():out[name]=read(b/name)
if (b/'preparation.json').exists():
 done,pub,root=suite();out.update(preparation=done,variants={});ids=[]
 for name,item in done['variants'].items():
  base=pathlib.Path(item['base']);own={}
  if (base/'job.id').exists():
   job=(base/'job.id').read_text().strip();assert job.isdigit();own['job_id']=job;ids.append(job)
  for filename in ['controller.json','progress.json','interrupted.json','failure.json']:
   if (base/filename).exists():
    v=read(base/filename);own[filename]={k:v[k] for k in ['status','pid','job_id','cursor','steps','budget','loss','mixed_balanced_accuracy','mixed_recall','nonmixed_recall','epoch_time','message','max_gpu_allocated'] if k in v}
  own['actual_replays']={}
  for path in base.glob('next-update-replay-cursor*.json'):
   v=read(path);own['actual_replays'][path.name]={k:v[k] for k in ['passed','source_steps','accepted_steps']}
  if (base/'complete.json').exists():
   v=read(base/'complete.json');assert v['config_sha256']==item['config']['sha256'];own['complete']=dict(status=v['status'],cursor=v['cursor'],steps=v['steps'],metrics=v['metrics'])
  own['unknown_submit_outcome']=(base/'submission-intent.json').exists() and not (base/'job.id').exists();out['variants'][name]=own
 out['squeue']=command(['squeue','-h','-u','mou176','-o','%i|%j|%T|%P|%R'])
 out['sacct']=command(['sacct','-X','-n','-P','-j',','.join(ids),'--format=JobID,State,ExitCode,Elapsed']) if ids else ''
print(json.dumps(out))
'''


def remote_source(inputs,action):
    return 'inputs='+repr(inputs)+'\n'+COMMON+'\n'+dict(init=INIT,chunk=transport.CHUNK,extract=transport.EXTRACT,cpu=CPU,submit=SUBMIT,status=STATUS)[action]


def self_test():
    repo=Path(__file__).resolve().parents[3];_,payload,files=bundle(repo,published=False)
    values=dict(base=TARGET,parent=PARENT,target=TARGET,selection_complete=SELECT,commit='a'*40,
        bundle_sha256=hashlib.sha256(payload).hexdigest(),bundle_bytes=len(payload),chunks=10,files=files,variant='targeted',cpu_driver=CPU_DRIVER)
    sizes={}
    for action in ('init','chunk','extract','cpu','submit','status'):
        own=dict(values)
        if action=='chunk':own.update(chunk_index=0,chunk_data=base64.b64encode(b'x'*4096).decode(),chunk_sha256='b'*64)
        source=remote_source(own,action);compile(source,'hard-deploy-'+action,'exec')
        sizes[action]=len(('python3 -c '+shlex.quote(source)).encode())
        if sizes[action]>=16000:raise ValueError('Control command >=16KB '+action)
    compile(CPU_DRIVER,'cpu-driver','exec')
    return dict(status='LOCAL_E4_HARD_RANDOM_DEPLOYMENT_CHECKS_PASSED',runtime_files=len(files),bundle_bytes=len(payload),command_bytes=sizes,remote_actions_executed=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--control-root',type=Path);p.add_argument('--action',choices=['prepare','status','submit'],default='status')
    p.add_argument('--variant',choices=VARIANTS);p.add_argument('--submit-authorized',action='store_true');p.add_argument('--self-test',action='store_true');a=p.parse_args()
    if a.self_test:print(json.dumps(self_test()));return
    if a.control_root is None:p.error('--control-root required')
    if a.action=='submit' and (not a.variant or not a.submit_authorized):p.error('Explicit --variant and --submit-authorized required')
    repo=Path(__file__).resolve().parents[3];commit,payload,files=bundle(repo) if a.action=='prepare' else ('',b'',{})
    chunks=[payload[i:i+4096] for i in range(0,len(payload),4096)]
    inputs=dict(base=TARGET,parent=PARENT,target=TARGET,selection_complete=SELECT,commit=commit,bundle_sha256=hashlib.sha256(payload).hexdigest(),
        bundle_bytes=len(payload),chunks=len(chunks),files=files,variant=a.variant,cpu_driver=CPU_DRIVER)
    spec=importlib.util.spec_from_file_location('hard_suite_control',a.control_root/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py')
    control=importlib.util.module_from_spec(spec);spec.loader.exec_module(control);address=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0'],text=True).strip()
    def call(action,extra=None):
        own=dict(inputs);own.update(extra or {});cmd='python3 -c '+shlex.quote(remote_source(own,action))
        if len(cmd.encode())>=16000:raise ValueError('Control command >=16KB')
        r=control.run(cmd,account='mou176',source_address=address,timeout=55)
        rows=[json.loads(line) for line in r.stdout.decode(errors='replace').splitlines() if line.startswith('{')]
        if r.returncode or not rows:raise RuntimeError('Action unconfirmed; inspect actual state without repeating: '+r.stderr.decode(errors='replace')[-1800:]+' '+r.stdout.decode(errors='replace')[-1800:])
        print(json.dumps(rows[-1]),flush=True);return rows[-1]
    if a.action=='prepare':
        result=call('init')
        if result['status']!='EXISTING_E4_HARD_RANDOM_SUITE_VERIFIED':
            for i,part in enumerate(chunks):call('chunk',dict(chunk_index=i,chunk_data=base64.b64encode(part).decode(),chunk_sha256=hashlib.sha256(part).hexdigest()))
            call('extract');result=call('cpu')
    else:result=call(a.action)
    path=a.control_root/'outputs/qwen7b-warmup-20261019'/('hard-random477-'+a.action+('-'+a.variant if a.action=='submit' else '')+'.json')
    path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
