"""Prepare/status/explicit guarded submit for three fresh four-epoch probes.

Immutable source closure is published before deployment; raw tar payload uses
4096B SHA-bound chunks and each actual control command stays below16KB.
"""
import argparse,ast,base64,gzip,hashlib,importlib.util,io,json,shlex,subprocess,tarfile
from pathlib import Path
from . import deploy_g10_pilot_suite as transport
from .multilayer_train_prepare import PARENT,TARGET,R2,R2_SHA

VARIANTS=('layer21','layer24','layer27')
SEEDS=tuple('reproduction/operations/qwen7b_warmup/'+n for n in
    ('deploy_multilayer_suite.py','multilayer_train_prepare.py','test_deploy_multilayer_suite.py',
     'replay_layer_train.py','replay_layer_probe.py','test_replay_layer_train.py','runtime.sh'))


def source_closure(repo):
    pending=list(SEEDS);result=set()
    while pending:
        name=pending.pop()
        if name in result:continue
        path=repo/name
        if not path.is_file():raise ValueError('Missing dependency '+name)
        result.add(name)
        if path.suffix!='.py':continue
        package=name[:-3].split('/')[:-1]
        for node in ast.walk(ast.parse(path.read_text(),filename=name)):
            modules=[]
            if isinstance(node,ast.ImportFrom):
                if node.level:
                    prefix=package[:len(package)-node.level+1]
                    modules=['.'.join(prefix+[node.module])] if node.module else ['.'.join(prefix+[a.name]) for a in node.names]
                elif node.module and node.module.startswith('reproduction.'):modules=[node.module]
            elif isinstance(node,ast.Import):modules=[a.name for a in node.names if a.name.startswith('reproduction.')]
            for module in modules:
                relative=module.replace('.','/')+'.py'
                if not (repo/relative).is_file():raise ValueError('Unresolved local dependency '+module)
                pending.append(relative)
    return tuple(sorted(result))


COMMON=transport.COMMON+r'''
def ready():
 assert sha(inputs['replay_complete'])==inputs['replay_sha'],'Actual R2 complete SHA differs'
 v=read(inputs['replay_complete']);assert v['status']=='ALL_FOUR_MATCHED9088Q_MULTILAYER_HF_REPLAY_BANKS_SHA_VERIFIED' and v['questions']==9088 and v['trajectories']==45440
def verify_suite():
 pub,root=publication();done=read(b/'preparation.json')
 assert done['commit']==pub['commit'] and done['code_root']==str(root) and done['epochs']==4 and done['updates']==512
 for key in ['replay_bank_complete','reference_config','source_plan','cpu_tests']:verify(done[key])
 for item in done['variants'].values():verify(item['config']);verify(item['sbatch'])
 return done,pub,root
def summary(done):
 keys=['status','commit','code_root','replay_bank_complete','reference_config','source_plan','cpu_tests','variants','actual_layer_counts','fit_questions','diagnostic_questions','selection_questions','epochs','updates','updates_per_epoch','global_groups','micro_groups','head_seed','Adam_initial_step','checkpoint_tensors_loaded_on_login','model_tensors_loaded_on_login','native_bitexact_claim','no_dependency','no_automatic_submission','untouched_final_read']
 return {k:done[k] for k in keys}
'''

INIT=r'''
ready();b.mkdir(parents=True,exist_ok=True);guard=lock()
if (b/'preparation.json').exists():
 done,pub,root=verify_suite();print(json.dumps(dict(status='EXISTING_MULTILAYER_SUITE_VERIFIED',preparation=summary(done))))
else:
 for name in ['logs','inputs']:(b/name).mkdir(exist_ok=True)
 own={k:inputs[k] for k in ['commit','bundle_sha256','bundle_bytes','chunks','files']};path=b/'upload-controller.json'
 if path.exists():assert read(path)==own,'Different immutable source version already bound'
 else:atomic(path,own)
 uploads.mkdir(exist_ok=True);print(json.dumps(dict(status='MULTILAYER_CODE_UPLOAD_BOUND',chunks=inputs['chunks'])))
'''

CPU_PREPARE=r'''
guard=lock();ready();pub,root=publication()
env=dict(os.environ,QWEN7B_CODE_ROOT=str(root),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
runtime=str(root/'reproduction/operations/qwen7b_warmup/runtime.sh')
cmd=['bash',runtime,'-m','unittest','reproduction.operations.qwen7b_warmup.test_replay_layer_train','reproduction.operations.qwen7b_warmup.test_deploy_multilayer_suite']
r=subprocess.run(cmd,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True,timeout=100)
(b/'actual-cpu-tests.log').write_text(r.stdout);assert r.returncode==0,r.stdout[-3000:]
cmd=['bash',runtime,'-u','-m','reproduction.operations.qwen7b_warmup.multilayer_train_prepare','--base',str(b)]
r=subprocess.run(cmd,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True,timeout=180)
(b/'cpu-prepare.log').write_text(r.stdout);assert r.returncode==0,r.stdout[-4000:]
done,pub,root=verify_suite();print(json.dumps(summary(done)))
'''

SUBMIT=r'''
guard=lock();ready();done,pub,root=verify_suite();name=inputs['variant'];assert name in done['variants']
item=done['variants'][name];base=pathlib.Path(item['base'])
if (base/'job.id').exists():
 job=(base/'job.id').read_text().strip();assert job.isdigit()
 print(json.dumps(dict(status='EXISTING_MULTILAYER_JOB_PRESERVED',variant=name,job_id=job,squeue=command(['squeue','-h','-j',job,'-o','%i|%j|%T|%R']),sacct=command(['sacct','-X','-n','-P','-j',job,'--format=JobID,JobName,State,ExitCode']))))
else:
 assert not (base/'submission-intent.json').exists(),'Prior unknown submit must be reconciled before another attempt'
 assert not (base/'latest.json').exists() and not (base/'progress.json').exists(),'Existing state requires explicit resume handling'
 assert not command(['squeue','-h','-u','mou176','-n',item['job_name'],'-o','%i|%T|%R']),'Unknown same-name job: do not duplicate'
 intent=dict(status='MULTILAYER_SUBMIT_INTENT',variant=name,job_name=item['job_name'],config=item['config'],script=item['sbatch'],
  requested_resources=dict(partition='acd_u',qos='normal',gpus=1,cpus=8,memory='64G',time='02:00:00'),dependency=None,time=time.time())
 atomic(base/'submission-intent.json',intent)
 try:r=subprocess.run(['sbatch','--hold','--parsable',item['sbatch']['path']],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=25)
 except subprocess.TimeoutExpired:
  intent['status']='SUBMIT_OUTCOME_UNKNOWN_DO_NOT_REPEAT';atomic(base/'submission-intent.json',intent);raise
 if r.returncode:
  intent.update(status='SBATCH_REJECTED',stderr=r.stderr[-2000:]);atomic(base/'submission-intent.json',intent);raise RuntimeError(r.stderr[-2000:])
 job=r.stdout.strip().split(';')[0];assert job.isdigit(),'Unknown sbatch reply; reconcile instead of repeating'
 with (base/'job.id').open('x') as f:f.write(job+'\n');f.flush();os.fsync(f.fileno())
 submitted=dict(intent,status='MULTILAYER_JOB_HELD',job_id=job);atomic(base/'submission.json',submitted)
 state=command(['squeue','-h','-j',job,'-o','%T|%R']);assert state.startswith('PENDING|') and 'JobHeldUser' in state,state
 command(['scontrol','release',job]);submitted.update(status='MULTILAYER_JOB_RELEASED',time=time.time(),queue=command(['squeue','-h','-j',job,'-o','%i|%j|%T|%P|%R']))
 atomic(base/'submission.json',submitted);print(json.dumps(submitted))
'''

STATUS=r'''
result=dict(base=str(b),exists=b.exists(),observed_epoch=time.time(),replay_complete_exists=pathlib.Path(inputs['replay_complete']).is_file())
if (b/'preparation.json').exists():
 done,pub,root=verify_suite();result.update(preparation=summary(done),variants={});ids=[]
 for name,item in done['variants'].items():
  base=pathlib.Path(item['base']);own={}
  if (base/'job.id').exists():
   job=(base/'job.id').read_text().strip();assert job.isdigit();own['job_id']=job;ids.append(job)
  for filename in ['controller.json','progress.json','restore-acceptance.json','failure.json','interrupted.json','gpu-memory.json']:
   if (base/filename).exists():
    v=read(base/filename);keys=['status','cursor','steps','epoch','within_epoch_update','epoch_time','loss','message','pid','job_id','block_layer','capture_layer','peak_allocated_bytes','peak_reserved_bytes']
    own[filename]={k:v[k] for k in keys if k in v}
  own['actual_replays']={}
  for path in base.glob('next-update-replay-cursor*.json'):
   v=read(path);own['actual_replays'][path.name]={k:v[k] for k in ['passed','source_cursor','steps']}
  own['epochs']={}
  for epoch in range(1,5):
   path=base/('evaluation-epoch-%04d.json'%epoch)
   if path.exists():
    v=read(path);assert v['config_sha256']==item['config']['sha256']
    own['epochs'][str(epoch)]=dict(cursor=v['cursor'],metrics={role:dict(pool=m['pool'],selected512=m['selected512']) for role,m in v['metrics'].items()})
  if (base/'complete.json').exists():
   v=read(base/'complete.json');assert v['config_sha256']==item['config']['sha256'];own['complete']={k:v[k] for k in ['status','cursor','steps','epochs']}
  own['unknown_submission_outcome']=(base/'submission-intent.json').exists() and not (base/'job.id').exists();result['variants'][name]=own
 result['squeue']=command(['squeue','-h','-u','mou176','-o','%i|%j|%T|%R'])
 result['sacct']=command(['sacct','-X','-n','-P','-j',','.join(ids),'--format=JobID,State,ExitCode,Elapsed']) if ids else ''
print(json.dumps(result))
'''


def remote_source(inputs,action):
    return 'inputs='+repr(inputs)+'\n'+COMMON+'\n'+dict(init=INIT,chunk=transport.CHUNK,extract=transport.EXTRACT,cpu=CPU_PREPARE,submit=SUBMIT,status=STATUS)[action]


def published_bundle(repo):
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/codex/qwen7b-probe-warmup-20261019'],cwd=repo,text=True).split()[0]
    if remote!=commit:raise RuntimeError('Publish exact latest runtime before prepare')
    stream=io.BytesIO();files={}
    with gzip.GzipFile(fileobj=stream,mode='wb',mtime=0,filename='') as zipped:
        with tarfile.open(fileobj=zipped,mode='w') as archive:
            for name in source_closure(repo):
                raw=(repo/name).read_bytes()
                if raw!=subprocess.check_output(['git','show',commit+':'+name],cwd=repo):raise RuntimeError('Unpublished runtime '+name)
                files[name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
                member=tarfile.TarInfo(name);member.size=len(raw);member.mode=0o444;member.mtime=0;archive.addfile(member,io.BytesIO(raw))
    return commit,stream.getvalue(),files


def self_test():
    repo=Path(__file__).resolve().parents[3];names=source_closure(repo)
    assert {Path(n).name for n in names}>={'replay_layer_train.py','replay_layer_probe.py','replay_feature_reader.py','multilayer_train_prepare.py','replay_bank_prepare.py','runtime.sh'}
    values=dict(base=TARGET,parent=PARENT,target=TARGET,replay_complete=R2,replay_sha=R2_SHA,commit='a'*40,
        bundle_sha256='b'*64,bundle_bytes=100000,chunks=25,files={n:dict(bytes=10000,sha256='c'*64) for n in names},variant='layer21')
    sizes={}
    for action in ('init','chunk','extract','cpu','submit','status'):
        own=dict(values)
        if action=='chunk':own.update(chunk_index=0,chunk_data=base64.b64encode(b'x'*4096).decode(),chunk_sha256='d'*64)
        source=remote_source(own,action);compile(source,'remote-'+action,'exec');sizes[action]=len(('python3 -c '+shlex.quote(source)).encode())
        if sizes[action]>=16000:raise AssertionError('16KB control command exceeded: '+action)
    value=dict(status='LOCAL_THREE_LAYER_REAL_DEPLOYMENT_CHECKS_PASSED',code_files=len(names),chunk_bytes=4096,command_bytes=sizes,
        remote_actions_executed=False,training_launched=False)
    print(json.dumps(value));return value


def main():
    p=argparse.ArgumentParser();p.add_argument('--control-root',type=Path);p.add_argument('--action',choices=['prepare','status','submit'],default='status')
    p.add_argument('--variant',choices=VARIANTS);p.add_argument('--submit-authorized',action='store_true');p.add_argument('--self-test',action='store_true');a=p.parse_args()
    if a.self_test:self_test();return
    if a.control_root is None:p.error('--control-root required')
    if a.action=='submit' and (not a.variant or not a.submit_authorized):p.error('Explicit --variant and --submit-authorized required')
    repo=Path(__file__).resolve().parents[3];commit,bundle,files=published_bundle(repo) if a.action=='prepare' else ('',b'',{})
    chunks=[bundle[i:i+4096] for i in range(0,len(bundle),4096)]
    inputs=dict(base=TARGET,parent=PARENT,target=TARGET,replay_complete=R2,replay_sha=R2_SHA,commit=commit,
        bundle_sha256=hashlib.sha256(bundle).hexdigest(),bundle_bytes=len(bundle),chunks=len(chunks),files=files,variant=a.variant)
    spec=importlib.util.spec_from_file_location('multilayer_control',a.control_root/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py')
    control=importlib.util.module_from_spec(spec);spec.loader.exec_module(control)
    address=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0'],text=True).strip()
    def call(action,extra=None):
        own=dict(inputs);own.update(extra or {});cmd='python3 -c '+shlex.quote(remote_source(own,action));assert len(cmd.encode())<16000
        r=control.run(cmd,account='mou176',source_address=address,timeout=290 if action=='cpu' else 55)
        rows=[json.loads(v) for v in r.stdout.decode(errors='replace').splitlines() if v.startswith('{')]
        if r.returncode or not rows:raise RuntimeError('Multilayer action unconfirmed; inspect actual state: '+r.stderr.decode(errors='replace')[-1500:]+' '+r.stdout.decode(errors='replace')[-4000:])
        print(json.dumps(rows[-1]),flush=True);return rows[-1]
    if a.action=='prepare':
        result=call('init')
        if result['status']!='EXISTING_MULTILAYER_SUITE_VERIFIED':
            for i,part in enumerate(chunks):call('chunk',dict(chunk_index=i,chunk_data=base64.b64encode(part).decode(),chunk_sha256=hashlib.sha256(part).hexdigest()))
            for action in ('extract','cpu'):result=call(action)
    else:result=call(a.action)
    dest=a.control_root/'outputs/qwen7b-warmup-20261019'/('multilayer4epoch-'+a.action+('-'+a.variant if a.action=='submit' else '')+'.json')
    dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
