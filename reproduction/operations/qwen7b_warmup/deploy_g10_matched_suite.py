"""Published-code, complete-data gated G10 random/equal-quota suite deployment.

No action occurs on import/self-test. Prepare cannot start training. Explicit
submit is guarded by ID/name/intent, never dependencies or an automatic retry.
"""
import argparse,ast,base64,gzip,hashlib,importlib.util,io,json,shlex,subprocess,tarfile
from pathlib import Path
from . import deploy_g10_pilot_suite as transport
from .g10_matched_inputs import PARENT,TARGET,VARIANTS,TRAIN_STATUS,VALIDATION_STATUS,VALIDATION_COMPLETE_SHA

TRAIN=PARENT+'/algorithm-improvement-20261021/training-extraG5-matched6528-20261021/materialized/bank/complete.json'
VALIDATION=PARENT+'/algorithm-improvement-20261021/validation-g10-20261021/materialized/bank/complete.json'
SEEDS=tuple('reproduction/operations/qwen7b_warmup/'+n for n in
    ('deploy_g10_matched_suite.py','g10_matched_inputs.py','test_deploy_g10_matched_suite.py',
     'g10_qualification_entry.py','probe_g10_train.py','test_probe_g10_train.py','test_g10_weighted_balanced.py','runtime.sh'))


def source_closure(repo):
    pending=list(SEEDS);result=set()
    while pending:
        name=pending.pop()
        if name in result:continue
        path=repo/name
        if not path.is_file():raise ValueError('Missing local dependency '+name)
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
                child=module.replace('.','/')+'.py'
                if not (repo/child).is_file():raise ValueError('Unresolved local dependency '+module)
                pending.append(child)
    return tuple(sorted(result))


COMMON=transport.COMMON+r'''
def ready():
 train=read(inputs['training_complete']);val=read(inputs['validation_complete'])
 assert train['status']==inputs['train_status'] and train['questions']==6528 and train['trajectories']==65280,'Actual complete6528 training G10 is required; no partial/fallback'
 assert val['status']==inputs['validation_status'] and val['questions']==2048 and val['trajectories']==20480,'Actual complete validation2048 G10 is required'
 assert sha(inputs['validation_complete'])==inputs['validation_complete_sha'],'Known actual validation complete SHA differs'
 assert val['fixed_G5_was_chosen_before_own_EM'] and val['consumer_inputs']['selection_labels_consulted'] is False
def verify_matched():
 done,pub,root=verify_prepared();verify(done['validation_g10_complete'])
 assert done['questions']==6528 and done['global_original_questions']==48 and done['updates']==136
 return done,pub,root
def summary(done):
 keys=['status','commit','code_root','g10_complete','validation_g10_complete','source_plan','fixed_g5','variants','heterogeneous_questions','questions','real_train_trajectories','global_original_questions','updates','views_per_question','uniform_view_weight','source_Adam_steps','optimizer_reset','smaller_question_budget_acknowledged','equal_budget_to_B51_128_baseline_claim','same6528_cohort_as_old_matched128','no_automatic_submission','no_dependency','final_pool_read']
 return {k:done[k] for k in keys}
'''

INIT=r'''
ready();b.mkdir(parents=True,exist_ok=True);guard=lock()
if (b/'preparation.json').exists():
 done,pub,root=verify_matched();print(json.dumps(dict(status='EXISTING_MATCHED6528_G10_PREPARATION_VERIFIED',preparation=summary(done))))
else:
 for name in ['logs','inputs']:(b/name).mkdir(exist_ok=True)
 own={k:inputs[k] for k in ['commit','bundle_sha256','bundle_bytes','chunks','files']};path=b/'upload-controller.json'
 if path.exists():assert read(path)==own,'Different immutable preparation already exists'
 else:atomic(path,own)
 uploads.mkdir(exist_ok=True);print(json.dumps(dict(status='MATCHED6528_G10_UPLOAD_BOUND',chunks=inputs['chunks'])))
'''

CPU_PREPARE=r'''
guard=lock();ready();pub,root=publication()
env=dict(os.environ,QWEN7B_CODE_ROOT=str(root),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
runtime=str(root/'reproduction/operations/qwen7b_warmup/runtime.sh')
test=['bash',runtime,'-m','unittest','reproduction.operations.qwen7b_warmup.test_probe_g10_train','reproduction.operations.qwen7b_warmup.test_g10_weighted_balanced','reproduction.operations.qwen7b_warmup.test_deploy_g10_matched_suite']
r=subprocess.run(test,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True,timeout=100)
(b/'actual-cpu-tests.log').write_text(r.stdout);assert r.returncode==0,r.stdout[-3000:]
args=['bash',runtime,'-u','-m','reproduction.operations.qwen7b_warmup.g10_matched_inputs','--base',str(b),'--training-complete',inputs['training_complete'],'--validation-complete',inputs['validation_complete']]
r=subprocess.run(args,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True,timeout=180)
(b/'cpu-input-preparation.log').write_text(r.stdout);assert r.returncode==0,r.stdout[-4000:]
done,pub,root=verify_matched();print(json.dumps(summary(done)))
'''

SUBMIT=r'''
guard=lock();ready();done,pub,root=verify_matched();name=inputs['variant'];assert name in done['variants']
item=done['variants'][name];base=pathlib.Path(item['base'])
if (base/'job.id').exists():
 job=(base/'job.id').read_text().strip();assert job.isdigit()
 print(json.dumps(dict(status='EXISTING_MATCHED_G10_JOB_PRESERVED',variant=name,job_id=job,squeue=command(['squeue','-h','-j',job,'-o','%i|%j|%T|%R']),sacct=command(['sacct','-X','-n','-P','-j',job,'--format=JobID,JobName,State,ExitCode']))))
else:
 assert not (base/'submission-intent.json').exists(),'Prior submission outcome must be reconciled, never repeat blindly'
 assert not (base/'latest.json').exists() and not (base/'progress.json').exists(),'Existing training state requires explicit resume handling'
 assert not command(['squeue','-h','-u','mou176','-n',item['job_name'],'-o','%i|%T|%R']),'Unknown same-name job'
 intent=dict(status='MATCHED_G10_SUBMIT_INTENT',variant=name,job_name=item['job_name'],config=item['config'],script=item['sbatch'],
  requested_resources=dict(partition='acd_u',qos='normal',gpus=1,cpus=8,memory='64G',time='01:00:00'),dependency=None,time=time.time())
 atomic(base/'submission-intent.json',intent)
 try:r=subprocess.run(['sbatch','--hold','--parsable',item['sbatch']['path']],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=25)
 except subprocess.TimeoutExpired:
  intent['status']='SUBMIT_OUTCOME_UNKNOWN_DO_NOT_REPEAT';atomic(base/'submission-intent.json',intent);raise
 if r.returncode:
  intent.update(status='SBATCH_REJECTED',stderr=r.stderr[-2000:]);atomic(base/'submission-intent.json',intent);raise RuntimeError(r.stderr[-2000:])
 job=r.stdout.strip().split(';')[0];assert job.isdigit(),'Unknown sbatch reply; reconcile before another attempt'
 with (base/'job.id').open('x') as f:f.write(job+'\n');f.flush();os.fsync(f.fileno())
 submitted=dict(intent,status='MATCHED_G10_JOB_HELD',job_id=job);atomic(base/'submission.json',submitted)
 state=command(['squeue','-h','-j',job,'-o','%T|%R']);assert state.startswith('PENDING|') and 'JobHeldUser' in state,state
 command(['scontrol','release',job]);submitted.update(status='MATCHED_G10_JOB_RELEASED',time=time.time(),queue=command(['squeue','-h','-j',job,'-o','%i|%j|%T|%P|%R']))
 atomic(base/'submission.json',submitted);print(json.dumps(submitted))
'''

STATUS=r'''
result=dict(base=str(b),exists=b.exists(),observed_epoch=time.time(),training_complete_exists=pathlib.Path(inputs['training_complete']).is_file(),validation_complete_exists=pathlib.Path(inputs['validation_complete']).is_file())
if (b/'preparation.json').exists():
 done,pub,root=verify_matched();result.update(preparation=summary(done),variants={});ids=[]
 for name,item in done['variants'].items():
  base=pathlib.Path(item['base']);own={}
  if (base/'job.id').exists():
   job=(base/'job.id').read_text().strip();assert job.isdigit();own['job_id']=job;ids.append(job)
  for filename in ['progress.json','next-update-replay.json','failure.json','interrupted.json','gpu-memory.json']:
   if (base/filename).exists():
    v=read(base/filename);keys=['status','cursor','steps','time','loss','message','source_steps','candidate_steps','recomputed_logits_bit_identical','peak_allocated_bytes','peak_reserved_bytes','exit_normally']
    own[filename]={k:v[k] for k in keys if k in v}
  if (base/'complete.json').exists():
   v=read(base/'complete.json');assert v['config_sha256']==item['config']['sha256']
   own['complete']={k:v[k] for k in ['status','cursor','steps','metrics']}
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
    if remote!=commit:raise RuntimeError('Exact current runtime must be committed and pushed first')
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
    required={'g10_weighted_balanced.py','probe_g10_train.py','g10_matched_inputs.py','g10_qualification_entry.py','model.py','native_capture.py','download_model.py','runtime.sh'}
    assert {Path(n).name for n in names}>=required
    values=dict(base=TARGET,parent=PARENT,target=TARGET,training_complete=TRAIN,validation_complete=VALIDATION,
        train_status=TRAIN_STATUS,validation_status=VALIDATION_STATUS,validation_complete_sha=VALIDATION_COMPLETE_SHA,commit='a'*40,bundle_sha256='b'*64,
        bundle_bytes=100000,chunks=25,files={n:dict(bytes=10000,sha256='c'*64) for n in names},variant='random')
    sizes={}
    for action in ['init','chunk','extract','cpu','submit','status']:
        own=dict(values)
        if action=='chunk':own.update(chunk_index=0,chunk_data=base64.b64encode(b'x'*4096).decode(),chunk_sha256='d'*64)
        source=remote_source(own,action);compile(source,'remote-'+action,'exec');sizes[action]=len(('python3 -c '+shlex.quote(source)).encode())
        if sizes[action]>=16000:raise AssertionError('Control command exceeds16KB: '+action)
    result=dict(status='LOCAL_MATCHED6528_G10_DEPLOYMENT_CHECKS_PASSED',code_files=len(names),raw_chunk_bytes=4096,
        command_bytes=sizes,training_launched=False,remote_actions_executed=False)
    print(json.dumps(result));return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--control-root',type=Path);p.add_argument('--action',choices=['prepare','status','submit'],default='status')
    p.add_argument('--variant',choices=VARIANTS);p.add_argument('--submit-authorized',action='store_true');p.add_argument('--self-test',action='store_true');args=p.parse_args()
    if args.self_test:self_test();return
    if args.control_root is None:p.error('--control-root required')
    if args.action=='submit' and (not args.variant or not args.submit_authorized):p.error('Explicit --variant and --submit-authorized required')
    repo=Path(__file__).resolve().parents[3];commit,bundle,files=published_bundle(repo) if args.action=='prepare' else ('',b'',{})
    chunks=[bundle[i:i+4096] for i in range(0,len(bundle),4096)]
    inputs=dict(base=TARGET,parent=PARENT,target=TARGET,training_complete=TRAIN,validation_complete=VALIDATION,
        train_status=TRAIN_STATUS,validation_status=VALIDATION_STATUS,validation_complete_sha=VALIDATION_COMPLETE_SHA,commit=commit,bundle_sha256=hashlib.sha256(bundle).hexdigest(),
        bundle_bytes=len(bundle),chunks=len(chunks),files=files,variant=args.variant)
    spec=importlib.util.spec_from_file_location('matched_g10_control',args.control_root/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py')
    control=importlib.util.module_from_spec(spec);spec.loader.exec_module(control)
    address=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0'],text=True).strip()
    def call(action,extra=None):
        own=dict(inputs);own.update(extra or {});command='python3 -c '+shlex.quote(remote_source(own,action));assert len(command.encode())<16000
        r=control.run(command,account='mou176',source_address=address,timeout=290 if action=='cpu' else 55)
        rows=[json.loads(v) for v in r.stdout.decode(errors='replace').splitlines() if v.startswith('{')]
        if r.returncode or not rows:raise RuntimeError('Matched-G10 action unconfirmed; inspect actual state: '+r.stderr.decode(errors='replace')[-1500:]+' '+r.stdout.decode(errors='replace')[-4000:])
        print(json.dumps(rows[-1]),flush=True);return rows[-1]
    if args.action=='prepare':
        result=call('init')
        if result['status']!='EXISTING_MATCHED6528_G10_PREPARATION_VERIFIED':
            for i,part in enumerate(chunks):call('chunk',dict(chunk_index=i,chunk_data=base64.b64encode(part).decode(),chunk_sha256=hashlib.sha256(part).hexdigest()))
            for action in ['extract','cpu']:result=call(action)
    else:result=call(args.action)
    dest=args.control_root/'outputs/qwen7b-warmup-20261019'/('g10-matched6528-'+args.action+('-'+args.variant if args.action=='submit' else '')+'.json')
    dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(result,indent=2)+'\n')


if __name__=='__main__':main()
