"""Complete-data gated authentic G20 matched pair, isolated normal1GPU jobs.

Prepare is CPU-only. Explicit submit preserves existing IDs and unknown
submission intents. No dependencies, GPU-priority escalation, or live edits.
"""
import argparse,base64,hashlib,importlib.util,inspect,json,shlex,subprocess
from pathlib import Path
from .deploy_confirmation_materialization import make_bundle
from . import deploy_peer_newseed_replay as transport
from .g20_matched_inputs import TARGET,PARENT,VARIANTS,TRAIN_STATUS,VALIDATION_STATUS,VALIDATION_COMPLETE_SHA
TRAIN=PARENT+'/algorithm-improvement-20261021/training-G20-matched6528-20261021/materialized/bank/complete.json'
VAL=PARENT+'/algorithm-improvement-20261021/validation-g10-20261021/materialized/bank/complete.json'
PACKAGE='reproduction/operations/qwen7b_warmup/'
SEEDS=tuple(PACKAGE+n for n in ('deploy_g20_matched_suite.py','g20_matched_inputs.py','g20_qualification_entry.py','probe_g20_train.py','test_probe_g20_train.py','test_g20_question_loss.py','test_g20_grouping.py','runtime.sh'))
MANIFEST='_g20_matched_source_manifest.json'

COMMON=transport.COMMON+r'''
def ready():
 train=read(inputs['train']);val=read(inputs['validation'])
 assert train['status']==inputs['train_status'] and train['questions']==6528 and train['trajectories']==130560
 assert fp(inputs['train'])['sha256']=='81e3e6b74666e3a0eb32d1b0a0fb9e6db13d54f30a351437ce57f13be46bc229'
 assert val['status']==inputs['validation_status'] and val['questions']==2048 and val['trajectories']==20480 and fp(inputs['validation'])['sha256']==inputs['validation_sha']
def prepared():
 done=read(b/'preparation.json');pub,root=publication()
 assert done['commit']==pub['commit'] and done['source_group_size']==20 and done['supports_G20'] and done['questions']==6528 and done['updates']==136
 for value in done['variants'].values():
  assert fp(value['config']['path'])==value['config'] and fp(value['sbatch']['path'])==value['sbatch']
 return done,pub,root
'''
INIT=r'''
ready();guard=lock();(b/'logs').mkdir(exist_ok=True);(b/'inputs').mkdir(exist_ok=True);uploads.mkdir(exist_ok=True)
v={k:inputs[k] for k in ('commit','bundle_sha256','manifest_sha256','chunks')};p=b/'upload-controller.json'
if p.exists():assert read(p)==v,'Different immutable G20 source exists; use a new branch'
else:atomic(p,v)
print(json.dumps(dict(status='AUTHENTIC_G20_SOURCE_UPLOAD_BOUND',chunks=inputs['chunks'])))
'''
CHUNK=transport.CHUNK
EXTRACT=r'''
guard=lock();data=b''.join(p.read_bytes() for p in sorted(uploads.glob('part-*')));assert hashlib.sha256(data).hexdigest()==inputs['bundle_sha256']
with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as tar:
 raw=tar.extractfile(inputs['manifest']).read();assert hashlib.sha256(raw).hexdigest()==inputs['manifest_sha256'];manifest=json.loads(raw);files=manifest['files'];assert {m.name for m in tar.getmembers()}==set(files)|{inputs['manifest']}
 for name in set(files)|{inputs['manifest']}:
  rel=pathlib.Path(name);assert not rel.is_absolute() and '..' not in rel.parts;member=tar.getmember(name);assert member.isfile();raw=tar.extractfile(member).read()
  if name in files:assert len(raw)==files[name]['bytes'] and hashlib.sha256(raw).hexdigest()==files[name]['sha256']
  p=code/rel;p.parent.mkdir(exist_ok=True,parents=True)
  if p.exists():assert p.read_bytes()==raw
  else:p.write_bytes(raw);p.chmod(0o444)
value=dict(commit=inputs['commit'],code_root=str(code),files=files,bundle_sha256=inputs['bundle_sha256']);p=b/'source-publication.json'
if p.exists():assert read(p)==value
else:atomic(p,value)
publication();print(json.dumps(dict(status='AUTHENTIC_G20_IMMUTABLE_CLOSURE_SHA_ACCEPTED',code_root=str(code),files=len(files))))
'''
CPU=r'''
guard=lock();ready();pub,root=publication();env=dict(os.environ,QWEN7B_CODE_ROOT=str(root),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
args=runtime(root,['unittest','reproduction.operations.qwen7b_warmup.test_probe_g20_train','reproduction.operations.qwen7b_warmup.test_g20_question_loss','reproduction.operations.qwen7b_warmup.test_g20_grouping'])
r=subprocess.run(args,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True,timeout=90);(b/'actual-cpu-tests.log').write_text(r.stdout);assert r.returncode==0,r.stdout[-3000:]
args=runtime(root,['reproduction.operations.qwen7b_warmup.g20_matched_inputs','--base',str(b),'--training-complete',inputs['train'],'--validation-complete',inputs['validation']])
r=subprocess.run(args,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True,timeout=180);(b/'actual-cpu-prepare.log').write_text(r.stdout);assert r.returncode==0,r.stdout[-4000:]
done,pub,root=prepared();print(json.dumps({k:done[k] for k in ('status','commit','code_root','g20_complete','validation_g10_complete','source_plan','variants','questions','real_train_trajectories','global_original_questions','updates','source_group_size','heterogeneous_questions','source_Adam_steps','optimizer_reset')}))
'''
SUBMIT=r'''
guard=lock();ready();done,pub,root=prepared();item=done['variants'][inputs['variant']];out=pathlib.Path(item['base'])
if (out/'job.id').exists():print(json.dumps(dict(status='EXISTING_G20_JOB_ID_PRESERVED',job_id=(out/'job.id').read_text().strip())))
else:
 assert not (out/'submission-intent.json').exists(),'Unknown existing submit outcome; reconcile before repeating'
 assert not (out/'latest.json').exists() and not (out/'progress.json').exists(),'Existing training state needs exact explicit resume, never a fresh submission'
 assert not command(['squeue','-h','-u','mou176','-n',item['job_name'],'-o','%i|%T|%R']),'Unknown same-name job'
 intent=dict(status='G20_MATCHED_SUBMIT_INTENT',variant=inputs['variant'],job_name=item['job_name'],config=item['config'],script=item['sbatch'],partition='acd_u',qos='normal',gpus=1,cpus=8,memory='64G',time_limit='02:00:00',dependency=None,time=time.time());atomic(out/'submission-intent.json',intent)
 try:r=subprocess.run(['sbatch','--hold','--parsable',item['sbatch']['path']],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=25)
 except subprocess.TimeoutExpired:atomic(out/'submission-unknown.json',dict(status='UNKNOWN_OUTCOME_DO_NOT_REPEAT',time=time.time()));raise
 if r.returncode:atomic(out/'submission-rejected.json',dict(stderr=r.stderr[-1500:],time=time.time()));raise RuntimeError(r.stderr[-1500:])
 job=r.stdout.strip().split(';')[0];assert job.isdigit(),'Unknown reply; preserve intent and reconcile';p=out/'job.id'
 with p.open('x') as f:f.write(job+'\n');f.flush();os.fsync(f.fileno())
 state=command(['squeue','-h','-j',job,'-o','%T|%R']);assert state.startswith('PENDING|') and 'JobHeldUser' in state,state
 command(['scontrol','release',job]);value=dict(intent,status='G20_MATCHED_JOB_RELEASED_NOT_PROGRESS_VERIFIED',job_id=job);atomic(out/'submission.json',value);print(json.dumps(value))
'''
STATUS=r'''
def optional(p):
 try:return read(p)
 except (FileNotFoundError,json.JSONDecodeError):return None
out=dict(epoch=time.time(),base=str(b),variants={})
done=optional(b/'preparation.json')
if done is not None:
 out['preparation']={k:done[k] for k in ('status','questions','real_train_trajectories','global_original_questions','updates','source_group_size','heterogeneous_questions')};ids=[]
 for name,item in done['variants'].items():
  d=pathlib.Path(item['base']);own={}
  try:own['job_id']=(d/'job.id').read_text().strip();ids.append(own['job_id'])
  except FileNotFoundError:pass
  for filename in ('progress.json','next-update-replay.json','failure.json','interrupted.json','gpu-memory.json','complete.json'):
   v=optional(d/filename)
   if v is not None:own[filename]={k:v[k] for k in ('status','cursor','steps','time','loss','components','message','source_steps','candidate_steps','recomputed_logits_bit_identical','mixed_balanced_accuracy','question_weighted_confusion','peak_allocated_bytes','peak_reserved_bytes','exit_normally','metrics') if k in v}
  out['variants'][name]=own
 if ids:out['squeue']=command(['squeue','-h','-j',','.join(ids),'-o','%i|%j|%T|%R|%N']);out['sacct']=command(['sacct','-X','-n','-P','-j',','.join(ids),'--format=JobID,State,ExitCode,Elapsed'])
print(json.dumps(out))
'''


def bundle(repo):
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip();remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/codex/qwen7b-probe-warmup-20261019'],cwd=repo,text=True).split()[0]
    if commit!=remote:raise ValueError('Publish exact source before preparation')
    paths=subprocess.check_output(['git','ls-tree','-r','--name-only',commit],cwd=repo,text=True).splitlines();cache={}
    def load(name):
        if name not in cache:cache[name]=subprocess.check_output(['git','show',commit+':'+name],cwd=repo)
        return cache[name]
    data,report=make_bundle(paths,load,commit,seeds=SEEDS,resource_files=('verl/version/version',),manifest_name=MANIFEST)
    for name in SEEDS:
        if (repo/name).read_bytes()!=load(name):raise ValueError('Unpublished source '+name)
    return commit,data,report


def main():
    p=argparse.ArgumentParser();p.add_argument('--control-root',type=Path);p.add_argument('--action',choices=('prepare','submit','status'),default='status');p.add_argument('--variant',choices=VARIANTS);p.add_argument('--submit-authorized',action='store_true');p.add_argument('--self-test',action='store_true');a=p.parse_args();repo=Path(__file__).resolve().parents[3]
    values=dict(base=TARGET,target=TARGET,parent=PARENT,train=TRAIN,validation=VAL,train_status=TRAIN_STATUS,validation_status=VALIDATION_STATUS,validation_sha=VALIDATION_COMPLETE_SHA,manifest=MANIFEST,variant=a.variant,commit='a'*40,bundle_sha256='b'*64,manifest_sha256='c'*64,chunks=2)
    blocks=dict(init=INIT,chunk=CHUNK,extract=EXTRACT,cpu=CPU,submit=SUBMIT,status=STATUS)
    def source(action,extra=None):
        own=dict(values);own.update(extra or {});return 'inputs='+repr(own)+'\n'+COMMON+'\n'+blocks[action]
    if a.self_test:
        lengths={}
        for action in blocks:
            s=source(action,dict(chunk_index=0,chunk_data=base64.b64encode(b'x'*4096).decode(),chunk_sha256='d'*64));compile(s,action,'exec');lengths[action]=len(('python3 -c '+shlex.quote(s)).encode());assert lengths[action]<16000
        from .g20_matched_inputs import render_sbatch
        for name in VARIANTS:
            text=render_sbatch(name,TARGET+'/'+name,TARGET+'/code-a');subprocess.run(['bash','-n'],input=text,text=True,check=True);assert '#SBATCH --time=02:00:00' in text and '--dependency' not in text and '--qos=normal' in text
            exports='\n'.join(x for x in text.splitlines() if x.startswith('export '));subprocess.run(['bash','-euc',exports+'\ntest "$QWEN7B_CODE_ROOT" = '+shlex.quote(TARGET+'/code-a')],check=True)
        print(json.dumps(dict(status='G20_NORMAL_PAIR_CODE_COMMAND_AND_SHELL_EXPORT_CHECKS_PASS',command_bytes=lengths)));return
    if a.control_root is None:p.error('--control-root required')
    if a.action=='submit' and (not a.variant or not a.submit_authorized):p.error('Explicit variant and --submit-authorized required')
    data=b''
    if a.action=='prepare':
        commit,data,report=bundle(repo);values.update(commit=commit,bundle_sha256=hashlib.sha256(data).hexdigest(),manifest_sha256=report['source_manifest_sha256'],chunks=(len(data)+4095)//4096)
    spec=importlib.util.spec_from_file_location('g20_control',a.control_root/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');control=importlib.util.module_from_spec(spec);spec.loader.exec_module(control);address=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0'],text=True).strip()
    def call(action,extra=None):
        command='python3 -c '+shlex.quote(source(action,extra));assert len(command.encode())<16000;r=control.run(command,account='mou176',source_address=address,timeout=240 if action=='cpu' else 55);rows=[json.loads(line) for line in r.stdout.decode(errors='replace').splitlines() if line.startswith('{')]
        if r.returncode or not rows:raise RuntimeError('Unconfirmed read/prepare/submit; inspect before repeating: '+r.stdout.decode(errors='replace')[-2500:]+r.stderr.decode(errors='replace')[-500:])
        return rows[-1]
    if a.action=='prepare':
        call('init')
        for i,start in enumerate(range(0,len(data),4096)):call('chunk',dict(chunk_index=i,chunk_data=base64.b64encode(data[start:start+4096]).decode(),chunk_sha256=hashlib.sha256(data[start:start+4096]).hexdigest()))
        call('extract');result=call('cpu')
    else:result=call(a.action)
    out=a.control_root/'outputs/qwen7b-warmup-20261019'/('g20-matched-'+a.action+('-'+a.variant if a.variant else '')+'.json');out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,sort_keys=True))
if __name__=='__main__':main()
