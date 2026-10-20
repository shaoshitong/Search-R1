"""Publish small metadata-only fork helper; explicit guarded normal-QoS arrays.

Prepared parent/current jobs are read-only. GPU jobs always invoke their already
qualified immutable collector source, not this newly uploaded metadata helper.
"""
import argparse,ast,base64,gzip,hashlib,importlib.util,io,json,shlex,subprocess,tarfile
from pathlib import Path
from . import deploy_g10_pilot_suite as transport
from .extra_g5_fork_prepare import PARENT,A,PARENT_BASE,PINNED_PARENT_SHA,target

TARGET=A+'/extraG5-offset-fork-control-v1'
SEEDS=tuple('reproduction/operations/qwen7b_warmup/'+n for n in
    ('deploy_extra_g5_forks.py','extra_g5_fork_prepare.py','test_extra_g5_forks.py','extra_g5_prepare.py'))


def source_closure(repo):
    # extra_g5_prepare is a test fixture/metadata validator only. Its rollout
    # factory is never called here; GPU collection uses the complete OLD closure.
    pending=list(SEEDS);result=set()
    while pending:
        name=pending.pop()
        if name in result:continue
        path=repo/name
        if not path.is_file():raise ValueError('Missing metadata helper dependency '+name)
        result.add(name)
        if path.suffix!='.py' or path.name=='extra_g5_prepare.py':continue
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
                if not (repo/child).is_file():raise ValueError('Unresolved metadata helper import '+module)
                pending.append(child)
    return tuple(sorted(result))


COMMON=transport.COMMON+r'''
def phase_base(offset):
 assert offset in (10,15)
 return pathlib.Path(inputs['phases'][str(offset)])
def verify_fork(offset):
 base=phase_base(offset);v=read(base/'prepared.json');c=read(verify(v['config']));m=read(verify(v['input_manifest']))
 verify(v['sampling_script']);p=read(verify(v['fork_provenance']))
 assert c['base']==str(base) and c['extra_response_indices']==list(range(offset,offset+5)) and c['extra_response_offset']==offset
 assert c['source_commit']==inputs['collector_commit'] and c['run_id']==v['extra_run_id']==m['run_id']
 assert v['questions']==6528 and v['trajectories']==32640 and not v['original_bulk_payloads_freshly_read']
 assert p['original_bulk_payloads_freshly_read'] is False and p['same_original_questions_prompt_tokens_own_EM_and_original_seeds']
 for item in m['batches']:verify(item)
 for relative,digest in c['code_files_sha256'].items():assert sha(pathlib.Path(c['code_root'])/relative)==digest
 parentc=read(inputs['parent_phase']+'/config.json')
 assert c['run_id'] not in (parentc['run_id'],c['original_run_id']) and c['sampling_attempt_id']!=parentc['sampling_attempt_id']
 other=phase_base(15 if offset==10 else 10)/'config.json'
 if other.exists():
  peer=read(other);assert c['run_id']!=peer['run_id'] and c['sampling_attempt_id']!=peer['sampling_attempt_id']
  assert not set(c['extra_response_indices']).intersection(peer['extra_response_indices'])
 return v,c
def concise(v,c):
 return dict(status=v['status'],base=v['base'],questions=v['questions'],trajectories=v['trajectories'],run_id=c['run_id'],sampling_attempt_id=c['sampling_attempt_id'],response_indices=c['extra_response_indices'],collector_commit=c['source_commit'],collector_code_root=c['code_root'],config=v['config'],input_manifest=v['input_manifest'],fork_provenance=v['fork_provenance'],resources=v['resources'],original_bulk_payloads_freshly_read=False)
'''

INIT=r'''
b.mkdir(parents=True,exist_ok=True);guard=lock()
for n in ['logs']:(b/n).mkdir(exist_ok=True)
own={k:inputs[k] for k in ['commit','bundle_sha256','bundle_bytes','chunks','files']};p=b/'upload-controller.json'
if p.exists():assert read(p)==own,'Different immutable helper publication already exists'
else:atomic(p,own)
uploads.mkdir(exist_ok=True);print(json.dumps(dict(status='EXTRA_G5_FORK_HELPER_UPLOAD_BOUND',chunks=inputs['chunks'])))
'''
CPU=r'''
guard=lock();pub,root=publication();offset=inputs['offset'];assert offset in (10,15)
parentc=read(inputs['parent_phase']+'/config.json');assert parentc['source_commit']==inputs['collector_commit']
runtime=str(pathlib.Path(parentc['code_root'])/'reproduction/operations/qwen7b_warmup/runtime.sh')
env=dict(os.environ,QWEN7B_CODE_ROOT=str(root),CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
if not (b/'cpu-tests.json').exists():
 r=subprocess.run(['bash',runtime,'-m','unittest','reproduction.operations.qwen7b_warmup.test_extra_g5_forks'],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True,timeout=80)
 (b/'actual-cpu-tests.log').write_text(r.stdout);assert r.returncode==0,r.stdout[-3000:]
 atomic(b/'cpu-tests.json',dict(status='PASSED',code_publication=fp(b/'source-publication.json'),log=fp(b/'actual-cpu-tests.log')))
else:assert read(b/'cpu-tests.json')['code_publication']==fp(b/'source-publication.json')
args=['bash',runtime,'-u','-m','reproduction.operations.qwen7b_warmup.extra_g5_fork_prepare','--parent',inputs['parent_phase'],'--base',str(phase_base(offset)),'--offset',str(offset),'--preparer-publication',str(b/'source-publication.json')]
r=subprocess.run(args,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True,timeout=100)
(b/('prepare-offset%d.log'%offset)).write_text(r.stdout);assert r.returncode==0,r.stdout[-4000:]
v,c=verify_fork(offset);result=concise(v,c);atomic(b/('offset%d-preparation.json'%offset),result);print(json.dumps(result))
'''
SUBMIT=r'''
guard=lock();publication();offset=inputs['offset'];v,c=verify_fork(offset);base=phase_base(offset)
name='q7b-g5-m6528-o%d'%offset
if (base/'job.id').exists():
 job=(base/'job.id').read_text().strip();assert job.isdigit()
 print(json.dumps(dict(status='EXISTING_OFFSET_G5_ARRAY_PRESERVED',offset=offset,job_id=job,squeue=command(['squeue','-h','-j',job,'-o','%i|%j|%T|%R']),sacct=command(['sacct','-X','-n','-P','-j',job,'--format=JobID,State,ExitCode']))))
else:
 assert not (base/'submission-intent.json').exists(),'Prior unknown/rejected submission requires explicit reconciliation; never blindly repeat'
 assert not (base/'sampling').exists(),'Existing collection requires explicit same-identity recovery, not new submission'
 assert not command(['squeue','-h','-u','mou176','-n',name,'-o','%i|%T|%R']),'Unknown same-name job'
 intent=dict(status='OFFSET_G5_SUBMISSION_INTENT',offset=offset,job_name=name,config=v['config'],script=v['sampling_script'],resources=v['resources'],dependency=None,collector_commit=c['source_commit'],time=time.time())
 atomic(base/'submission-intent.json',intent)
 try:r=subprocess.run(['sbatch','--hold','--parsable',v['sampling_script']['path']],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=25)
 except subprocess.TimeoutExpired:
  intent['status']='SUBMIT_OUTCOME_UNKNOWN_DO_NOT_REPEAT';atomic(base/'submission-intent.json',intent);raise
 if r.returncode:
  intent.update(status='SBATCH_REJECTED',stderr=r.stderr[-2000:]);atomic(base/'submission-intent.json',intent);raise RuntimeError(r.stderr[-2000:])
 job=r.stdout.strip().split(';')[0];assert job.isdigit(),'Unknown submission reply; reconcile before retry'
 with (base/'job.id').open('x') as f:f.write(job+'\n');f.flush();os.fsync(f.fileno())
 submitted=dict(intent,status='OFFSET_G5_ARRAY_HELD',job_id=job);atomic(base/'submission.json',submitted)
 state=command(['squeue','-h','-j',job,'-o','%T|%R']);assert state and all(x.startswith('PENDING|') and 'JobHeldUser' in x for x in state.splitlines()),state
 command(['scontrol','release',job]);submitted.update(status='OFFSET_G5_ARRAY_RELEASED',time=time.time(),queue=command(['squeue','-h','-j',job,'-o','%i|%j|%T|%P|%R']))
 atomic(base/'submission.json',submitted);print(json.dumps(submitted))
'''
STATUS=r'''
result=dict(observed_epoch=time.time(),phases={});ids=[]
for offset in (10,15):
 base=phase_base(offset);row=dict(base=str(base),exists=base.exists())
 if (base/'prepared.json').exists():
  v,c=verify_fork(offset);row['prepared']=concise(v,c);row['shards']={}
  if (base/'job.id').exists():
   job=(base/'job.id').read_text().strip();assert job.isdigit();row['job_id']=job;ids.append(job)
  row['unknown_submission_outcome']=(base/'submission-intent.json').exists() and not (base/'job.id').exists()
  for shard in range(4):
   d=base/'sampling'/('shard-%02d'%shard);own={}
   for filename in ['extra-g5-adapter-controller.json','controller.json','progress.json','failure.json','complete.json','extra-g5-complete.json']:
    p=d/filename
    if p.exists():
     value=read(p);keys=['status','pid','job_id','run_id','sampling_attempt_id','response_indices','completed_batches','total_batches','questions','trajectories','epoch','elapsed','error_type','error']
     own[filename]={k:value[k] for k in keys if k in value}
   sealed=list((d/'batches').glob('*/complete.json')) if (d/'batches').exists() else [];own['sealed_batch_count']=len(sealed)
   row['shards'][str(shard)]=own
 result['phases'][str(offset)]=row
result['squeue']=command(['squeue','-h','-u','mou176','-o','%i|%j|%T|%R'])
result['sacct']=command(['sacct','-X','-n','-P','-j',','.join(ids),'--format=JobID,State,ExitCode,Elapsed']) if ids else ''
print(json.dumps(result))
'''


def remote_source(inputs,action):
    return 'inputs='+repr(inputs)+'\n'+COMMON+'\n'+dict(init=INIT,chunk=transport.CHUNK,extract=transport.EXTRACT,cpu=CPU,submit=SUBMIT,status=STATUS)[action]


def published_bundle(repo):
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
    remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/codex/qwen7b-probe-warmup-20261019'],cwd=repo,text=True).split()[0]
    if remote!=commit:raise RuntimeError('Exact helper source must first be committed and pushed')
    stream=io.BytesIO();files={}
    with gzip.GzipFile(fileobj=stream,mode='wb',mtime=0,filename='') as zipped:
        with tarfile.open(fileobj=zipped,mode='w') as archive:
            for name in source_closure(repo):
                raw=(repo/name).read_bytes()
                if raw!=subprocess.check_output(['git','show',commit+':'+name],cwd=repo):raise RuntimeError('Unpublished helper '+name)
                files[name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest());member=tarfile.TarInfo(name)
                member.size=len(raw);member.mode=0o444;member.mtime=0;archive.addfile(member,io.BytesIO(raw))
    return commit,stream.getvalue(),files


def self_test():
    repo=Path(__file__).resolve().parents[3];names=source_closure(repo)
    values=dict(base=TARGET,parent=PARENT,target=TARGET,parent_phase=PARENT_BASE,phases={str(o):target(o) for o in (10,15)},collector_commit='2b293d6e6a0be5a8b99f976b3ac0c2b15e19ee40',
        offset=10,commit='a'*40,bundle_sha256='b'*64,bundle_bytes=100000,chunks=25,files={n:dict(bytes=10000,sha256='c'*64) for n in names})
    sizes={}
    for action in ['init','chunk','extract','cpu','submit','status']:
        own=dict(values)
        if action=='chunk':own.update(chunk_index=0,chunk_data=base64.b64encode(b'x'*4096).decode(),chunk_sha256='d'*64)
        source=remote_source(own,action);compile(source,'remote-'+action,'exec');sizes[action]=len(('python3 -c '+shlex.quote(source)).encode())
        if sizes[action]>=16000:raise AssertionError('Control command exceeds16KB: '+action)
    return dict(status='LOCAL_EXTRA_G5_FORK_DEPLOYMENT_CHECKS_PASSED',files=len(names),raw_chunk_bytes=4096,command_bytes=sizes,GPU_launched=False,remote_actions_executed=False)


def main():
    p=argparse.ArgumentParser();p.add_argument('--control-root',type=Path);p.add_argument('--action',choices=['prepare','submit','status'],default='status')
    p.add_argument('--offset',type=int,choices=[10,15]);p.add_argument('--submit-authorized',action='store_true');p.add_argument('--self-test',action='store_true');args=p.parse_args()
    if args.self_test:print(json.dumps(self_test()));return
    if args.control_root is None:p.error('--control-root required')
    if args.action in ('prepare','submit') and args.offset is None:p.error('--offset required')
    if args.action=='submit' and not args.submit_authorized:p.error('Explicit --submit-authorized required')
    repo=Path(__file__).resolve().parents[3];commit,bundle,files=published_bundle(repo) if args.action=='prepare' else ('',b'',{})
    chunks=[bundle[i:i+4096] for i in range(0,len(bundle),4096)]
    inputs=dict(base=TARGET,parent=PARENT,target=TARGET,parent_phase=PARENT_BASE,phases={str(o):target(o) for o in (10,15)},collector_commit='2b293d6e6a0be5a8b99f976b3ac0c2b15e19ee40',
        offset=args.offset,commit=commit,bundle_sha256=hashlib.sha256(bundle).hexdigest(),bundle_bytes=len(bundle),chunks=len(chunks),files=files)
    spec=importlib.util.spec_from_file_location('offset_extra_control',args.control_root/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py')
    control=importlib.util.module_from_spec(spec);spec.loader.exec_module(control)
    address=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0'],text=True).strip()
    def call(action,extra=None):
        own=dict(inputs);own.update(extra or {});command='python3 -c '+shlex.quote(remote_source(own,action));assert len(command.encode())<16000
        r=control.run(command,account='mou176',source_address=address,timeout=160 if action=='cpu' else 55)
        rows=[json.loads(v) for v in r.stdout.decode(errors='replace').splitlines() if v.startswith('{')]
        if r.returncode or not rows:raise RuntimeError('Fork action unconfirmed; reconcile actual state: '+r.stderr.decode(errors='replace')[-1000:]+' '+r.stdout.decode(errors='replace')[-3000:])
        print(json.dumps(rows[-1]),flush=True);return rows[-1]
    if args.action=='prepare':
        call('init')
        for i,part in enumerate(chunks):call('chunk',dict(chunk_index=i,chunk_data=base64.b64encode(part).decode(),chunk_sha256=hashlib.sha256(part).hexdigest()))
        call('extract');result=call('cpu')
    else:result=call(args.action)
    dest=args.control_root/'outputs/qwen7b-warmup-20261019'/('extraG5-fork-'+args.action+('-offset%d'%args.offset if args.offset else '')+'.json')
    dest.parent.mkdir(parents=True,exist_ok=True);dest.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
