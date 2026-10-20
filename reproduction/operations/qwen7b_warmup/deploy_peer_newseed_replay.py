"""Code-only stage, guarded CPU ingestion/projection, and four1GPU replay jobs.

Every action is explicit. No dependency, automatic fallback, duplicate submit,
model download, Mac payload transport, retriever or actor/probe training.
"""
import argparse,base64,gzip,hashlib,importlib.util,io,json,shlex,subprocess,tarfile
from pathlib import Path
from .peer_newseed_replay import TARGET,B,A

SNAPSHOT='newseed-20261021-185747-445fc691'
RECEIVER=B/'peer-newseed-h100-20261021'/SNAPSHOT
INGEST=A/('peer-newseed-ingest-'+SNAPSHOT)
PACKAGE='reproduction/operations/qwen7b_warmup/'
# Explicit runtime import closure: CPU scorer additionally needs the exact
# production strict_label, signals, qa_em and verl package initialization.
FILES=tuple(PACKAGE+n for n in ('peer_newseed_replay.py','peer_newseed_ingest.py',
 'peer_newseed_receive.py','peer_newseed_export.py','peer_newseed_audit_reference.py',
 'replay_probe_layers.py','replay_bank_prepare.py','replay_bank_capture.py','replay_feature_reader.py','runtime.sh',
 'test_peer_newseed_ingest.py','test_peer_newseed_replay.py','test_peer_newseed_receive.py','deploy_peer_newseed_replay.py'))+(
 'reproduction/prefix20_online/rollout.py','reproduction/prefix20_online/pool.py',
 'search_g1/__init__.py','search_g1/signals.py','search_r1/__init__.py','search_r1/qa_protocol.py',
 'verl/__init__.py','verl/protocol.py','verl/version/version','verl/utils/__init__.py',
 'verl/utils/logging_utils.py','verl/utils/tokenizer.py','verl/utils/py_functional.py','verl/utils/reward_score/__init__.py','verl/utils/reward_score/qa_em.py')


def render_sbatch(shard,base,code,selection_sha):
 if type(shard) is not int or not 0<=shard<4 or Path(base)!=TARGET/('shard-%02d'%shard) or Path(code).parent!=TARGET:
  raise ValueError('Four explicit independent feature shards required')
 if len(selection_sha)!=64:raise ValueError('Actual prepared selection SHA required')
 return '''#!/usr/bin/env bash
#SBATCH --job-name=q7b-peer8192-replay-SHARD
#SBATCH --partition=acd_u
#SBATCH --account=mou176
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=BASE/logs/replay-%j.log
set -euo pipefail
export QWEN7B_CODE_ROOT=CODEQ
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
python3 - CODEQ <<'VERIFY'
import hashlib,json,pathlib,sys
r=pathlib.Path(sys.argv[1]);p=json.loads((r.parent/'source-publication.json').read_text())
assert str(r)==p['code_root']
for n,v in p['files'].items():
 q=r/n;assert q.stat().st_size==v['bytes'] and hashlib.sha256(q.read_bytes()).hexdigest()==v['sha256'],n
VERIFY
srun --kill-on-bad-exit=1 bash "$QWEN7B_CODE_ROOT/reproduction/operations/qwen7b_warmup/runtime.sh" -u -m reproduction.operations.qwen7b_warmup.peer_newseed_replay --action capture --selection BASEQ/inputs/selection.json --selection-sha256 SELSHA --output BASEQ/capture
'''.replace('SHARD','%02d'%shard).replace('BASEQ',shlex.quote(str(base))).replace('BASE',str(base)).replace('CODEQ',shlex.quote(str(code))).replace('SELSHA',selection_sha)


COMMON=r'''
import base64,fcntl,gzip,hashlib,io,json,os,pathlib,shlex,subprocess,tarfile,time
b=pathlib.Path(inputs['base']);assert str(b)==inputs['target'] and b.resolve()==b
code=b/('code-'+inputs['commit'][:12]);uploads=b/('upload-'+inputs['bundle_sha256'][:16])
def read(p):return json.loads(pathlib.Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb') as f:
  for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
 return h.hexdigest()
def fp(p):
 p=pathlib.Path(p);return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def atomic(p,v):
 p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+'.%d.tmp'%os.getpid())
 with q.open('w') as f:json.dump(v,f,sort_keys=True);f.write(chr(10));f.flush();os.fsync(f.fileno())
 q.replace(p)
def lock():
 b.mkdir(parents=True,exist_ok=True);f=(b/'operator.lock').open('a');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);return f
def publication():
 p=read(b/'source-publication.json');root=pathlib.Path(p['code_root']);assert root.parent==b
 for name,v in p['files'].items():assert fp(root/name)==dict(v,path=str(root/name)),name
 return p,root
def command(args,timeout=25):
 r=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=timeout)
 assert r.returncode==0,(r.stdout[-1000:],r.stderr[-2000:]);return r.stdout.strip()
def ticks(pid):
 try:return pathlib.Path('/proc/%d/stat'%pid).read_text().rsplit(')',1)[1].split()[19]
 except FileNotFoundError:return None
def runtime(root,args):
 return ['bash',str(root/'reproduction/operations/qwen7b_warmup/runtime.sh'),'-u','-m']+args
'''
INIT=r'''
guard=lock();(b/'logs').mkdir(exist_ok=True);uploads.mkdir(exist_ok=True)
value={key:inputs[key] for key in ['commit','bundle_sha256','bundle_bytes','chunks','files']}
p=b/'upload-controller.json'
if p.exists() and read(p)!=value:
 assert not (b/'submission-plan.json').exists() and not (b/'preparation.json').exists(),'Existing GPU/feature preparation; immutable source preserved'
 assert not list(b.glob('cpu-*-launch.json')) and not list(pathlib.Path(inputs['ingest']).glob('cpu-*-launch.json')),'A CPU writer has already launched; no source handoff'
 old=read(p);atomic(b/'source-history'/('upload-'+old['commit']+'.json'),old);atomic(p,value)
elif not p.exists():atomic(p,value)
print(json.dumps(dict(status='CODE_STAGE_BOUND',chunks=inputs['chunks'])))
'''
CHUNK=r'''
guard=lock();assert read(b/'upload-controller.json')['commit']==inputs['commit']
x=base64.b64decode(inputs['chunk_data']);assert len(x)<=4096 and hashlib.sha256(x).hexdigest()==inputs['chunk_sha256']
p=uploads/('part-%05d'%inputs['chunk_index'])
if p.exists():assert p.read_bytes()==x
else:
 with p.open('xb') as f:f.write(x);f.flush();os.fsync(f.fileno())
print(json.dumps(dict(status='CODE_CHUNK_SHA_ACCEPTED',index=inputs['chunk_index'])))
'''
EXTRACT=r'''
guard=lock();parts=sorted(uploads.glob('part-*'));assert len(parts)==inputs['chunks'];data=b''.join(p.read_bytes() for p in parts)
assert len(data)==inputs['bundle_bytes'] and hashlib.sha256(data).hexdigest()==inputs['bundle_sha256']
if not code.exists():
 stage=code.with_name(code.name+'.staging');stage.mkdir(exist_ok=True)
 with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as tar:
  members=tar.getmembers();assert {m.name for m in members}==set(inputs['files'])
  for member in members:
   rel=pathlib.Path(member.name);assert member.isfile() and not rel.is_absolute() and '..' not in rel.parts
   x=tar.extractfile(member).read();assert hashlib.sha256(x).hexdigest()==inputs['files'][member.name]['sha256']
   p=stage/rel;p.parent.mkdir(parents=True,exist_ok=True)
   if p.exists():assert p.read_bytes()==x
   else:p.write_bytes(x)
   p.chmod(0o444)
 stage.rename(code)
value=dict(commit=inputs['commit'],remote_sha=inputs['commit'],code_root=str(code),files=inputs['files'])
p=b/'source-publication.json'
if p.exists() and read(p)!=value:
 assert not (b/'submission-plan.json').exists() and not (b/'preparation.json').exists()
 assert not list(b.glob('cpu-*-launch.json')) and not list(pathlib.Path(inputs['ingest']).glob('cpu-*-launch.json'))
 old=read(p);atomic(b/'source-history'/('publication-'+old['commit']+'.json'),old);atomic(p,value)
elif not p.exists():atomic(p,value)
publication();print(json.dumps(dict(status='PUBLISHED_CODE_ALL_SHA_ACCEPTED',code_root=str(code),files=len(inputs['files']))))
'''
CPU=r'''
guard=lock();pub,root=publication();env=dict(os.environ,QWEN7B_CODE_ROOT=str(root),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
args=runtime(root,['unittest','reproduction.operations.qwen7b_warmup.test_peer_newseed_ingest','reproduction.operations.qwen7b_warmup.test_peer_newseed_replay'])
r=subprocess.run(args,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True,timeout=90)
(b/'actual-cpu-tests.log').write_text(r.stdout);assert r.returncode==0,r.stdout[-3000:]
smoke="from reproduction.prefix20_online.rollout import strict_label; from reproduction.operations.qwen7b_warmup.test_peer_newseed_ingest import CharTokenizer; x=list(map(ord,'<answer>Paris</answer>')); assert strict_label(CharTokenizer(),x,[1]*len(x),{'target':['Paris']})==1; assert strict_label(CharTokenizer(),x,[0]*len(x),{'target':['Paris']})==0; print('UNCHANGED_PRODUCTION_STRICT_LABEL_IMPORT_AND_MASK_PASS')"
r=subprocess.run(['bash',str(root/'reproduction/operations/qwen7b_warmup/runtime.sh'),'-c',smoke],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True,timeout=60)
(b/'actual-production-strict-EM-smoke.log').write_text(r.stdout);assert r.returncode==0,r.stdout[-3000:]
atomic(b/'cpu-qualification.json',dict(status='ACTUAL_REMOTE_CPU_11_TESTS_PASSED',log=fp(b/'actual-cpu-tests.log'),production_strict_label_smoke=fp(b/'actual-production-strict-EM-smoke.log'),model_tensors_loaded=False,feature_capture_started=False))
print(json.dumps(read(b/'cpu-qualification.json')))
'''
INGEST_PREPARE=r'''
guard=lock();pub,root=publication();receiver=pathlib.Path(inputs['receiver']);out=pathlib.Path(inputs['ingest'])
if not (receiver/'complete.json').exists():print(json.dumps(dict(status='NOT_READY_TARGET_RECEIVER_COMPLETE_ABSENT')))
else:
 env=dict(os.environ,QWEN7B_CODE_ROOT=str(root),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2')
 args=runtime(root,['reproduction.operations.qwen7b_warmup.peer_newseed_ingest','--action','prepare','--receiver',str(receiver),'--output',str(out)])
 r=subprocess.run(args,env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True,timeout=90)
 (b/'ingest-prepare.log').write_text(r.stdout);assert r.returncode==0,r.stdout[-3000:]
 print(json.dumps(dict(status='INGEST_PREPARED',config=fp(out/'config.json'))))
'''
START=r'''
guard=lock();pub,root=publication();mode=inputs['cpu_mode'];assert mode in ['ingest','features']
out=pathlib.Path(inputs['ingest']) if mode=='ingest' else b;receipt=out/('cpu-'+mode+'-launch.json')
finish=out/'complete.json' if mode=='ingest' else b/'preparation.json'
if finish.exists():print(json.dumps(dict(status='ALREADY_COMPLETE_CPU_STAGE_PRESERVED',mode=mode,complete=fp(finish))))
elif receipt.exists():
 value=read(receipt);print(json.dumps(dict(status='EXISTING_CPU_LAUNCH_PRESERVED',mode=mode,launch=value,same_process_birth=ticks(value['pid'])==value['start_ticks'])))
else:
 assert not (out/('cpu-'+mode+'-intent.json')).exists(),'Unknown prior launch outcome; do not repeat'
 if mode=='ingest':
  assert (out/'config.json').exists();args=runtime(root,['reproduction.operations.qwen7b_warmup.peer_newseed_ingest','--action','ingest','--config',str(out/'config.json')])
 else:
  assert (pathlib.Path(inputs['ingest'])/'complete.json').exists();args=runtime(root,['reproduction.operations.qwen7b_warmup.peer_newseed_replay','--action','prepare','--raw-complete',str(pathlib.Path(inputs['ingest'])/'complete.json'),'--code-root',str(root),'--source-commit',pub['commit']])
 env=dict(os.environ,QWEN7B_CODE_ROOT=str(root),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
 intent=dict(mode=mode,status='CPU_LAUNCH_INTENT_NOT_SUCCESS',source_publication=fp(b/'source-publication.json'),epoch=time.time());atomic(out/('cpu-'+mode+'-intent.json'),intent)
 with (out/('cpu-'+mode+'.log')).open('ab') as log:child=subprocess.Popen(args,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 value=dict(intent,status='CPU_STARTED_NOT_PROGRESS_VERIFIED',pid=child.pid,start_ticks=ticks(child.pid));atomic(receipt,value);print(json.dumps(value))
'''
FINALIZE=r'''
guard=lock();pub,root=publication();done=read(b/'preparation.json');assert done['source_commit']==pub['commit'] and done['questions']==8192 and done['trajectories']==81920
variants={}
for shard,item in enumerate(done['selections']):
 assert fp(item['path'])==item;base=b/('shard-%02d'%shard);script=render_sbatch(shard,str(base),str(root),item['sha256']);p=base/'capture.sbatch'
 if p.exists():assert p.read_text()==script
 else:p.write_text(script)
 variants[str(shard)]=dict(base=str(base),job_name='q7b-peer8192-replay-%02d'%shard,selection=item,sbatch=fp(p))
value=dict(status='FOUR_FEATURE_ONLY_1GPU_REPLAY_SCRIPTS_READY_NOT_SUBMITTED',variants=variants,preparation=fp(b/'preparation.json'),source_publication=fp(b/'source-publication.json'),no_dependency=True)
atomic(b/'submission-plan.json',value);print(json.dumps(value))
'''
SUBMIT=r'''
guard=lock();pub,root=publication();plan=read(b/'submission-plan.json');item=plan['variants'][str(inputs['shard'])];base=pathlib.Path(item['base'])
assert fp(item['selection']['path'])==item['selection'] and fp(item['sbatch']['path'])==item['sbatch']
if (base/'job.id').exists():print(json.dumps(dict(status='EXISTING_FEATURE_JOB_PRESERVED',job_id=(base/'job.id').read_text().strip())))
else:
 assert not (base/'submission-intent.json').exists(),'Unknown submit outcome; reconcile before another attempt'
 assert not (base/'capture').exists(),'Existing feature output needs inspection; do not duplicate'
 assert not command(['squeue','-h','-u','mou176','-n',item['job_name'],'-o','%i|%T|%R']),'Unknown same-name job'
 intent=dict(status='FEATURE_SUBMIT_INTENT',shard=inputs['shard'],job_name=item['job_name'],script=item['sbatch'],selection=item['selection'],partition='acd_u',qos='normal',gpus=1,cpus=8,memory='64G',time='02:00:00',dependency=None,epoch=time.time())
 atomic(base/'submission-intent.json',intent)
 try:r=subprocess.run(['sbatch','--hold','--parsable',item['sbatch']['path']],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=25)
 except subprocess.TimeoutExpired:
  atomic(base/'submission-unknown.json',dict(status='UNKNOWN_SUBMIT_DO_NOT_REPEAT',epoch=time.time()));raise
 if r.returncode:
  atomic(base/'submission-rejected.json',dict(status='SBATCH_REJECTED',stderr=r.stderr[-1500:]));raise RuntimeError(r.stderr[-1500:])
 job=r.stdout.strip().split(';')[0];assert job.isdigit(),'Unknown submission reply'
 with (base/'job.id').open('x') as f:f.write(job+'\n');f.flush();os.fsync(f.fileno())
 state=command(['squeue','-h','-j',job,'-o','%T|%R']);assert state.startswith('PENDING|') and 'JobHeldUser' in state,state
 command(['scontrol','release',job]);value=dict(intent,status='FEATURE_JOB_RELEASED_NOT_PROGRESS_VERIFIED',job_id=job)
 atomic(base/'submission.json',value);print(json.dumps(value))
'''
STATUS=r'''
result=dict(status='ACTUAL_READ_ONLY_FEATURE_INGEST_OBSERVATION',epoch=time.time(),base=str(b),receiver_complete=(pathlib.Path(inputs['receiver'])/'complete.json').exists(),stages={})
for label,out in [('ingest',pathlib.Path(inputs['ingest'])),('features',b)]:
 own={}
 for name in ['progress.json','prepare-progress.json','failure.json']:
  if (out/name).exists():own[name]=read(out/name)
 p=out/('cpu-'+label+'-launch.json')
 if p.exists():
  v=read(p);own['launch']=v;own['same_process_birth']=ticks(v['pid'])==v['start_ticks']
  q=out/('cpu-'+label+'.log')
  if q.exists():own['log_tail']=q.read_text(errors='replace')[-1500:]
 p=out/'complete.json' if label=='ingest' else out/'preparation.json'
 if p.exists():
  v=read(p);own['complete']={k:v[k] for k in ['status','questions','trajectories','elapsed_seconds','free_space_bytes'] if k in v}
 result['stages'][label]=own
result['shards']={};ids=[]
for i in range(4):
 base=b/('shard-%02d'%i);own={'complete_chunks':len(list((base/'capture').glob('chunk-*/complete.json')))}
 if (base/'job.id').exists():own['job_id']=(base/'job.id').read_text().strip();ids.append(own['job_id'])
 for name in ['progress.json','failure.json','complete.json']:
  p=base/'capture'/name
  if p.exists():
   v=read(p);own[name]={k:v[k] for k in ['status','questions','trajectories','expected_trajectories','causal_prefill_passes_this_process','elapsed_seconds','max_cuda_memory_allocated','time'] if k in v}
 own['logs']={q.name:q.read_text(errors='replace')[-1300:] for q in (base/'logs').glob('*.log')}
 result['shards'][str(i)]=own
if ids:
 result['squeue']=command(['squeue','-h','-j',','.join(ids),'-o','%i|%j|%T|%R|%N'])
 result['sacct']=command(['sacct','-X','-n','-P','-j',','.join(ids),'--format=JobID,JobName,State,ExitCode,Elapsed'])
print(json.dumps(result))
'''


def render(inputs,action):
 own=dict(inputs)
 if action not in ('init','extract'):own.pop('files',None)
 if action!='chunk':
  for key in ('chunk_data','chunk_sha256','chunk_index'):own.pop(key,None)
 extra=''
 if action=='finalize':
  import inspect
  extra='Path=pathlib.Path\nTARGET=pathlib.Path('+repr(str(TARGET))+')\n'+inspect.getsource(render_sbatch)+'\n'
 return 'inputs='+repr(own)+'\n'+COMMON+'\n'+extra+{'init':INIT,'chunk':CHUNK,'extract':EXTRACT,'cpu':CPU,'ingest-prepare':INGEST_PREPARE,'start':START,'finalize':FINALIZE,'submit':SUBMIT,'status':STATUS}[action]


def bundle(repo):
 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
 actual=subprocess.check_output(['git','ls-remote','origin','refs/heads/codex/qwen7b-probe-warmup-20261019'],cwd=repo,text=True).split()[0]
 if commit!=actual:raise ValueError('Publish exact current helper/runtime first')
 stream=io.BytesIO();files={}
 with gzip.GzipFile(fileobj=stream,mode='wb',mtime=0,filename='') as zipped:
  with tarfile.open(fileobj=zipped,mode='w') as archive:
   for name in FILES:
    data=(repo/name).read_bytes()
    if data!=subprocess.check_output(['git','show',commit+':'+name],cwd=repo):raise ValueError('Unpublished runtime source '+name)
    files[name]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest());item=tarfile.TarInfo(name);item.size=len(data);item.mode=0o444;archive.addfile(item,io.BytesIO(data))
 return commit,stream.getvalue(),files


def self_test():
 source=dict(base=str(TARGET),target=str(TARGET),receiver=str(RECEIVER),ingest=str(INGEST),commit='a'*40,bundle_sha256='b'*64,bundle_bytes=10000,chunks=3,
  files={name:dict(bytes=10000,sha256='c'*64) for name in FILES},shard=0,cpu_mode='ingest',chunk_data=base64.b64encode(b'x'*4096).decode(),chunk_sha256='d'*64,chunk_index=0)
 lengths={}
 for action in ['init','chunk','extract','cpu','ingest-prepare','start','finalize','submit','status']:
  text=render(source,action);compile(text,'control-'+action,'exec');lengths[action]=len(('python3 -c '+shlex.quote(text)).encode())
  if lengths[action]>=16000:raise ValueError('16KB control command exceeded: '+action)
 for i in range(4):
  script=render_sbatch(i,str(TARGET/('shard-%02d'%i)),str(TARGET/'code-aaaaaaaaaaaa'),'e'*64)
  subprocess.run(['bash','-n'],input=script,text=True,check=True)
  assert '--gres=gpu:1' in script and '--dependency' not in script and '--qos=normal' in script
 return dict(status='LOCAL_FEATURE_DEPLOYMENT_RENDER_CHECKS_PASSED',code_files=len(FILES),control_command_bytes=lengths,remote_actions_executed=False)


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--control-root',type=Path);p.add_argument('--action',choices=('stage','status','ingest-prepare','ingest-start','features-start','finalize','submit'),default='status');p.add_argument('--shard',type=int);p.add_argument('--submit-authorized',action='store_true');p.add_argument('--self-test',action='store_true');a=p.parse_args()
 if a.self_test:print(json.dumps(self_test()));return
 if a.control_root is None:p.error('--control-root required')
 if a.action=='submit' and (a.shard not in range(4) or not a.submit_authorized):p.error('Explicit shard0..3 and --submit-authorized required')
 repo=Path(__file__).resolve().parents[3];commit,data,files=bundle(repo) if a.action=='stage' else ('',b'',{})
 chunks=[data[i:i+4096] for i in range(0,len(data),4096)]
 inputs=dict(base=str(TARGET),target=str(TARGET),receiver=str(RECEIVER),ingest=str(INGEST),commit=commit,bundle_sha256=hashlib.sha256(data).hexdigest(),bundle_bytes=len(data),chunks=len(chunks),files=files,shard=a.shard)
 spec=importlib.util.spec_from_file_location('peer_feature_control',a.control_root/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');control=importlib.util.module_from_spec(spec);spec.loader.exec_module(control)
 address=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0'],text=True).strip()
 def call(action,extra=None):
  own=dict(inputs);own.update(extra or {});code=render(own,action);command='python3 -c '+shlex.quote(code)
  if len(command.encode())>=16000:raise ValueError('16KB control command bound exceeded')
  result=control.run(command,account='mou176',source_address=address,timeout=100 if action in ('cpu','ingest-prepare') else 55)
  values=[json.loads(line) for line in result.stdout.decode(errors='replace').splitlines() if line.startswith('{')]
  if result.returncode or not values:raise RuntimeError('Remote action not confirmed; inspect before repeating: '+result.stderr.decode(errors='replace')[-1000:]+' '+result.stdout.decode(errors='replace')[-2000:])
  print(json.dumps(values[-1]),flush=True);return values[-1]
 if a.action=='stage':
  call('init')
  for i,chunk in enumerate(chunks):call('chunk',dict(chunk_index=i,chunk_data=base64.b64encode(chunk).decode(),chunk_sha256=hashlib.sha256(chunk).hexdigest()))
  call('extract');result=call('cpu')
 elif a.action.endswith('-start'):result=call('start',dict(cpu_mode=a.action.split('-')[0]))
 else:result=call(a.action)
 out=a.control_root/'outputs/qwen7b-warmup-20261019'/('peer-newseed-feature-'+a.action+('-'+str(a.shard) if a.shard is not None else '')+'.json');out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(result,indent=2)+'\n')
if __name__=='__main__':main()
