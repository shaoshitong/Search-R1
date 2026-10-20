"""Independent code-only staging and guarded one-GPU new fixed-G5 validation replay."""
import argparse,base64,gzip,hashlib,importlib.util,io,json,shlex,subprocess,tarfile
from pathlib import Path
from . import deploy_peer_newseed_replay as legacy
from .validation_fixedg5_replay import TARGET,V
P='reproduction/operations/qwen7b_warmup/'
FILES=tuple(dict.fromkeys(legacy.FILES+tuple(P+n for n in ('validation_fixedg5_replay.py','test_validation_fixedg5_replay.py','deploy_validation_fixedg5_replay.py','replay_batched_context.py'))))

def render_sbatch(base,code,selection_sha):
 if Path(base)!=TARGET or Path(code).parent!=TARGET or len(selection_sha)!=64:raise ValueError('Independent fixed validation capture scope required')
 return '''#!/usr/bin/env bash
#SBATCH --job-name=q7b-newval-fixed5-hfreplay
#SBATCH --partition=acd_u
#SBATCH --account=mou176
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=BASE/logs/capture-%j.log
set -euo pipefail
export QWEN7B_CODE_ROOT=CODEQ
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
python3 - CODEQ <<'VERIFY'
import hashlib,json,pathlib,sys
r=pathlib.Path(sys.argv[1]);p=json.loads((r.parent/'source-publication.json').read_text());assert str(r)==p['code_root']
for n,v in p['files'].items():
 q=r/n;assert q.stat().st_size==v['bytes'] and hashlib.sha256(q.read_bytes()).hexdigest()==v['sha256'],n
VERIFY
srun --kill-on-bad-exit=1 bash "$QWEN7B_CODE_ROOT/reproduction/operations/qwen7b_warmup/runtime.sh" -u -m reproduction.operations.qwen7b_warmup.validation_fixedg5_replay --action capture --selection BASEQ/inputs/selection.json --selection-sha256 SELSHA --output BASEQ/capture
'''.replace('BASEQ',shlex.quote(str(base))).replace('BASE',str(base)).replace('CODEQ',shlex.quote(str(code))).replace('SELSHA',selection_sha)

CPU=r'''
guard=lock();pub,root=publication();env=dict(os.environ,QWEN7B_CODE_ROOT=str(root),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
r=subprocess.run(runtime(root,['unittest','reproduction.operations.qwen7b_warmup.test_validation_fixedg5_replay']),env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True,timeout=90)
(b/'actual-cpu-tests.log').write_text(r.stdout);assert r.returncode==0,r.stdout[-2000:]
atomic(b/'cpu-qualification.json',dict(status='ACTUAL_METADATA_ONLY_FIXED_G5_6_TESTS_PASSED',log=fp(b/'actual-cpu-tests.log'),model_tensors_loaded=False,GPU_started=False));print(json.dumps(read(b/'cpu-qualification.json')))
'''
PREPARE=r'''
guard=lock();pub,root=publication();gold=pathlib.Path(inputs['golden']);assert gold.is_file() and inputs['batch_size'] in (16,32)
finish=b/'preparation.json';receipt=b/'cpu-prepare-launch.json'
if finish.exists():print(json.dumps(dict(status='EXISTING_CPU_PREPARATION_PRESERVED',preparation=fp(finish))))
elif receipt.exists():
 v=read(receipt);print(json.dumps(dict(status='EXISTING_CPU_PREPARATION_LAUNCH_PRESERVED',launch=v,same_process_birth=ticks(v['pid'])==v['start_ticks'])))
else:
 assert not (b/'cpu-prepare-intent.json').exists(),'Unknown previous prepare outcome; inspect before retry'
 intent=dict(status='CPU_PREPARE_INTENT',source_publication=fp(b/'source-publication.json'),golden=fp(gold),batch_size=inputs['batch_size'],epoch=time.time());atomic(b/'cpu-prepare-intent.json',intent)
 args=runtime(root,['reproduction.operations.qwen7b_warmup.validation_fixedg5_replay','--action','prepare','--base',str(b),'--code-root',str(root),'--commit',pub['commit'],'--golden',str(gold),'--batch-size',str(inputs['batch_size'])])
 env=dict(os.environ,QWEN7B_CODE_ROOT=str(root),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
 with (b/'cpu-prepare.log').open('ab') as log:child=subprocess.Popen(args,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 value=dict(intent,status='CPU_PREPARE_STARTED_NOT_VERIFIED',pid=child.pid,start_ticks=ticks(child.pid));atomic(receipt,value);print(json.dumps(value))
'''
FINALIZE=r'''
guard=lock();pub,root=publication();done=read(b/'preparation.json');assert done['source_commit']==pub['commit'] and done['questions']==2048 and done['trajectories']==10240
assert fp(done['selection']['path'])==done['selection'];script=render_sbatch(str(b),str(root),done['selection']['sha256']);p=b/'capture.sbatch'
if p.exists():assert p.read_text()==script
else:p.write_text(script)
value=dict(status='NEW_FIXED_VALIDATION_G5_ONE_GPU_CAPTURE_READY_NOT_SUBMITTED',preparation=fp(b/'preparation.json'),source_publication=fp(b/'source-publication.json'),variants={'0':dict(base=str(b),job_name='q7b-newval-fixed5-hfreplay',selection=done['selection'],sbatch=fp(p))},no_dependency=True)
atomic(b/'submission-plan.json',value);print(json.dumps(value))
'''
STATUS=r'''
result=dict(status='ACTUAL_READ_ONLY_NEW_FIXED_G5_HF_CAPTURE_OBSERVATION',epoch=time.time(),base=str(b))
for name in ['cpu-qualification.json','cpu-prepare-launch.json','prepare-progress.json','preparation.json']:
 p=b/name
 if p.exists():
  v=read(p);result[name]={k:v[k] for k in ['status','pid','start_ticks','questions','trajectories','expected_trajectories','selection','batching','source_complete','source_fixed_G5'] if k in v}
  if 'pid' in v:result[name]['same_process_birth']=ticks(v['pid'])==v['start_ticks']
p=b/'cpu-prepare.log'
if p.exists():result['CPU_log_tail']=p.read_text(errors='replace')[-1000:]
result['complete_chunks']=len(list((b/'capture').glob('chunk-*/complete.json')))
result['actual_capture_bytes']=sum(p.stat().st_size for p in (b/'capture').glob('chunk-*/layer-26/*') if p.is_file())
for name in ['progress.json','failure.json','complete.json']:
 p=b/'capture'/name
 if p.exists():
  v=read(p);result[name]={k:v[k] for k in ['status','trajectories','questions','expected_trajectories','elapsed_seconds','features','source_complete','source_fixed_G5','fixed_G5_sha256','error_type','error','time'] if k in v}
  if name=='complete.json':result['complete_ref']=fp(p)
if (b/'job.id').exists():
 job=(b/'job.id').read_text().strip();result['job_id']=job;result['squeue']=command(['squeue','-h','-j',job,'-o','%i|%j|%T|%R|%N']);result['sacct']=command(['sacct','-X','-n','-P','-j',job,'--format=JobID,JobName,State,ExitCode,Elapsed'])
result['logs']={p.name:p.read_text(errors='replace')[-1000:] for p in (b/'logs').glob('*.log')}
print(json.dumps(result))
'''

def render(inputs,action):
 own=dict(inputs)
 if action not in ('init','extract'):own.pop('files',None)
 if action!='chunk':
  for k in ('chunk_data','chunk_index','chunk_sha256'):own.pop(k,None)
 extra=''
 if action=='finalize':
  import inspect
  extra=inspect.getsource(render_sbatch)+'\n'
 codes={'init':legacy.INIT,'chunk':legacy.CHUNK,'extract':legacy.EXTRACT,'cpu':CPU,'prepare':PREPARE,'finalize':FINALIZE,'submit':legacy.SUBMIT,'status':STATUS}
 if action not in codes:raise ValueError('Unknown explicit fixed-G5 control action')
 return 'inputs='+repr(own)+'\n'+legacy.COMMON+extra+codes[action]

def bundle(repo):
 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo).decode().strip();remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/codex/qwen7b-probe-warmup-20261019'],cwd=repo).split()[0].decode()
 if commit!=remote:raise ValueError('Publish exact capture helper/runtime first')
 stream=io.BytesIO();files={}
 with gzip.GzipFile(fileobj=stream,mode='wb',mtime=0,filename='') as zipped:
  with tarfile.open(fileobj=zipped,mode='w') as archive:
   for name in FILES:
    data=(repo/name).read_bytes()
    if data!=subprocess.check_output(['git','show',commit+':'+name],cwd=repo):raise ValueError('Unpublished runtime source '+name)
    files[name]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest());item=tarfile.TarInfo(name);item.size=len(data);item.mode=0o444;archive.addfile(item,io.BytesIO(data))
 return commit,stream.getvalue(),files

def self_test():
 values=dict(base=str(TARGET),target=str(TARGET),receiver=str(V),ingest=str(TARGET),commit='a'*40,bundle_sha256='b'*64,bundle_bytes=20000,chunks=5,files={n:dict(bytes=10000,sha256='c'*64) for n in FILES},shard=0,golden=str(TARGET/'golden-reference.json'),batch_size=32,chunk_data=base64.b64encode(b'x'*4096).decode(),chunk_sha256='d'*64,chunk_index=0)
 lengths={}
 for action in ['init','chunk','extract','cpu','prepare','finalize','submit','status']:
  source=render(values,action);compile(source,'control-'+action,'exec');lengths[action]=len(('python3 -c '+shlex.quote(source)).encode())
  if lengths[action]>=16000:raise ValueError('Control command exceeds16KB: '+action)
 script=render_sbatch(str(TARGET),str(TARGET/'code-aaaaaaaaaaaa'),'e'*64);subprocess.run(['bash','-n'],input=script,text=True,check=True)
 assert '--gres=gpu:1' in script and '--qos=normal' in script and '--dependency' not in script
 return dict(status='LOCAL_NEW_FIXED_G5_DEPLOYMENT_RENDER_CHECKS_PASSED',source_files=len(FILES),control_command_bytes=lengths,remote_actions_executed=False)

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--control-root',type=Path);p.add_argument('--action',choices=('stage','prepare','finalize','submit','status'),default='status');p.add_argument('--golden');p.add_argument('--batch-size',type=int,choices=(16,32));p.add_argument('--submit-authorized',action='store_true');p.add_argument('--self-test',action='store_true');a=p.parse_args()
 if a.self_test:print(json.dumps(self_test()));return
 if a.control_root is None:p.error('--control-root required')
 if a.action=='prepare' and (not a.golden or not a.batch_size):p.error('Actual completed golden and explicit qualified batch required')
 if a.action=='submit' and not a.submit_authorized:p.error('Explicit --submit-authorized required')
 repo=Path(__file__).resolve().parents[3];commit,data,files=bundle(repo) if a.action=='stage' else ('',b'',{})
 chunks=[data[i:i+4096] for i in range(0,len(data),4096)];inputs=dict(base=str(TARGET),target=str(TARGET),receiver=str(V),ingest=str(TARGET),commit=commit,bundle_sha256=hashlib.sha256(data).hexdigest(),bundle_bytes=len(data),chunks=len(chunks),files=files,shard=0,golden=a.golden,batch_size=a.batch_size)
 spec=importlib.util.spec_from_file_location('fixed_validation_capture_control',a.control_root/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');control=importlib.util.module_from_spec(spec);spec.loader.exec_module(control);address=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0'],text=True).strip()
 def call(action,extra=None):
  own=dict(inputs);own.update(extra or {});source=render(own,action);cmd='python3 -c '+shlex.quote(source)
  if len(cmd.encode())>=16000:raise ValueError('Control command exceeds16KB')
  response=control.run(cmd,account='mou176',source_address=address,timeout=100 if action=='cpu' else 55);values=[json.loads(line) for line in response.stdout.decode(errors='replace').splitlines() if line.startswith('{')]
  if response.returncode or not values:raise RuntimeError('Remote action not confirmed; inspect before repeating: '+response.stderr.decode(errors='replace')[-1000:]+' '+response.stdout.decode(errors='replace')[-1500:])
  print(json.dumps(values[-1]),flush=True);return values[-1]
 if a.action=='stage':
  call('init')
  for i,chunk in enumerate(chunks):call('chunk',dict(chunk_index=i,chunk_data=base64.b64encode(chunk).decode(),chunk_sha256=hashlib.sha256(chunk).hexdigest()))
  call('extract');value=call('cpu')
 else:value=call(a.action)
 out=a.control_root/'outputs/qwen7b-warmup-20261019'/('new-validation-fixedG5-replay-'+a.action+'.json');out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(value,indent=2)+'\n')
if __name__=='__main__':main()
