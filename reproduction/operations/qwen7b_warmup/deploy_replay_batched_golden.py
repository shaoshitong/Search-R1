"""Isolated short normal1GPU golden deployment; code chunks <=4KiB.

No training source, prior capture bank, model view, or existing job is changed.
"""
import argparse,base64,hashlib,importlib.util,inspect,json,shlex,subprocess
from pathlib import Path
from .deploy_confirmation_materialization import make_bundle
B='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019'
BASE=B+'/algorithm-improvement-20261021/replay-batched-golden-v2'
P='reproduction/operations/qwen7b_warmup/'
MANIFEST='_batched_golden_manifest.json'
SEEDS=tuple(P+n for n in ('replay_batched_golden.py','replay_batched_context.py','test_replay_batched_context.py','runtime.sh'))

def script(base,code,config_sha):
 return '''#!/usr/bin/env bash
#SBATCH --job-name=q7b-batched-golden
#SBATCH --account=mou176
#SBATCH --partition=acd_u
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=BASE/logs/golden-%j.log
set -euo pipefail
export QWEN7B_CODE_ROOT=@@GOLDEN_CODE@@
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
export FLASH_ATTENTION_DETERMINISTIC=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
exec 9>BASEQ/golden-job.lock
flock -n 9
python3 - @@GOLDEN_CODE@@ <<'VERIFY'
import pathlib,json,hashlib,sys
root=pathlib.Path(sys.argv[1]);v=json.loads((root.parent/'source-publication.json').read_text());assert str(root)==v['code_root']
for name,item in v['files'].items():
 p=root/name;assert p.stat().st_size==item['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256'],name
VERIFY
srun --kill-on-bad-exit=1 bash "$QWEN7B_CODE_ROOT/reproduction/operations/qwen7b_warmup/runtime.sh" -u -m reproduction.operations.qwen7b_warmup.replay_batched_golden --config BASEQ/config.json --config-sha256 SHA --output BASEQ/output
'''.replace('BASEQ',shlex.quote(base)).replace('BASE',base).replace('@@GOLDEN_CODE@@',shlex.quote(code)).replace('SHA',config_sha)

COMMON=r'''
import pathlib,json,hashlib,os,time,fcntl,subprocess,base64,io,tarfile,gzip
b=pathlib.Path(inputs['base']);B=pathlib.Path(inputs['parent']);root=b/('code-'+inputs['commit'][:12]);u=b/('upload-'+inputs['bundle_sha256'][:12])
def read(p):return json.loads(pathlib.Path(p).read_text())
def fp(p):
 p=pathlib.Path(p);h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(8*1024**2),b''):h.update(x)
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())
def checked(v):
 assert fp(v['path'])==v,v['path'];return pathlib.Path(v['path'])
def atomic(p,v):
 p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(v,sort_keys=True)+'\n');q.replace(p)
def cmd(args,timeout=25):
 r=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=timeout);assert r.returncode==0,(r.stdout[-1000:],r.stderr[-1000:]);return r.stdout.strip()
def lock():
 b.mkdir(exist_ok=True,parents=True);x=(b/'operator.lock').open('a');fcntl.flock(x,fcntl.LOCK_EX|fcntl.LOCK_NB);return x
def pub():
 v=read(b/'source-publication.json');r=pathlib.Path(v['code_root'])
 for name,item in v['files'].items():assert fp(r/name)==dict(item,path=str(r/name))
 return v,r
'''
INIT=r'''
g=lock();u.mkdir(exist_ok=True);(b/'logs').mkdir(exist_ok=True)
p=b/'stage-intent.json';v={k:inputs[k] for k in ('commit','bundle_sha256','manifest_sha256','chunks')}
if p.exists():assert read(p)==v,'Immutable golden stage already belongs to another source'
else:atomic(p,v)
print(json.dumps(dict(status='GOLDEN_STAGE_INTENT_BOUND')))
'''
CHUNK=r'''
g=lock();x=base64.b64decode(inputs['chunk_data']);assert len(x)<=4096;p=u/('%05d.part'%inputs['index'])
if p.exists():assert p.read_bytes()==x
else:p.write_bytes(x)
print(json.dumps(dict(status='GOLDEN_CHUNK_ACCEPTED',index=inputs['index'])))
'''
EXTRACT=r'''
g=lock();data=b''.join(p.read_bytes() for p in sorted(u.glob('*.part')));assert hashlib.sha256(data).hexdigest()==inputs['bundle_sha256']
with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as tar:
 manifest=tar.extractfile(inputs['manifest']).read();assert hashlib.sha256(manifest).hexdigest()==inputs['manifest_sha256'];v=json.loads(manifest);files=v['files'];assert {m.name for m in tar.getmembers()}==set(files)|{inputs['manifest']}
 for name in set(files)|{inputs['manifest']}:
  p=pathlib.Path(name);assert not p.is_absolute() and '..' not in p.parts;m=tar.getmember(name);assert m.isfile();x=tar.extractfile(m).read()
  if name in files:assert len(x)==files[name]['bytes'] and hashlib.sha256(x).hexdigest()==files[name]['sha256']
  target=root/p;target.parent.mkdir(parents=True,exist_ok=True)
  if target.exists():assert target.read_bytes()==x
  else:target.write_bytes(x);target.chmod(0o444)
atomic(b/'source-publication.json',dict(code_root=str(root),commit=inputs['commit'],files=files))
env=dict(os.environ,QWEN7B_CODE_ROOT=str(root),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
r=subprocess.run(['bash',str(root/'reproduction/operations/qwen7b_warmup/runtime.sh'),'-m','unittest','reproduction.operations.qwen7b_warmup.test_replay_batched_context'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True,env=env,timeout=60);(b/'cpu-tests.log').write_text(r.stdout);assert r.returncode==0,r.stdout[-2000:]
print(json.dumps(dict(status='GOLDEN_CLOSURE_STAGED_CPU_TESTS_PASSED',files=len(files),log=fp(b/'cpu-tests.log'))))
'''
PREPARE=r'''
g=lock();publication,root=pub();r2=B/'replay-matched9088-bank-v2-20261021/prepared';bank=read(r2/'bank-complete.json');prep=read(checked(bank['source_preparation']));ids=read(checked(bank['group_identities']));assert bank['roles']['fit']==6528 and ids['labels_included'] is False
fit={str(row['qid']) for row in ids['groups'] if row['role']=='fit'};selection=read(checked(prep['selections'][0]));groups={}
for row in selection['trajectories']:
 if str(row['qid']) in fit:groups.setdefault(str(row['qid']),[]).append(row)
qids=sorted(groups,key=lambda q:hashlib.sha256(('batched-golden1729:'+q).encode()).hexdigest())[:8];assert len(qids)==8 and all(len(groups[q])==5 for q in qids)
ordered=[row for q in qids for row in groups[q]];wanted={row['trajectory_id'] for row in ordered};found={};source_refs=[]
for ref in selection['prefix_files']:
 # R2 bank's all-source verification is inherited. Read fresh only files that
 # contain selected training identities, then bind their exact content digest.
 with gzip.open(ref['path'],'rt') as f:
  own=[json.loads(line) for line in f]
 chosen=[row for row in own if row['trajectory_id'] in wanted]
 if chosen:
  checked(ref);source_refs.append(ref)
  for row in chosen:
   assert row['trajectory_id'] not in found;found[row['trajectory_id']]=row
 if set(found)==wanted:break
assert set(found)==wanted
p=b/'inputs/prefix.jsonl.gz';p.parent.mkdir(exist_ok=True)
data=b''.join((json.dumps(found[row['trajectory_id']],sort_keys=True,separators=(',',':'))+'\n').encode() for row in ordered);z=io.BytesIO()
with gzip.GzipFile(fileobj=z,mode='wb',mtime=0) as f:f.write(data)
compressed=z.getvalue()
if p.exists():assert p.read_bytes()==compressed
else:p.write_bytes(compressed)
fixture=dict(questions=8,trajectories=ordered,labels_read_for_selection=False,selection_rule='hash1729 among R2 shard0 fit qids; original complete G5 unchanged',prefix_files=[fp(p)],source_prefix_files=source_refs,source_R2_complete=fp(r2/'bank-complete.json'),source_R2_selection=prep['selections'][0],source_model_binding=selection['source_model_binding'],model_files=selection['model_files'],model_revision=selection['model_revision'])
atomic(b/'inputs/fixture.json',fixture);e4=B/'balanced-rank1-from-epoch1-prefix20-20261020/probe/epoch-0004.json';receipt=read(e4);assert receipt['epoch']==4 and receipt['completed_epoch'] and receipt['steps']==14418
config=dict(fixture=fp(b/'inputs/fixture.json'),checkpoint={k:receipt[k] for k in ('path','bytes','sha256')},checkpoint_receipt=fp(e4),source_publication=fp(b/'source-publication.json'),actor_updates=0,probe_updates=0,labels_read=False)
atomic(b/'config.json',config);text=script(str(b),str(root),fp(b/'config.json')['sha256']);(b/'golden.sbatch').write_text(text);cmd(['bash','-n',str(b/'golden.sbatch')]);atomic(b/'preparation.json',dict(status='GOLDEN_REAL40_TRAIN_PREFIXES_CPU_PREPARED',config=fp(b/'config.json'),script=fp(b/'golden.sbatch'),questions=8,trajectories=40,model_loaded=False,source_checkpoint_SHA_inherited_from_receipt_GPU_will_fresh_hash=True))
print(json.dumps(read(b/'preparation.json')))
'''
SUBMIT=r'''
g=lock();publication,root=pub();pre=read(b/'preparation.json');checked(pre['config']);checked(pre['script']);name='q7b-batched-golden'
if (b/'job.id').exists():print(json.dumps(dict(status='EXISTING_GOLDEN_JOB_PRESERVED',job_id=(b/'job.id').read_text().strip())))
else:
 assert not (b/'submission-intent.json').exists(),'Unknown submit result; inspect before repeating';existing=cmd(['squeue','-h','-u','mou176','-n',name,'-o','%A']);assert not existing,'Same named owned job already exists'
 atomic(b/'submission-intent.json',dict(job_name=name,preparation=fp(b/'preparation.json'),partition='acd_u',qos='normal',gpus=1,dependency=None,time=time.time()));job=cmd(['sbatch','--hold','--parsable',str(b/'golden.sbatch')]).split(';')[0];assert job.isdigit();(b/'job.id').write_text(job+'\n');cmd(['scontrol','release',job]);print(json.dumps(dict(status='GOLDEN_SUBMITTED_NOT_PROGRESS_VERIFIED',job_id=job)))
'''
STATUS=r'''
out=dict(time=time.time())
for name in ('preparation.json','output/progress.json','output/failure.json','output/complete.json'):
 p=b/name
 if p.exists():out[name]=read(p)
if (b/'job.id').exists():
 job=(b/'job.id').read_text().strip();out['job_id']=job;out['squeue']=cmd(['squeue','-h','-j',job,'-o','%i|%T|%M|%R']);out['sacct']=cmd(['sacct','-n','-P','-j',job,'--format=JobID,State,ExitCode,Elapsed,MaxRSS']);logs=sorted((b/'logs').glob('*'));out['log_tail']=logs[-1].read_text(errors='replace')[-1800:] if logs else ''
print(json.dumps(out))
'''

def main():
 p=argparse.ArgumentParser();p.add_argument('--control-root',type=Path);p.add_argument('--action',choices=('prepare','submit','status'),default='status');p.add_argument('--submit-authorized',action='store_true');p.add_argument('--self-test',action='store_true');a=p.parse_args();repo=Path(__file__).resolve().parents[3]
 values=dict(base=BASE,parent=B,manifest=MANIFEST,commit='a'*40,bundle_sha256='b'*64,manifest_sha256='c'*64,chunks=2)
 if a.self_test:
  lengths={}
  for name,block in [('init',INIT),('chunk',CHUNK),('extract',EXTRACT),('prepare',PREPARE),('submit',SUBMIT),('status',STATUS)]:
   own=dict(values,index=0,chunk_data=base64.b64encode(b'x'*4096).decode());code='inputs='+repr(own)+'\n'+COMMON+'\n'+(inspect.getsource(script) if name=='prepare' else '')+'\n'+block;compile(code,name,'exec');lengths[name]=len(('python3 -c '+shlex.quote(code)).encode());assert lengths[name]<16000
  rendered=script(BASE,BASE+'/code-a','a'*64);subprocess.run(['bash','-n'],input=rendered,text=True,check=True);exports='\n'.join(line for line in rendered.splitlines() if line.startswith('export '));subprocess.run(['bash','-euc',exports+'\ntest \"$QWEN7B_CODE_ROOT\" = '+shlex.quote(BASE+'/code-a')],check=True);print(json.dumps(dict(status='GOLDEN_RENDER_BOUND_CHECKS_PASS',lengths=lengths)));return
 if a.control_root is None:p.error('--control-root required')
 if a.action=='submit' and not a.submit_authorized:p.error('--submit-authorized required')
 data=b''
 if a.action=='prepare':
  head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip();remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/codex/qwen7b-probe-warmup-20261019'],cwd=repo,text=True).split()[0];assert head==remote,'Publish source first';paths=subprocess.check_output(['git','ls-tree','-r','--name-only',head],cwd=repo,text=True).splitlines();load=lambda name:subprocess.check_output(['git','show',head+':'+name],cwd=repo);assert Path(__file__).read_bytes()==load(str(Path(__file__).relative_to(repo)));data,report=make_bundle(paths,load,head,seeds=SEEDS,resource_files=('verl/version/version',),manifest_name=MANIFEST);values.update(commit=head,bundle_sha256=hashlib.sha256(data).hexdigest(),manifest_sha256=report['source_manifest_sha256'],chunks=(len(data)+4095)//4096)
 s=importlib.util.spec_from_file_location('golden_control',a.control_root/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');control=importlib.util.module_from_spec(s);s.loader.exec_module(control);address=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0'],text=True).strip()
 def call(block,extra=None):
  own=dict(values);own.update(extra or {});code='inputs='+repr(own)+'\n'+COMMON+'\n'+(inspect.getsource(script) if block==PREPARE else '')+'\n'+block;command='python3 -c '+shlex.quote(code);assert len(command.encode())<16000;v=control.run(command,account='mou176',source_address=address,timeout=100 if block==EXTRACT else 55);rows=[json.loads(x) for x in v.stdout.decode(errors='replace').splitlines() if x.startswith('{')];assert v.returncode==0 and rows, v.stderr.decode(errors='replace')[-500:]+' '+v.stdout.decode(errors='replace')[-1500:];return rows[-1]
 if a.action=='prepare':
  call(INIT)
  for i,start in enumerate(range(0,len(data),4096)):call(CHUNK,dict(index=i,chunk_data=base64.b64encode(data[start:start+4096]).decode()))
  call(EXTRACT);result=call(PREPARE)
 else:result=call(SUBMIT if a.action=='submit' else STATUS)
 out=a.control_root/'outputs/qwen7b-warmup-20261019'/('batched-golden-'+a.action+'.json');out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,sort_keys=True))
if __name__=='__main__':main()
