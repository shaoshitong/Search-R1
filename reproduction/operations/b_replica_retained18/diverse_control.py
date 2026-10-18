"""Once-only deployment of the authorized independent CP50 eight-card fork."""
import argparse,base64,gzip,hashlib,io,json,subprocess,tarfile,time
from pathlib import Path
from .control import REPO,OUT,remote
PHYSICAL='/mnt/data/shaoshitong/search-g1-prefix-selection200-reference-replica-b-diverse10to5-cp50-eight-20261018'
CANONICAL='/mnt/data/sihe-share5/workspace/training_data_node_10/shaoshitong/'+PHYSICAL.rsplit('/',1)[1]
PREPARED='/mnt/data/shaoshitong/search-g1-prefix-selection200-reference-replica-eight-prepared-20261018'
SOURCE='/mnt/data/sihe-share5/workspace/training_data_node_10/shaoshitong/search-g1-prefix-selection200-reference-replica-b-20261018-d9e5385ebec2-cache-repair1-kv-profile-repair1-actor-memory-repair1-h10011-r1'
REMOTE=r'''
import pathlib,os,sys,json,subprocess,time,hashlib,uuid
p=pathlib.Path(PHYSICAL);canonical=CANONICAL
def read(f):return json.loads(f.read_text())if f.exists()else None
def sha(f):return hashlib.sha256(f.read_bytes()).hexdigest()
wrapper=p/'repo/reproduction/operations/b_replica_retained18/private_paths.sh'
python='/dataset/shaoshitong/Search-G1-common8/env/bin/python'
def command(args):
 return ['unshare','--mount','--propagation','private','bash',str(wrapper),'/usr/bin/env',
 'SEARCH_G1_PHYSICAL_WORLD_SIZE=8','SEARCH_G1_B_PREFIX_TOKENS=20','PYTHONDONTWRITEBYTECODE=1',
 'PYTHONPATH='+canonical+'/repo','HF_HUB_OFFLINE=1','HF_DATASETS_OFFLINE=1','CUDA_VISIBLE_DEVICES=',
 'OMP_NUM_THREADS=2','MKL_NUM_THREADS=2',python]+args
if ACTION=='stage':
 import io,tarfile,base64,gzip
 assert not p.exists(),'Existing stage must be reconciled, never overwritten'
 p.mkdir();(p/'repo').mkdir();(p/'preparation').mkdir();(p/'logs').mkdir()
 (p/'preparation/stage-intent.json').write_text(json.dumps(dict(commit=COMMIT,created_unix=time.time())))
 with tarfile.open(fileobj=io.BytesIO(gzip.decompress(base64.b64decode(ARCHIVE))))as tf:
  for member in tf.getmembers():assert not member.name.startswith('/')and'..'not in pathlib.Path(member.name).parts and(member.isfile()or member.isdir())
  tf.extractall(p/'repo')
 files=json.loads(gzip.decompress(base64.b64decode(MANIFEST)))
 for row in files:
  target=p/'repo'/row['path']
  if not target.exists():
   original=pathlib.Path(PREPARED)/'repo'/row['path'];data=original.read_bytes();assert hashlib.sha256(data).hexdigest()==row['sha256'],row['path']
   target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data)
  assert sha(target)==row['sha256'],row['path']
 c=read(pathlib.Path(PREPARED)/'preparation/launch-config.json');old=c['base']
 def rebase(x):
  if isinstance(x,dict):return{k:rebase(v)for k,v in x.items()}
  if isinstance(x,list):return[rebase(v)for v in x]
  if isinstance(x,str)and(x==old or x.startswith(old+'/')):return canonical+x[len(old):]
  return x
 c=rebase(c)
 source_physical=pathlib.Path('/mnt/data/shaoshitong')/pathlib.Path(SOURCE).name
 joint=read(source_physical/'outers/outer-000050/complete.json')
 c.update(run_id=joint['run_id'],execution_id=uuid.uuid4().hex,source_commit=COMMIT,actor_version=joint['actor']['actor_version'],
  preparation_only=False,training_authorized=True,experiment_assignment_pending=False,environment_only_8gpu=False,
  initialization_assignment_pending=False,checkpoint_imported=True,full_state_resume=True,
  fresh_actor_optimizer_scheduler_rng_sampler=False,old_actor_state_imported=True,old_online_probe_optimizer_imported=True,
  actor_global_micro_batch_size=8,prefix_policy_tokens=20,hidden_capture_contract='sample21_keep20_postblock27_native_residual_sum_v1',
  b_progressive=dict(mode='b_cp50_prefix20_40_80_two_stage_v1',resume_outer=50),
  prefix_diversity=dict(mode='ten_native_prefixes_farthest_first_five_v1',candidates_per_question=10,retained_per_question=5,discarded_probe_evaluations=0),
  resume_joint=dict(adapter='b_diverse_4to8_cp50_v1',source_base=SOURCE,completed_outer=50,source_joint_sha256=sha(source_physical/'outers/outer-000050/complete.json')),
  retrieval_port=19660,search_url='http://127.0.0.1:19660/retrieve',grounding_probe_base_port=19670,
  reference_B_snapshot_commit='2b50a4512d1a00213894ac2ac8235c9dd466eeef',
  unchanged_generation_budget_and_sampling_parameters=False)
 (p/'preparation/launch-config.json').write_text(json.dumps(c,sort_keys=True))
 (p/'preparation/human-authorization.json').write_text(json.dumps(AUTH,ensure_ascii=False,indent=2))
 (p/'preparation/source-full-sha.json').write_text(json.dumps(dict(commit=COMMIT,files=files)))
 (p/'preparation/stage-complete.json').write_text(json.dumps(dict(commit=COMMIT,files=len(files),completed_unix=time.time())))
 test=subprocess.run(command(['-m','reproduction.operations.b_replica_retained18.diverse_cpu_check','--base',canonical]),text=True,capture_output=True,timeout=120)
 (p/'preparation/cpu-check-terminal.json').write_text(json.dumps(dict(returncode=test.returncode,stdout=test.stdout,stderr=test.stderr)))
 print(json.dumps(dict(config=c,check=read(p/'preparation/cpu-check-terminal.json'))))
elif ACTION=='migrate':
 check=read(p/'preparation/cpu-check-terminal.json');assert check['returncode']==0,check
 assert not(p/'preparation/import-worker-launch.json').exists(),'CPU migration already dispatched; inspect instead'
 intent=p/'preparation/import-worker-launch.json'
 with intent.open('x')as f:json.dump(dict(created_unix=time.time(),action='CP50_4to8_CPU_import_once'),f);f.flush();os.fsync(f.fileno())
 with (p/'logs/cp50-import.log').open('ab',buffering=0)as log:
  child=subprocess.Popen(command(['-u','-m','reproduction.prefix_selection200.diverse_resume','--base',canonical]),stdout=log,stderr=log,start_new_session=True,close_fds=True)
 st=pathlib.Path('/proc')/str(child.pid)/'stat';fields=st.read_text().rsplit(')',1)[1].split()
 launched=dict(pid=child.pid,start_ticks=int(fields[19]),sid=int(fields[3]),created_unix=time.time())
 (p/'preparation/import-worker-receipt.json').write_text(json.dumps(launched));print(json.dumps(launched))
elif ACTION=='submit':
 assert read(p/'preparation/import-worker-terminal.json')['status']=='COMPLETE'
 code='import json;from reproduction.prefix_selection200.launch import submit;print(json.dumps(submit('+repr(canonical)+')))'
 value=subprocess.run(command(['-c',code]),capture_output=True,text=True,timeout=180)
 print(json.dumps(dict(returncode=value.returncode,stdout=value.stdout,stderr=value.stderr)))
else:
 def tail(f,n=2500):
  with f.open('rb')as stream:stream.seek(max(0,f.stat().st_size-n));return stream.read().decode(errors='replace')
 j=read(p/'latest-joint.json');stage=read(p/'logs/pipeline-status.json');outer=max(51,(stage or{}).get('outer',51));root=p/'outers'/('outer-%06d'%outer)
 progress={str(f.relative_to(p)):read(f)for f in root.rglob('*progress*.json')if f.stat().st_size<15000}
 ranks=[dict(path=str(f),bytes=f.stat().st_size,mtime=f.stat().st_mtime)for f in(p/'actor/imported-cp50-eight').glob('rank-*.pt*')]
 logs=[dict(path=str(f.relative_to(p)),bytes=f.stat().st_size,mtime=f.stat().st_mtime,tail=tail(f))for f in sorted((p/'logs').glob('*.log'),key=lambda f:f.stat().st_mtime,reverse=True)[:4]]
 if root.exists():logs +=[dict(path=str(f.relative_to(p)),bytes=f.stat().st_size,mtime=f.stat().st_mtime,tail=tail(f,1200))for f in root.rglob('rank-*.log')if f.stat().st_size][:8]
 def birth(name):
  r=read(p/'preparation'/name)
  if not r:return None
  who=r.get('leader',r);q=pathlib.Path('/proc')/str(who.get('pid'));actual=None
  if q.exists():actual=dict(stat=(q/'stat').read_text(),command=(q/'cmdline').read_bytes().replace(b'\0',b' ').decode())
  return dict(receipt=r,actual=actual)
 facts=[]
 for n in range(51,201):
  v=read(p/'outers'/('outer-%06d'%n)/'complete.json')
  if v:facts.append({k:v.get(k)for k in ('outer','source_commit','selection_metrics','iteration_wall','phase_timings')}|dict(actor=v['actor']['actual_adam_steps_per_rank'],probe=v['probe']['actual_adam_steps_per_rank']))
 print(json.dumps(dict(observed_unix=time.time(),source_commit=read(p/'preparation/launch-config.json')['source_commit'],stage=stage,
  joint_outer=(j or{}).get('outer'),facts=facts,progress=progress,logs=logs,import_files=ranks,
  importer=birth('import-worker-receipt.json'),import_terminal=read(p/'preparation/import-worker-terminal.json'),
  training=birth('submit-receipt.json'),cleanup=read(p/'logs/cleanup-status.json'),complete=read(p/'complete.json'))))
'''

def main():
 parser=argparse.ArgumentParser();parser.add_argument('action',choices=['stage','migrate','submit','observe']);args=parser.parse_args()
 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
 values=dict(ACTION=args.action,PHYSICAL=PHYSICAL,CANONICAL=CANONICAL,PREPARED=PREPARED,SOURCE=SOURCE,COMMIT=commit)
 if args.action=='stage':
  assert not subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip(),'Publish clean source first'
  published=subprocess.check_output(['git','ls-remote','origin','refs/heads/codex/b-diverse10to5-cp50-eight-20261018'],cwd=REPO,text=True).split()[0];assert published==commit
  tracked=subprocess.check_output(['git','ls-files','-z'],cwd=REPO).decode().rstrip('\0').split('\0')
  files=[dict(path=n,sha256=hashlib.sha256((REPO/n).read_bytes()).hexdigest())for n in tracked]
  changed=set(subprocess.check_output(['git','diff','--name-only','3d05b867a3db7823f92c68da9aa3e704b142a7a4',commit],cwd=REPO,text=True).splitlines())
  buffer=io.BytesIO()
  with tarfile.open(fileobj=buffer,mode='w')as tf:
   for n in tracked:
    if n in changed:tf.add(REPO/n,arcname=n,recursive=False)
  values.update(ARCHIVE=base64.b64encode(gzip.compress(buffer.getvalue())).decode(),MANIFEST=base64.b64encode(gzip.compress(json.dumps(files).encode())).decode(),AUTH=json.loads((OUT/'DIVERSE_HUMAN_AUTHORIZATION.json').read_text()))
 script='\n'.join(k+'='+repr(v)for k,v in values.items())+'\n'+REMOTE
 compile(script,'diverse-'+args.action,'exec')
 evidence=OUT/('diverse-'+args.action+'-'+str(time.time_ns())+'-transport.json')
 value=json.loads(remote(script,evidence,target='h100-10',established_controller=True))
 actual=evidence.with_name(evidence.name.replace('-transport','-actual'));actual.write_text(json.dumps(value,indent=2))
 (OUT/'DIVERSE_LATEST.json').write_text(json.dumps(dict(action=args.action,evidence=str(actual),actual=value),indent=2))
 if args.action=='stage':
  (OUT/'DIVERSE_CONFIG.json').write_text(json.dumps(value['config'],indent=2));print(json.dumps(dict(evidence=str(actual),check=value['check'])))
 else:print(json.dumps(dict(evidence=str(actual),actual=value)))
if __name__=='__main__':main()
