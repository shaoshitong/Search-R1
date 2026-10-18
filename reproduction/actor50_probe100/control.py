"""Published code-only deployment and bounded monitoring through the existing SSH route."""
import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
import time
import uuid

from reproduction.operations.a800_prefix5_variable_pool_control import route,ROOT


REMOTE = r'''
import base64,hashlib,io,json,os,pathlib,re,shutil,socket,subprocess,sys,tarfile,time
sys.dont_write_bytecode=True
p=json.load(sys.stdin);h=p['handle'];b=pathlib.Path(h['base']);repo=b/'repo'
assert re.fullmatch('[0-9a-f]{32}',h['run_id']) and re.fullmatch('[0-9a-f]{32}',h['execution_id'])
expected='/data/shaoshitong/search-g1-prefix-selection200-a800-prefix5-'+h['run_id']+'-actor50-probe100'
assert str(b)==expected+('-resume-'+h['execution_id'] if h.get('resume_parent') else '')
assert b.resolve()==b
def read(path):
 path=pathlib.Path(path)
 assert path.resolve()==path and not path.is_symlink() and path.stat().st_size<=64*1024**2
 return json.loads(path.read_text())
def sha(path):
 result=hashlib.sha256()
 with pathlib.Path(path).open('rb') as stream:
  for block in iter(lambda:stream.read(8*1024**2),b''):result.update(block)
 return result.hexdigest()
def once(path,value):
 path=pathlib.Path(path)
 with path.open('x') as stream:json.dump(value,stream,sort_keys=True,allow_nan=False)
def who(pid):
 try:
  path=pathlib.Path('/proc')/str(pid);raw=(path/'stat').read_text();parts=raw[raw.rfind(')')+2:].split()
  return dict(pid=pid,sid=int(parts[3]),uid=path.stat().st_uid,start_ticks=int(parts[19]),state=parts[0])
 except (FileNotFoundError,ProcessLookupError):return None
def same(identity):
 actual=who(identity['pid'])
 return actual if actual and all(actual[k]==identity[k] for k in ('pid','sid','uid','start_ticks')) else None
if p['action'] in ('stage','resume-stage'):
 assert not b.exists(),'New staging namespace already exists; inspect, do not replace'
 old=pathlib.Path(h['resume_parent']['base'] if h.get('resume_parent') else p['formal_handle']['base']);cleanup=read(old/'logs/cleanup-status.json')
 assert cleanup['owned_births_all_gone'] and cleanup['cleaned']
 assert same(read(old/'preparation/submit-receipt.json')['leader']) is None
 assert p['human_intent']['scope']['actor_frozen_at50'] and p['human_intent']['scope']['new_probe_epochs']==100
 if h.get('resume_parent'):assert read(old/'preparation/deployment.json')==h['resume_parent']
 jointpath=pathlib.Path(h['joint50_path']);assert sha(jointpath)==h['joint50_sha256']
 joint=read(jointpath);assert joint['outer']==50 and joint['probe']['completed_outer']==50
 c=read(old/'preparation/launch-config.json')
 b.mkdir();(b/'preparation').mkdir();(b/'logs').mkdir();repo.mkdir()
 raw=base64.b64decode(p['archive_b64']);assert hashlib.sha256(raw).hexdigest()==h['archive_sha256']
 with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as archive:
  members=archive.getmembers()
  assert all(not pathlib.PurePosixPath(x.name).is_absolute() and '..' not in pathlib.PurePosixPath(x.name).parts and (x.isdir() or x.isfile()) for x in members)
  archive.extractall(repo,members=members)
 for name,digest in h['source_files'].items():assert sha(repo/name)==digest,name
 assert shutil.disk_usage(b).free>=240*1024**3,'Insufficient remote space for retained50 training caches/checkpoints'
 for key in ('resume_joint','resume_pending_probe','resume_import','candidate_schedule','continue_reuse','probe_source_epoch'):
  c.pop(key,None)
 port=None
 for candidate in range(19000,20000):
  try:
   with socket.socket() as probe:probe.bind(('127.0.0.1',candidate))
   port=candidate;break
  except OSError:pass
 assert port is not None,'No unused local retrieval port'
 c.update(base=str(b),run_id=h['run_id'],execution_id=h['execution_id'],source_commit=h['source_commit'],
  role='frozen_actor50_balanced_probe100',target_probe_epochs=100,cycles=50,epochs_per_cycle=2,
  actor_training_enabled=False,actor_updates=0,actor_version=joint['actor']['actor_version'],
  actor_model_dir=joint['actor']['hf_model_dir'],joint50_path=str(jointpath),joint50_sha256=h['joint50_sha256'],
  probe50_snapshot=joint['probe_snapshot'],probe50_sha256=joint['probe']['sha256'],
  probe_source_checkpoint=joint['probe_snapshot'],probe_source_sha256=joint['probe']['sha256'],probe_source_model_kind='raw_online50_model_only',
  probe_learning_rate=5e-6,ema_decay=.99,probe_new_optimizer=True,source_optimizer_imported=False,
  retrieval_port=port,search_url='http://127.0.0.1:%d/retrieve'%port,
  rollout_candidate_questions=512,rollout_continuation_questions=512,selected_questions=512,
  seed=1729,source_human_intent_sha256=p['human_intent_sha256'],sampling_attempt_id=c.get('sampling_attempt_id') if h.get('resume_parent') else h['execution_id'])
 if h.get('resume_parent'):
  import torch
  latest=read(old/'probe/latest.json');n=latest['cycle'];steps=latest['ema_updates']
  assert 1<=n<50 and latest['completed_epochs']==2*n and latest['actual_adam_steps_per_rank']==[steps]*4
  source=pathlib.Path(latest['checkpoint']);assert sha(source)==latest['sha256']
  state=torch.load(source,map_location='cpu',weights_only=False)
  assert state['cycle']==n and state['completed_epochs']==2*n and state['new_adam_updates']==state['ema_updates']==steps
  assert state['run_id']==h['run_id'] and state['actor_updates']==0 and len(state['rank_rng'])==4
  assert [row['rank'] for row in state['rank_rng']]==list(range(4))
  assert state['source_probe50_sha256']==c['probe50_sha256'] and state['ema_decay']==.99
  assert all(group['lr']==5e-6 and group['weight_decay']==0 for group in state['optimizer']['param_groups'])
  sys.path.insert(0,str(repo))
  from reproduction.prefix_selection200.prefix5_model import validate_model_state,validate_native_buffers
  validate_model_state(state['raw_model']);validate_model_state(state['ema_model'])
  validate_native_buffers(state['raw_buffers']);validate_native_buffers(state['ema_buffers'])
  assert len(state['optimizer']['state'])==14
  for item in state['optimizer']['state'].values():
   assert int(item['step'])==steps
   for value in item.values():
    if isinstance(value,torch.Tensor):assert torch.isfinite(value).all()
  def link(src,dest):
   src=pathlib.Path(src);dest=pathlib.Path(dest);dest.parent.mkdir(parents=True,exist_ok=True)
   assert src.resolve()==src and src.is_file() and not src.is_symlink()
   os.link(src,dest)
  for rel in ('preparation/initial-probe50-model.pt','preparation/source50-qualified.json','fixed-test/complete.json','metrics.jsonl'):
   link(old/rel,b/rel)
  (b/'probe').mkdir(exist_ok=True)
  target=b/'probe'/source.name;link(source,target);link(target,b/'probe/latest.pt')
  once(b/'probe/latest.json',latest)
  for cycle in range(1,n+1):
   for rel in ('cycles/cycle-%04d.json'%cycle,'training/cycle-%04d/complete.json'%cycle):link(old/rel,b/rel)
  # The scalar curve is appended by this new writer; never append to the old inode.
  curve=b/'metrics.jsonl';raw_curve=curve.read_bytes();curve.unlink();curve.write_bytes(raw_curve)
  c.update(resume_completed_cycles=n,resume_new_adam_updates=steps,resume_parent_base=str(old),
   warmup_optimizer_restored=True,resume_checkpoint_sha256=latest['sha256'],
   rollout_memory_utilization='auto',rollout_utilization_cap=.85,rollout_free_reserve_gib=4,
   rollout_max_num_seqs=128,rollout_max_batched_tokens=4096,cuda_launch_blocking=True)
  parent_failure=read(old/'logs/failure.json')
  once(b/'preparation/resume-qualified.json',dict(parent_base=str(old),parent_source_commit=h['resume_parent']['source_commit'],
   source_checkpoint=str(source),source_checkpoint_sha256=latest['sha256'],completed_cycles=n,completed_epochs=2*n,
   new_adam_updates=steps,ema_updates=steps,raw_EMA_optimizer_full_CPU_finite=True,all_four_rank_RNG_preserved=True,
   fixed_test_events_reused=True,initial_actor50_qualification_reused=True,actor_updates=0,
   parent_failure=parent_failure,parent_prefix_error_log_sha256={path.name:sha(path) for path in sorted((old/'outers'/('outer-%06d'%(n+1))/'prefix').glob('rank-*.log'))},
   parent_owned_writers_gone=True,pending_rollout_recomputed_only_after_complete_checkpoint=True,
   actual_CUDA_restoration='PENDING_not_CPU_acceptance',checked_unix=time.time()))
  del state
 once(b/'preparation/deployment.json',h);once(b/'preparation/launch-config.json',c)
 once(b/'preparation/human-intent.json',p['human_intent'])
 value=dict(status='CODE_STAGED_NOT_LAUNCHED',run_id=h['run_id'],base=str(b),source_commit=h['source_commit'],
  source_files_verified=len(h['source_files']),config_sha256=sha(b/'preparation/launch-config.json'),
  source50_actor_version=c['actor_version'],source50_probe_sha256=c['probe50_sha256'],
  fixed_test_identity_sha256=c['fixed_test_identity_sha256'],free_disk_bytes=shutil.disk_usage(b).free,
  native_train_indices=c['physical_gpu_indices'],auxiliary_indices=c['auxiliary_gpu_indices'],checked_unix=time.time())
 once(b/'preparation/source-receipt.json',value)
elif p['action']=='submit':
 assert read(b/'preparation/deployment.json')==h
 for name,digest in h['source_files'].items():assert sha(repo/name)==digest
 if (b/'preparation/submit-intent.json').exists():
  value=dict(status='EXISTING_ONCE_SUBMIT_INSPECT_ONLY',submit=read(b/'preparation/submit-receipt.json') if (b/'preparation/submit-receipt.json').exists() else None)
 else:
  c=read(b/'preparation/launch-config.json')
  once(b/'preparation/submit-intent.json',dict(run_id=h['run_id'],source_commit=h['source_commit'],checked_unix=time.time()))
  env=dict(os.environ,PYTHONPATH=str(repo),PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1')
  for key in ('PYTHONHOME','RAY_ADDRESS'):env.pop(key,None)
  command=[str(pathlib.Path(c['environment_root'])/'bin/python'),'-m','reproduction.actor50_probe100.pipeline','--base',str(b)]
  with (b/'logs/pipeline.log').open('xb') as log:
   child=subprocess.Popen(command,env=env,cwd=repo,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  identity=who(child.pid);assert identity and identity['sid']==child.pid and identity['uid']==os.getuid()
  value=dict(status='ONCE_SUBMITTED_NOT_TRAINING_ACCEPTANCE',run_id=h['run_id'],source_commit=h['source_commit'],
   leader=identity,command=command,checked_unix=time.time())
  once(b/'preparation/submit-receipt.json',value)
else:
 assert read(b/'preparation/deployment.json')==h
 submit=read(b/'preparation/submit-receipt.json');leader=same(submit['leader'])
 small=lambda path:read(path) if pathlib.Path(path).is_file() else None
 value=dict(run_id=h['run_id'],source_commit=h['source_commit'],checked_unix=time.time(),leader=leader,
  status=small(b/'logs/status.json'),source50_qualification=small(b/'preparation/source50-qualified.json'),
  allocation=small(b/'logs/native-allocation.json'),latest=small(b/'probe/latest.json'),complete=small(b/'complete.json'),
  failure=small(b/'logs/failure.json'),exit=small(b/'logs/exit.json'),cleanup=small(b/'logs/cleanup-status.json'),
  recovery=small(b/'preparation/resume-qualified.json'))
 current=(value['status'] or {}).get('cycle',0)
 if current:
  out=b/'training'/('cycle-%04d'%current)
  value['rank_progress']=[small(out/('rank-%d-progress.json'%rank)) for rank in range(4)]
  root=b/'outers'/('outer-%06d'%current)
  value['rollout_progress']={phase:[small(root/phase/('rank-%d-progress.json'%rank)) for rank in range(4)] for phase in ('prefix','full')}
 for path in (b/'logs/pipeline.log',):
  if path.is_file():
   with path.open('rb') as stream:stream.seek(max(0,path.stat().st_size-4096));value['log_tail']=stream.read().decode(errors='replace')
raw=json.dumps(value,allow_nan=False);assert len(raw.encode())<=128*1024
print(raw)
'''


def archive_source(workdir):
    raw=subprocess.check_output(['git','archive','HEAD'],cwd=workdir)
    files={}
    with tarfile.open(fileobj=io.BytesIO(raw),mode='r:') as archive:
        for member in archive.getmembers():
            if member.isfile():files[member.name]=hashlib.sha256(archive.extractfile(member).read()).hexdigest()
            elif not member.isdir():raise ValueError('Only regular code/config/docs are allowed')
    out=io.BytesIO()
    with tarfile.open(fileobj=out,mode='w:gz') as compressed:
        with tarfile.open(fileobj=io.BytesIO(raw),mode='r:') as original:
            for member in original.getmembers():compressed.addfile(member,original.extractfile(member) if member.isfile() else None)
    return out.getvalue(),files


def main():
    p=argparse.ArgumentParser();p.add_argument('--action',choices=('stage','resume-stage','submit','observe'),required=True)
    p.add_argument('--handle',type=Path,required=True);p.add_argument('--human-intent',type=Path)
    p.add_argument('--parent-handle',type=Path)
    a=p.parse_args();workdir=Path(__file__).resolve().parents[2]
    payload=dict(action=a.action)
    if a.action in ('stage','resume-stage'):
        if a.handle.exists():raise FileExistsError('Never recreate a staged namespace')
        if subprocess.check_output(['git','status','--porcelain'],cwd=workdir).strip():raise ValueError('Publish clean exact source first')
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=workdir,text=True).strip()
        archive,files=archive_source(workdir)
        parent=json.loads(a.parent_handle.read_text()) if a.action=='resume-stage' else None
        run=parent['run_id'] if parent else uuid.uuid4().hex;execution=uuid.uuid4().hex
        h=dict(run_id=run,execution_id=execution,source_commit=commit,
            base='/data/shaoshitong/search-g1-prefix-selection200-a800-prefix5-'+run+'-actor50-probe100'+('-resume-'+execution if parent else ''),
            joint50_path='/data/shaoshitong/search-g1-prefix-selection200-a800-prefix5-54ac33dcebe64e62a11fe4177f41f116-resume-a4a0c2a18cca40bc8c3850d40eed8554/outers/outer-000050/complete.json',
            joint50_sha256='80c5b4cdbafed1c92441352cd725cc7b2819319da37f7672498f58524ec79f2c',
            archive_sha256=hashlib.sha256(archive).hexdigest(),source_files=files,created_unix=time.time())
        if parent:h['resume_parent']=parent
        a.handle.write_text(json.dumps(h,indent=2)+'\n')
        human=a.human_intent.read_bytes()
        payload.update(archive_b64=base64.b64encode(archive).decode(),human_intent=json.loads(human),
            human_intent_sha256=hashlib.sha256(human).hexdigest(),
            formal_handle=json.loads((ROOT/'preparation/a800-prefix5-chunked-prefill-fullstate-deployment-20261018.json').read_text()))
    else:h=json.loads(a.handle.read_text())
    payload['handle']=h;started=time.time()
    result=subprocess.run(route(REMOTE),input=json.dumps(payload).encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=65)
    receipt=dict(action=a.action,checked_unix=time.time(),elapsed_seconds=time.time()-started,exit_code=result.returncode,
        stdout=result.stdout.decode(),stderr=result.stderr.decode(),run_id=h['run_id'],execution_id=h['execution_id'],source_commit=h['source_commit'])
    output=ROOT/'preparation'/('actor50-probe100-'+a.action+'-'+str(time.time_ns())+'.json')
    output.write_text(json.dumps(receipt,indent=2)+'\n')
    if result.returncode or len(result.stdout)>128*1024:raise RuntimeError('Inspect actual receipt, never blindly replay: '+str(output))
    print(json.dumps(dict(receipt=str(output),result=json.loads(result.stdout))))


if __name__=='__main__':main()
