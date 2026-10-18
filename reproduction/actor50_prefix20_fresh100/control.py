"""New immutable full-state continuation namespace; bounded scalar observation."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import time
import uuid

from reproduction.actor50_probe100.control import archive_source,route,ROOT


REMOTE=r'''
import base64,hashlib,io,json,os,pathlib,re,shutil,socket,subprocess,sys,tarfile,time
p=json.load(sys.stdin);h=p['handle'];b=pathlib.Path(h['base']);repo=b/'repo'
assert re.fullmatch('[0-9a-f]{32}',h['run_id']) and re.fullmatch('[0-9a-f]{32}',h['execution_id'])
assert str(b)=='/data/shaoshitong/search-g1-prefix-selection200-a800-prefix5-'+h['run_id']+'-prefix20-fresh-resume-'+h['execution_id']
def read(path):
 path=pathlib.Path(path);assert path.resolve()==path and not path.is_symlink() and path.stat().st_size<64*1024**2
 return json.loads(path.read_text())
def sha(path):
 value=hashlib.sha256()
 with pathlib.Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8*1024**2),b''):value.update(block)
 return value.hexdigest()
def once(path,value):
 with pathlib.Path(path).open('x') as f:json.dump(value,f,sort_keys=True,allow_nan=False)
def who(pid):
 try:
  path=pathlib.Path('/proc')/str(pid);raw=(path/'stat').read_text();v=raw[raw.rfind(')')+2:].split()
  return dict(pid=pid,sid=int(v[3]),uid=path.stat().st_uid,start_ticks=int(v[19]),state=v[0])
 except (FileNotFoundError,ProcessLookupError):return None
def same(identity):
 actual=who(identity['pid'])
 return actual if actual and all(actual[k]==identity[k] for k in ('pid','sid','uid','start_ticks')) else None
if p['action']=='stage':
 assert not b.exists();parent=pathlib.Path(h['parent']['base'])
 assert read(parent/'preparation/deployment.json')==h['parent']
 clean=read(parent/'logs/cleanup-status.json');assert clean['owned_births_all_gone'] and clean['cleaned']
 assert same(read(parent/'preparation/submit-receipt.json')['leader']) is None
 scope=p['human']['scope'];assert scope['resume_existing_prefix20_epoch20_full_state'] and scope['fresh512_G5_training_per_new_cycle'] and scope['never_reuse_first20_epoch_train_questions']
 latest=read(parent/'probe/latest.json');n=latest['cycle'];clock=latest['ema_updates']
 assert 10<=n<50 and latest['completed_epochs']==2*n and latest['actual_adam_steps_per_rank']==[clock]*4
 cp=pathlib.Path(latest['checkpoint']);assert cp.stat().st_size==latest['bytes'] and sha(cp)==latest['sha256']
 b.mkdir();(b/'preparation').mkdir();(b/'logs').mkdir();repo.mkdir()
 raw=base64.b64decode(p['archive_b64']);assert hashlib.sha256(raw).hexdigest()==h['archive_sha256']
 with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as archive:
  members=archive.getmembers();assert all(not pathlib.PurePosixPath(x.name).is_absolute() and '..' not in pathlib.PurePosixPath(x.name).parts and (x.isfile() or x.isdir()) for x in members)
  archive.extractall(repo,members=members)
 for name,digest in h['source_files'].items():assert sha(repo/name)==digest
 assert shutil.disk_usage(b).free>280*1024**3
 import torch
 sys.path.insert(0,str(repo))
 from reproduction.actor50_prefix20_reuse20.model import validate_model_state
 from reproduction.prefix_selection200.prefix5_model import validate_native_buffers
 state=torch.load(cp,map_location='cpu',weights_only=False)
 assert state['run_id']==h['run_id'] and state['completed_epochs']==2*n and state['cycle']==n
 assert state['new_adam_updates']==state['ema_updates']==clock and state['prefix_policy_tokens']==20 and state['actor_updates']==0
 validate_model_state(state['raw_model']);validate_model_state(state['ema_model'])
 validate_native_buffers(state['raw_buffers']);validate_native_buffers(state['ema_buffers'])
 assert len(state['optimizer']['state'])==14 and len(state['rank_rng'])==4 and [r['rank'] for r in state['rank_rng']]==list(range(4))
 for item in state['optimizer']['state'].values():
  assert int(item['step'])==clock
  for value in item.values():
   if isinstance(value,torch.Tensor):assert torch.isfinite(value).all()
 assert all(x['lr']==5e-6 and x['weight_decay']==0 for x in state['optimizer']['param_groups'])
 def link(source,target):
  source=pathlib.Path(source);target=pathlib.Path(target);target.parent.mkdir(parents=True,exist_ok=True)
  assert source.resolve()==source and source.is_file() and not source.is_symlink();os.link(source,target)
 link(cp,b/'probe'/cp.name);link(cp,b/'probe/latest.pt');once(b/'probe/latest.json',latest)
 for cycle in range(1,n+1):link(parent/'training'/('cycle-%04d'%cycle)/'complete.json',b/'training'/('cycle-%04d'%cycle)/'complete.json')
 (b/'metrics.jsonl').write_bytes((parent/'metrics.jsonl').read_bytes())
 for name in ('prepared.json','index.jsonl','features-complete.json'):link(parent/'data/test'/name,b/'data/test'/name)
 c=read(parent/'preparation/config.json');original=c.get('original_prefix20_base',str(parent))
 aux=read(pathlib.Path(c['source_prefix5_base'])/'preparation/launch-config.json')
 port=None
 for candidate in range(19000,20000):
  try:
   with socket.socket() as s:s.bind(('127.0.0.1',candidate))
   port=candidate;break
  except OSError:pass
 assert port is not None
 c.update(base=str(b),run_id=h['run_id'],execution_id=h['execution_id'],source_commit=h['source_commit'],
  role='frozen_actor50_prefix20_fresh_G5_continue100',original_prefix20_base=original,
  resume_completed_cycles=n,resume_completed_epochs=2*n,resume_checkpoint_sha256=latest['sha256'],
  resume_actual_Adam_EMA=clock,target_probe_epochs=100,cycles=50,prefix_policy_tokens=20,
  source_model_kind='full_prefix20_raw_EMA_Adam_4rankRNG_resume',source_optimizer_imported=True,new_optimizer=False,probe_new_optimizer=False,
  actor_training_enabled=False,actor_updates=0,actor_hidden_native=True,candidate_schedule=None,
  rollout_candidate_questions=512,rollout_continuation_questions=512,selected_questions=512,
  diagnostic_only=False,sampling_attempt_id=h['execution_id'],retrieval_port=port,search_url='http://127.0.0.1:%d/retrieve'%port,
  auxiliary_gpu_indices=aux['auxiliary_gpu_indices'],auxiliary_gpu_uuids=aux['auxiliary_gpu_uuids'],
  rollout_memory_utilization='auto',rollout_utilization_cap=.85,rollout_free_reserve_gib=4,
  rollout_max_num_seqs=128,rollout_max_batched_tokens=4096,rollout_enforce_eager=True,cuda_launch_blocking=True,
  fresh_training_questions_after20=20480,previous20_epoch_training_data_reused=False)
 assert c['actor_version']=='54ac33dcebe64e62a11fe4177f41f116-actor0050' and state['source_probe50_sha256']==c['probe50_sha256']
 once(b/'preparation/config.json',c);once(b/'preparation/deployment.json',h);once(b/'preparation/human-intent.json',p['human'])
 proof=dict(source_checkpoint=str(cp),source_checkpoint_sha256=latest['sha256'],source_latest=latest,
  raw_EMA_Adam_native_buffers_CPU_finite=True,all_four_rank_RNG_preserved=True,
  actual_Adam_EMA=clock,complete_epochs=2*n,fixed512_prefix20_cache_reused=True,parent_owned_writers_gone=True,
  new_training_will_be_fresh_G5=True,actual_CUDA_continuation='PENDING_REAL_NEXT_WRITES',checked_unix=time.time())
 once(b/'preparation/resume-qualified.json',proof)
 value=dict(status='FULLSTATE_PREFIX20_FRESH_SOURCE_STAGED_NOT_LAUNCHED',source_commit=h['source_commit'],
  source_files_verified=len(h['source_files']),config_sha256=sha(b/'preparation/config.json'),
  resumed_epochs=2*n,Adam_EMA=clock,checkpoint_sha256=latest['sha256'],free_remote_bytes=shutil.disk_usage(b).free,checked_unix=time.time())
 once(b/'preparation/source-receipt.json',value)
elif p['action']=='submit':
 assert read(b/'preparation/deployment.json')==h
 for name,digest in h['source_files'].items():assert sha(repo/name)==digest
 assert not (b/'preparation/submit-intent.json').exists(),'Onceclaim exists: inspect, do not replay'
 once(b/'preparation/submit-intent.json',dict(run_id=h['run_id'],checked_unix=time.time()))
 c=read(b/'preparation/config.json');env=dict(os.environ,PYTHONPATH=str(repo),PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='')
 command=[str(pathlib.Path(c['environment_root'])/'bin/python'),'-m','reproduction.actor50_prefix20_fresh100.pipeline','--base',str(b)]
 with (b/'logs/pipeline.log').open('xb') as f:child=subprocess.Popen(command,cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
 identity=who(child.pid);assert identity and identity['sid']==child.pid
 value=dict(status='PREFIX20_FRESH_FULLSTATE_ONCE_SUBMITTED_NOT_ACCEPTED',leader=identity,checked_unix=time.time())
 once(b/'preparation/submit-receipt.json',value)
else:
 assert read(b/'preparation/deployment.json')==h
 small=lambda path:read(path) if pathlib.Path(path).is_file() else None
 value=dict(run_id=h['run_id'],execution_id=h['execution_id'],source_commit=h['source_commit'],checked_unix=time.time(),
  leader=same(read(b/'preparation/submit-receipt.json')['leader']),status=small(b/'logs/status.json'),latest=small(b/'probe/latest.json'),
  failure=small(b/'logs/failure.json'),exit=small(b/'logs/exit.json'),cleanup=small(b/'logs/cleanup-status.json'),
  complete=small(b/'complete.json'),allocation=small(b/'logs/native-allocation.json'),fresh_plan=small(b/'preparation/fresh-plan-proof.json'),
  pool_cursor=small(b/'preparation/pool-cursor.json'),recovery=small(b/'preparation/resume-qualified.json'))
 cycle=(value['status'] or {}).get('cycle',0)
 if cycle:
  root=b/'outers'/('outer-%06d'%cycle)
  value['rollout_progress']={phase:[small(root/phase/('rank-%d-progress.json'%rank)) for rank in range(4)] for phase in ('prefix','full')}
  value['rank_progress']=[small(b/'training'/('cycle-%04d'%cycle)/('rank-%d-progress.json'%rank)) for rank in range(4)]
  value['fresh_complete']=small(b/'cycles'/('cycle-%04d.json'%cycle))
 if (b/'logs/pipeline.log').exists():
  path=b/'logs/pipeline.log'
  with path.open('rb') as f:f.seek(max(0,path.stat().st_size-4096));value['log_tail']=f.read().decode(errors='replace')
raw=json.dumps(value,allow_nan=False);assert len(raw.encode())<128*1024;print(raw)
'''


def main():
    p=argparse.ArgumentParser();p.add_argument('--action',choices=('stage','submit','observe'),required=True)
    p.add_argument('--handle',type=Path,required=True);p.add_argument('--parent-handle',type=Path)
    a=p.parse_args();workdir=Path(__file__).resolve().parents[2];payload=dict(action=a.action)
    if a.action=='stage':
        assert not a.handle.exists() and not subprocess.check_output(['git','status','--porcelain'],cwd=workdir).strip()
        parent=json.loads((a.parent_handle or ROOT/'preparation/actor50-prefix20-reuse20-deployment-20261018.json').read_text())
        archive,files=archive_source(workdir);execution=uuid.uuid4().hex
        h=dict(run_id=parent['run_id'],execution_id=execution,parent=parent,
            source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=workdir,text=True).strip(),
            base='/data/shaoshitong/search-g1-prefix-selection200-a800-prefix5-'+parent['run_id']+'-prefix20-fresh-resume-'+execution,
            source_files=files,archive_sha256=hashlib.sha256(archive).hexdigest(),created_unix=time.time())
        a.handle.write_text(json.dumps(h,indent=2)+'\n')
        payload.update(archive_b64=base64.b64encode(archive).decode(),human=json.loads((ROOT/'preparation/actor50-prefix20-fresh100-human-intent-20261018.json').read_text()))
    else:h=json.loads(a.handle.read_text())
    payload['handle']=h;r=subprocess.run(route(REMOTE),input=json.dumps(payload).encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=65)
    out=ROOT/'preparation'/('actor50-prefix20-fresh-'+a.action+'-'+str(time.time_ns())+'.json')
    out.write_text(json.dumps(dict(exit_code=r.returncode,stdout=r.stdout.decode(),stderr=r.stderr.decode(),checked_unix=time.time()),indent=2)+'\n')
    if r.returncode:raise RuntimeError('Inspect exact once-operation receipt: '+str(out))
    print(json.dumps(dict(receipt=str(out),result=json.loads(r.stdout))))


if __name__=='__main__':main()
