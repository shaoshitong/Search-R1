"""One immutable actor50 branch: CPU prepare, latest-warmup seed, then launch."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import time
import uuid

from reproduction.actor50_probe100.control import archive_source,route,ROOT
from .bootstrap import SOURCE_BASE,SOURCE_JOINT_SHA,RUN

REMOTE=r'''
import base64,hashlib,io,json,os,pathlib,subprocess,sys,tarfile,time
p=json.load(sys.stdin);h=p['handle'];b=pathlib.Path(h['base']);repo=b/'repo'
assert str(b)=='/data/shaoshitong/search-g1-prefix-selection200-a800-prefix5-'+h['run_id']+'-resume-'+h['execution_id']
def read(f):
 f=pathlib.Path(f);assert f.resolve()==f and not f.is_symlink() and f.stat().st_size<8*1024**2
 return json.loads(f.read_text())
def sha(f):return hashlib.sha256(pathlib.Path(f).read_bytes()).hexdigest()
def once(f,v):
 f=pathlib.Path(f);f.parent.mkdir(parents=True,exist_ok=True)
 with f.open('x') as out:json.dump(v,out,sort_keys=True,allow_nan=False)
def who(pid):
 try:
  f=pathlib.Path('/proc')/str(pid);v=(f/'stat').read_text();v=v[v.rfind(')')+2:].split()
  return dict(pid=pid,start_ticks=int(v[19]),sid=int(v[3]),uid=f.stat().st_uid,state=v[0])
 except (FileNotFoundError,ProcessLookupError):return None
def same(v):
 now=who(v['pid']);return now if now and all(now[k]==v[k] for k in ('pid','start_ticks','sid','uid')) else None
def source():
 assert read(b/'preparation/deployment.json')==h
 for name,d in h['source_files'].items():assert sha(repo/name)==d
def launch(name,command,c):
 env=dict(os.environ,PYTHONPATH=str(repo),PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',
  SEARCH_G1_GRPO_FROM50='1',SEARCH_G1_GRPO_PREFIX='20',CUDA_VISIBLE_DEVICES='')
 env.pop('PYTHONHOME',None)
 with (b/'logs'/(name+'.log')).open('xb') as log:
  child=subprocess.Popen(command,cwd=repo,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 identity=who(child.pid);assert identity['sid']==child.pid
 value=dict(leader=identity,checked_unix=time.time(),source_commit=h['source_commit'])
 once(b/'preparation'/(name+'-launch.json'),value);return value
action=p['action']
if action=='stage':
 assert not b.exists();b.mkdir();repo.mkdir();(b/'preparation').mkdir();(b/'logs').mkdir()
 raw=base64.b64decode(p['archive_b64']);assert hashlib.sha256(raw).hexdigest()==h['archive_sha256']
 with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as tf:
  members=tf.getmembers();assert all(not pathlib.PurePosixPath(x.name).is_absolute() and '..' not in pathlib.PurePosixPath(x.name).parts and (x.isfile() or x.isdir()) for x in members)
  tf.extractall(repo,members=members)
 for name,d in h['source_files'].items():assert sha(repo/name)==d
 sys.path.insert(0,str(repo))
 from reproduction.actor50_grpo.bootstrap import SOURCE_BASE,SOURCE_JOINT_SHA
 from reproduction.actor50_grpo.schedule import BRANCH_SCHEDULE
 old=pathlib.Path(SOURCE_BASE);assert sha(old/'outers/outer-000050/complete.json')==SOURCE_JOINT_SHA
 c=read(old/'preparation/launch-config.json')
 for key in ('resume_joint','resume_pending_probe','resume_import','continue_reuse','source_compatibility','source_compatibility_sha256','authorized_candidate_delta','authorized_infrastructure_delta'):c.pop(key,None)
 c.update(base=str(b),run_id=h['run_id'],execution_id=h['execution_id'],source_commit=h['source_commit'],
  grpo_from_actor50=True,actor50_branch_schedule=BRANCH_SCHEDULE,actor50_pool_consumed_candidates=230400,
  resume_joint=dict(adapter='actor50_prefix2040_v1',source_base=SOURCE_BASE,completed_outer=50),
  prefix_policy_tokens=20,hidden_capture_contract='sample21_keep20_postblock27_native_residual_sum_v1',
  source_joint50_sha256=SOURCE_JOINT_SHA,pool_state_path=str(b/'question-pool/state.json'),
  prefix_probe_receipt=str(b/'probe/latest.json'),actual_em_selection_fallback='probe_score_fill',
  rollout_enforce_eager=True,rollout_max_num_seqs=128,rollout_max_batched_tokens=4096,
  rollout_utilization_cap=.85,rollout_free_reserve_gib=4,cuda_launch_blocking=True,
  probe_source_checkpoint='PENDING_LATEST_COMPLETE_WARMUP_HANDOFF',probe_source_sha256='',probe_source_epoch=0,
  probe_source_model_kind='ema',probe_source_run=p['warmup']['run_id'])
 # Pick unused loopback ports now; owned services will validate their run IDs.
 import socket
 def port(start,width):
  for n in range(start,60000-width):
   sockets=[]
   try:
    for i in range(width):
     s=socket.socket();sockets.append(s);s.bind(('127.0.0.1',n+i))
    return n
   except OSError:pass
   finally:
    for s in sockets:s.close()
  raise RuntimeError('No unused service port range')
 c['retrieval_port']=port(19000,1);c['search_url']='http://127.0.0.1:%d/retrieve'%c['retrieval_port']
 c['grounding_probe_base_port']=port(24000,4)
 once(b/'preparation/launch-config.json',c);once(b/'preparation/deployment.json',h)
 once(b/'preparation/warmup-parent-handle.json',p['warmup']);once(b/'preparation/human-intent.json',p['intent'])
 for prefix in ('20','40'):
  env=dict(os.environ,PYTHONPATH=str(repo),PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='',SEARCH_G1_GRPO_FROM50='1',SEARCH_G1_GRPO_PREFIX=prefix)
  env.pop('PYTHONHOME',None)
  checked=subprocess.run([str(pathlib.Path(c['environment_root'])/'bin/python'),'-m','reproduction.actor50_grpo.check_entries'],
   cwd=repo,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=25)
  if checked.returncode:raise RuntimeError('Actual CPU entry failed: '+checked.stderr.decode()[-4000:])
 value=dict(status='CODE_AND_CONFIG_STAGED_NO_GPU',checked_unix=time.time(),source_files=len(h['source_files']),source_commit=h['source_commit'])
 once(b/'preparation/source-receipt.json',value)
elif action in ('prepare','seed'):
 source();c=read(b/'preparation/launch-config.json');phase='actor' if action=='prepare' else 'seed';name='prepare-'+phase
 assert not (b/'preparation'/(name+'-intent.json')).exists(),'Onceclaim exists: inspect, do not replay'
 if action=='seed':
  assert read(b/'preparation/prepare-actor-exit.json')['success']
  wh=read(b/'preparation/warmup-parent-handle.json');clean=read(pathlib.Path(wh['base'])/'logs/cleanup-status.json')
  assert clean['cleaned'] and clean['owned_births_all_gone']
 once(b/'preparation'/(name+'-intent.json'),dict(checked_unix=time.time()))
 value=launch(name,[str(pathlib.Path(c['environment_root'])/'bin/python'),'-m','reproduction.actor50_grpo.bootstrap',phase,'--base',str(b)],c)
elif action=='submit':
 source();c=read(b/'preparation/launch-config.json');sys.path.insert(0,str(repo))
 os.environ['SEARCH_G1_GRPO_FROM50']='1';os.environ['SEARCH_G1_GRPO_PREFIX']='20'
 from reproduction.actor50_grpo.bootstrap import validate_imported_boundary
 assert validate_imported_boundary(b,c)==50
 assert not (b/'preparation/submit-intent.json').exists(),'Onceclaim exists: inspect, do not replay'
 once(b/'preparation/submit-intent.json',dict(config_sha256=sha(b/'preparation/launch-config.json'),checked_unix=time.time()))
 value=launch('pipeline',[str(pathlib.Path(c['environment_root'])/'bin/python'),'-m','reproduction.actor50_grpo.run','--base',str(b)],c)
 once(b/'preparation/submit-receipt.json',value)
else:
 source();small=lambda f:read(f) if pathlib.Path(f).is_file() else None
 value=dict(run_id=h['run_id'],execution_id=h['execution_id'],source_commit=h['source_commit'],checked_unix=time.time(),
  prepare_progress=small(b/'preparation/prepare-progress.json'),prepare_actor_exit=small(b/'preparation/prepare-actor-exit.json'),
  prepare_seed_exit=small(b/'preparation/prepare-seed-exit.json'),stage=small(b/'logs/pipeline-status.json'),
  probe=small(b/'probe/latest.json'),failure=small(b/'logs/failure.json'),exit=small(b/'logs/exit.json'),
  cleanup=small(b/'logs/cleanup-status.json'))
 completed=small(b/'logs/pipeline-complete.json')
 value['complete']=None if completed is None else dict(run_id=completed['run_id'],complete_outers=completed['complete_outers'],
  final_evaluation={k:v for k,v in completed['final_evaluation'].items() if k in ('questions','strict_em','exact_match','em','f1','count','ordered_ids_sha256','question_ids_sha256')})
 value['workers']={}
 for name in ('prepare-actor','prepare-seed','pipeline'):
  item=small(b/'preparation'/(name+'-launch.json'));value['workers'][name]=same(item['leader']) if item else None
 joint=small(b/'latest-joint.json')
 if joint:
  a=joint['actor'];value['joint']=dict(outer=joint['outer'],actor_Adam=a['actual_adam_steps_per_rank'],
   scheduler=a['scheduler_completed_outer'],probe=joint['probe'],rollout_mixed=joint['rollout'].get('actual_selected_mixed_fraction'),
   source_commit=joint['source_commit'],actor50_branch_plan=joint.get('actor50_branch_plan'))
 outer=(value['stage'] or {}).get('outer')
 if outer:
  root=b/'outers'/('outer-%06d'%outer)
  value['rollout']={phase:[small(root/phase/('rank-%d-progress.json'%r)) for r in range(4)] for phase in ('prefix','full','full-pool')}
  value['QA']=small(root/'validation/complete.json')
  name=(value['stage'] or {}).get('stage','');value['actor_inflight']=[]
  if '-actor-' in name:
   import re
   attempt=name.rsplit('-actor-',1)[1];assert re.fullmatch('[0-9a-f]{32}',attempt)
   directory=b/'actor/outers'/('outer-%04d-attempt-%s'%(outer,attempt))
   advantage=small(directory/'advantage-statistics.json')
   value['advantage_summary']=None if advantage is None else advantage.get('summary')
   value['grounding_existing_logs']=[]
   for rank in range(4):
    logs=sorted((b/'logs').glob('grounding-*-rank-%d.log'%rank))
    if not logs:value['grounding_existing_logs'].append(None);continue
    path=logs[-1];info=path.stat()
    with path.open('rb') as f:
     if info.st_size>8*1024**2:f.seek(max(0,info.st_size-65536))
     rawlog=f.read(8*1024**2)
    lines=rawlog.decode(errors='replace').splitlines();requests=[line for line in lines if 'POST /judge_probe ' in line]
    value['grounding_existing_logs'].append(dict(rank=rank,name=path.name,bytes=info.st_size,mtime_ns=info.st_mtime_ns,
     counts_scope='entire_existing_log' if info.st_size<=8*1024**2 else 'tail64KiB_only',
     existing_HTTP200=sum(' 200 ' in line for line in requests),other_HTTP=len(requests)-sum(' 200 ' in line for line in requests)))
   for rank in range(4):
    path=directory/('rank-%d-updates.jsonl'%rank)
    if not path.exists():value['actor_inflight'].append(None);continue
    assert path.stat().st_size<2*1024**2;lines=path.read_text().splitlines();rows=[]
    for i,line in enumerate(lines):
     try:rows.append(json.loads(line))
     except json.JSONDecodeError:
      if i!=len(lines)-1:raise
    restored=next((x for x in rows if x['event']=='ACTUAL_RESTORE'),None)
    writes=[x for x in rows if x['event']=='ACTUAL_ADAM_WRITE']
    value['actor_inflight'].append(dict(rank=rank,restore=None if restored is None else
     {k:restored.get(k) for k in ('restored_completed_outer','actual_adam_before','restored_scheduler','actual_learning_rates','actual_global_micro')},
     returned_writes=len(writes),latest_actual_Adam=None if not writes else writes[-1]['actual_adam_steps']))
 for name in ('pipeline','prepare-actor','prepare-seed'):
  path=b/'logs'/(name+'.log')
  if path.exists():
   with path.open('rb') as f:f.seek(max(0,path.stat().st_size-4096));value[name+'_tail']=f.read().decode(errors='replace')
raw=json.dumps(value,allow_nan=False);assert len(raw.encode())<128*1024;print(raw)
'''


def main():
    p=argparse.ArgumentParser();p.add_argument('--action',choices=('stage','prepare','seed','submit','observe'),required=True)
    p.add_argument('--handle',type=Path,required=True);a=p.parse_args();w=Path(__file__).resolve().parents[2];payload=dict(action=a.action)
    if a.action=='stage':
        assert not a.handle.exists() and not subprocess.check_output(['git','status','--porcelain'],cwd=w).strip()
        archive,files=archive_source(w);execution=uuid.uuid4().hex
        h=dict(run_id=RUN,execution_id=execution,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=w,text=True).strip(),
            base='/data/shaoshitong/search-g1-prefix-selection200-a800-prefix5-'+RUN+'-resume-'+execution,
            source_files=files,archive_sha256=hashlib.sha256(archive).hexdigest(),created_unix=time.time())
        a.handle.write_text(json.dumps(h,indent=2)+'\n');payload.update(archive_b64=base64.b64encode(archive).decode(),
            warmup=json.loads((ROOT/'preparation/actor50-prefix20-fresh100-entryfix-deployment-20261018.json').read_text()),
            intent=json.loads((ROOT/'preparation/actor50-grpo-prefix20to40-human-intent-20261018.json').read_text()))
    else:h=json.loads(a.handle.read_text())
    payload['handle']=h;r=subprocess.run(route(REMOTE),input=json.dumps(payload).encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=65)
    out=ROOT/'preparation'/('actor50-grpo-'+a.action+'-'+str(time.time_ns())+'.json')
    out.write_text(json.dumps(dict(checked_unix=time.time(),exit_code=r.returncode,stdout=r.stdout.decode(),stderr=r.stderr.decode()),indent=2)+'\n')
    if r.returncode:raise RuntimeError('Inspect actual once-operation receipt: '+str(out))
    print(json.dumps(dict(receipt=str(out),result=json.loads(r.stdout))))


if __name__=='__main__':main()
