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
 if p.get('parent'):
  parent=pathlib.Path(p['parent']['base']);assert read(parent/'preparation/deployment.json')==p['parent']
  clean=read(parent/'logs/cleanup-status.json');assert clean['cleaned'] and clean['owned_births_all_gone']
  assert same(read(parent/'preparation/submit-receipt.json')['leader']) is None
  assert read(parent/'latest-joint.json')['outer']==50 and read(parent/'probe/latest.json')['actual_adam_steps_per_rank']==[0]*4
  c=read(parent/'preparation/launch-config.json')
  c.update(base=str(b),execution_id=h['execution_id'],source_commit=h['source_commit'],
   pool_state_path=str(b/'question-pool/state.json'),prefix_probe_receipt=str(b/'probe/latest.json'),
   grounding_rpc_batch=32,recovery_sampling=dict(source_base=str(parent),source_commit=p['parent']['source_commit'],
    source_execution_id=p['parent']['execution_id'],outer=51,zero_write_attempt='58c1b020c75940568ce59e5b56b454d9'))
  once(b/'preparation/recovery-parent-handle.json',p['parent'])
 if p.get('restart_parent'):
  parent=pathlib.Path(p['restart_parent']['base']);assert read(parent/'preparation/deployment.json')==p['restart_parent']
  clean=read(parent/'logs/cleanup-status.json');assert clean['cleaned'] and clean['owned_births_all_gone']
  assert same(read(parent/'preparation/submit-receipt.json')['leader']) is None
  stop=read(parent/'preparation/human-restart-mixed30.json')
  assert stop['intent']['stop_execution_id']==p['restart_parent']['execution_id']
  c=read(parent/'preparation/launch-config.json');c.pop('recovery_sampling',None)
  c.update(base=str(b),execution_id=h['execution_id'],source_commit=h['source_commit'],
   actor50_branch_schedule=BRANCH_SCHEDULE,pool_state_path=str(b/'question-pool/state.json'),
   prefix_probe_receipt=str(b/'probe/latest.json'),grounding_rpc_batch=32,
   restart_from_actor50=dict(source_base=str(parent),source_execution_id=p['restart_parent']['execution_id'],
    human_intent_sha256=hashlib.sha256(json.dumps(p['intent'],sort_keys=True).encode()).hexdigest()))
  once(b/'preparation/restart-parent-handle.json',p['restart_parent'])
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
elif action in ('prepare','seed','recover','restart'):
 source();c=read(b/'preparation/launch-config.json');phase={'prepare':'actor','seed':'seed','recover':'recover','restart':'restart'}[action];name='prepare-'+phase
 assert not (b/'preparation'/(name+'-intent.json')).exists(),'Onceclaim exists: inspect, do not replay'
 if action=='seed':
  assert read(b/'preparation/prepare-actor-exit.json')['success']
  wh=read(b/'preparation/warmup-parent-handle.json');clean=read(pathlib.Path(wh['base'])/'logs/cleanup-status.json')
  assert clean['cleaned'] and clean['owned_births_all_gone']
 once(b/'preparation'/(name+'-intent.json'),dict(checked_unix=time.time()))
 command=([str(pathlib.Path(c['environment_root'])/'bin/python'),'-m','reproduction.actor50_grpo.'+('restart' if action=='restart' else 'recovery'),'--base',str(b)]
  if action in ('recover','restart') else [str(pathlib.Path(c['environment_root'])/'bin/python'),'-m','reproduction.actor50_grpo.bootstrap',phase,'--base',str(b)])
 value=launch(name,command,c)
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
 def scalar_metrics(item):
  keys=('count','mixed_count','nonmixed_count','accuracy','balanced_accuracy','tp','fp','fn','tn',
   'mixed_recall','nonmixed_recall','threshold','formula','true_mixed','trajectory_bce','group_bce','total_loss','group_weight')
  return None if item is None else {k:item[k] for k in keys if k in item}
 def qa_summary(item):
  keys=('run_id','actor_version','evaluation_contract','interaction_protocol','actual_compute_dtype','qa_count',
   'question_source_sha256','em','f1','per_source','completed_utc','actual_worker_exit_codes')
  return None if item is None else {k:item[k] for k in keys if k in item}
 def audit_json(path):
  # Read only bounded existing metadata, never the checkpoint/head files it names.
  path=pathlib.Path(path);assert path.is_relative_to(b) and path.resolve()==path and not path.is_symlink()
  if not path.is_file():return None,dict(metadata_status='NOT_YET_RECORDED')
  size=path.stat().st_size
  if size>1024**2:return None,dict(metadata_status='ABOVE_1MIB_READ_BUDGET',bytes=size)
  raw=path.read_bytes();assert len(raw)<=1024**2
  return json.loads(raw),dict(metadata_status='PRESENT',metadata_sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
 def grounding_summary(item):
  keys=('run_id','round','version','source_actor_version','encoder','encoder_sha256','head_sha256',
   'manifest_sha256','adoption_receipt_sha256','freshly_fitted_in_this_run','imported_actual_round0_fit',
   'source_actor_counters')
  return None if item is None else {k:item[k] for k in keys if k in item}
 transition,transition_meta=audit_json(b/'probe/prefix40-transition.json')
 value_transition=dict(transition_meta)
 if transition is not None:
  keys=('kind','source_sha256','source_completed_outer','actual_adam_steps_per_rank',
   'original_model_and_moments_bit_preserved','new20_head_columns_and_moments_zero',
   'optimizer_reset','rank_RNG_preserved','checked_unix')
  value_transition['receipt']={k:transition[k] for k in keys if k in transition}
  value_transition['checkpoint']={k:v for k,v in transition.get('checkpoint',{}).items() if k in
   ('sha256','bytes','completed_outer','actual_adam_steps_per_rank','binding_sha256','learning_rate')}
 grounding_latest,grounding_latest_meta=audit_json(b/'grounding/latest.json')
 grounding_lifecycle=dict(latest=dict(grounding_latest_meta,state=grounding_summary(grounding_latest)),milestones=[])
 if grounding_latest is not None and grounding_latest.get('adoption_receipt'):
  adopted,adopted_meta=audit_json(grounding_latest['adoption_receipt'])
  grounding_lifecycle['latest_adoption']=dict(adopted_meta)
  if adopted is not None:
   grounding_lifecycle['latest_adoption'].update(
    receipt_sha256_matches=adopted_meta['metadata_sha256']==grounding_latest.get('adoption_receipt_sha256'),
    run_id=adopted.get('run_id'),round=adopted.get('round'),version=adopted.get('version'),
    manifest_sha256=adopted.get('manifest_sha256'),adopted_unix=adopted.get('adopted_unix'),
    replicas=[{k:row[k] for k in ('rank','ok','round','version','encoder','owned_service','pid') if k in row}
     for row in adopted.get('replicas',[])])
 for milestone in (100,150):
  fitted,fit_meta=audit_json(b/'grounding/rounds'/('round-%06d'%milestone)/'fit-complete.json')
  refreshed,refresh_meta=audit_json(b/'outers'/('outer-%06d'%milestone)/'grounding-refresh.json')
  grounding_lifecycle['milestones'].append(dict(outer=milestone,fit=dict(fit_meta,state=grounding_summary(fitted)),
   refresh=dict(refresh_meta,run_id=None if refreshed is None else refreshed.get('run_id'),
    outer=None if refreshed is None else refreshed.get('outer'),
    required_before_outer=None if refreshed is None else refreshed.get('required_before_outer'),
    state=None if refreshed is None else grounding_summary(refreshed.get('grounding')))))
 value=dict(run_id=h['run_id'],execution_id=h['execution_id'],source_commit=h['source_commit'],checked_unix=time.time(),
  prepare_progress=small(b/'preparation/prepare-progress.json'),prepare_actor_exit=small(b/'preparation/prepare-actor-exit.json'),
  prepare_seed_exit=small(b/'preparation/prepare-seed-exit.json'),stage=small(b/'logs/pipeline-status.json'),
  prepare_recover_exit=small(b/'preparation/prepare-recover-exit.json'),
  prepare_restart_exit=small(b/'preparation/prepare-restart-exit.json'),
  probe=small(b/'probe/latest.json'),failure=small(b/'logs/failure.json'),exit=small(b/'logs/exit.json'),
  cleanup=small(b/'logs/cleanup-status.json'),prefix40_transition=value_transition,grounding_lifecycle=grounding_lifecycle)
 completed=small(b/'logs/pipeline-complete.json')
 value['complete']=None if completed is None else dict(run_id=completed['run_id'],complete_outers=completed['complete_outers'],
  final_evaluation=qa_summary(completed['final_evaluation']))
 probe_outer=(value['probe'] or {}).get('completed_outer',50)
 value['probe_diagnostics']=None
 if 50<probe_outer<200 and probe_outer%10==0:
  directory=b/'outers'/('outer-%06d'%probe_outer)/'probe-update'
  item=small(directory/'complete.json')
  value['probe_diagnostics']=dict(outer=probe_outer,status='PENDING_EXISTING_COMPLETE_METADATA')
  if item is not None:
   assert item['run_id']==h['run_id'] and item['outer']==probe_outer
   assert item['metrics_sha256']==sha(directory/'metrics.json')
   value['probe_diagnostics']=dict(outer=probe_outer,status=item['status'],
    actual_adam_steps_per_rank=item['actual_adam_steps_per_rank'],new_adam_steps_per_rank=item['new_adam_steps_per_rank'],
    actual_learning_rate=item['actual_learning_rate'],actor_version=item['actor_version'],
    before_update={split:scalar_metrics(item['before_update'].get(split)) for split in ('train','test')},
    after_update={split:scalar_metrics(item.get(split)) for split in ('train','test')},
    diagnostic_semantics=item.get('diagnostic_semantics'),
    unselected_training_questions=item['unselected_training_questions'],
    checkpoint={k:v for k,v in item['checkpoint'].items() if k in
     ('sha256','bytes','completed_outer','actual_adam_steps_per_rank','binding_sha256','learning_rate')},
    complete_metadata_sha256=sha(directory/'complete.json'),metrics_sha256=item['metrics_sha256'],
    scored_features_manifest_sha256=item['scored_features_manifest_sha256'],own_em_sha256=item['own_em_sha256'],
    selection_sha256=item['selection_sha256'],observed_utc=item.get('observed_utc'))
 value['workers']={}
 for name in ('prepare-actor','prepare-seed','prepare-recover','prepare-restart','pipeline'):
  item=small(b/'preparation'/(name+'-launch.json'));value['workers'][name]=same(item['leader']) if item else None
 joint_path=b/'latest-joint.json';joint_raw=joint_path.read_bytes() if joint_path.is_file() else None
 assert joint_raw is None or len(joint_raw)<8*1024**2
 joint=json.loads(joint_raw) if joint_raw is not None else None
 if joint:
  a=joint['actor'];value['joint']=dict(outer=joint['outer'],actor_Adam=a['actual_adam_steps_per_rank'],
   scheduler=a['scheduler_completed_outer'],probe=joint['probe'],rollout_mixed=joint['rollout'].get('actual_selected_mixed_fraction'),
   source_commit=joint['source_commit'],actor50_branch_plan=joint.get('actor50_branch_plan'),
   metadata_sha256=hashlib.sha256(joint_raw).hexdigest(),new_actor_Adam=a.get('new_adam_steps_per_rank'),
   processed_opportunities=a.get('opportunities_processed'),skipped_opportunities=a.get('skipped_update_opportunities'),
   mixed_floor=joint['rollout'].get('mixed_floor'),actor_version=a.get('actor_version'),
   grounding=grounding_summary(joint.get('grounding')),actor_grounding=grounding_summary(a.get('grounding_state')),
   grounding_snapshot_sha256=joint.get('grounding_snapshot_sha256'))
 value['recent_complete_metadata']=[]
 if joint:
  for ordinal in range(max(51,joint['outer']-4),joint['outer']+1):
   recent_path=b/'outers'/('outer-%06d'%ordinal)/'complete.json'
   if not recent_path.is_file():
    value['recent_complete_metadata'].append(dict(outer=ordinal,metadata_status='MISSING'));continue
   assert recent_path.resolve()==recent_path and not recent_path.is_symlink()
   size=recent_path.stat().st_size
   if size>1024**2:
    value['recent_complete_metadata'].append(dict(outer=ordinal,metadata_status='ABOVE_1MIB_READ_BUDGET',bytes=size));continue
   recent_raw=recent_path.read_bytes();assert len(recent_raw)<=1024**2
   recent=json.loads(recent_raw);assert recent['run_id']==h['run_id'] and recent['outer']==ordinal
   actor=recent['actor'];probe=recent['probe'];floor=recent['rollout'].get('mixed_floor') or {}
   floor_keys=('initial_mixed','final_mixed','initial_fraction','final_fraction','extra_native_batches',
    'completed_candidate_questions','remaining_uncontinued_questions','triggered','shortfall','exhausted',
    'target_mixed','target_reached','native_prefix_or_score_regenerated')
   value['recent_complete_metadata'].append(dict(outer=ordinal,metadata_status='PRESENT',
    source_commit=recent.get('source_commit'),metadata_sha256=hashlib.sha256(recent_raw).hexdigest(),
    actor_Adam=actor['actual_adam_steps_per_rank'],scheduler=actor['scheduler_completed_outer'],
    new_actor_Adam=actor.get('new_adam_steps_per_rank'),
    processed_opportunities=actor.get('opportunities_processed'),skipped_opportunities=actor.get('skipped_update_opportunities'),
    probe_completed_outer=probe.get('completed_outer'),probe_Adam=probe.get('actual_adam_steps_per_rank'),
    rollout_mixed=recent['rollout'].get('actual_selected_mixed_fraction'),
    mixed_floor={k:floor[k] for k in floor_keys if k in floor},replacement_count=len(floor.get('replacements',[]))))
 value['latest_milestone_QA']=None
 for milestone in (200,150,100):
  if joint and joint['outer']>=milestone:
   directory=b/'outers'/('outer-%06d'%milestone)/'validation';item=small(directory/'complete.json')
   if item is not None:
    assert item['run_id']==h['run_id']
    value['latest_milestone_QA']=dict(outer=milestone,metrics=qa_summary(item),
     complete_metadata_sha256=sha(directory/'complete.json'))
    break
 outer=(value['stage'] or {}).get('outer')
 if outer:
  root=b/'outers'/('outer-%06d'%outer)
  value['rollout']={phase:[small(root/phase/('rank-%d-progress.json'%r)) for r in range(4)] for phase in ('prefix','full','full-pool')}
  value['QA']=small(root/'validation/complete.json')
  value['mixed_floor']=small(root/'mixed-floor-complete.json')
  value['mixed_floor_progress']=small(root/'mixed-floor-progress.json')
  fp=value['mixed_floor_progress'] or {};batch=fp.get('batch')
  import re
  if batch and re.fullmatch(r'floor-(initial|extra-\d{4})',batch):
   directory=root/('full-pool' if (outer%10==0 and outer<200) else batch)
   value['floor_native_progress']=[small(directory/('rank-%d-progress.json'%r)) for r in range(4)]
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
 for name in ('pipeline','prepare-actor','prepare-seed','prepare-recover','prepare-restart'):
  path=b/'logs'/(name+'.log')
  if path.exists():
   with path.open('rb') as f:f.seek(max(0,path.stat().st_size-4096));value[name+'_tail']=f.read().decode(errors='replace')
raw=json.dumps(value,allow_nan=False);assert len(raw.encode())<128*1024;print(raw)
'''


def main():
    p=argparse.ArgumentParser();p.add_argument('--action',choices=('stage','prepare','seed','recover','restart','submit','observe'),required=True)
    p.add_argument('--handle',type=Path,required=True);p.add_argument('--parent-handle',type=Path);p.add_argument('--restart-handle',type=Path);a=p.parse_args();w=Path(__file__).resolve().parents[2];payload=dict(action=a.action)
    if a.action=='stage':
        assert not a.handle.exists() and not subprocess.check_output(['git','status','--porcelain'],cwd=w).strip()
        archive,files=archive_source(w);execution=uuid.uuid4().hex
        h=dict(run_id=RUN,execution_id=execution,source_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=w,text=True).strip(),
            base='/data/shaoshitong/search-g1-prefix-selection200-a800-prefix5-'+RUN+'-resume-'+execution,
            source_files=files,archive_sha256=hashlib.sha256(archive).hexdigest(),created_unix=time.time())
        a.handle.write_text(json.dumps(h,indent=2)+'\n');payload.update(archive_b64=base64.b64encode(archive).decode(),
            warmup=json.loads((ROOT/'preparation/actor50-prefix20-fresh100-entryfix-deployment-20261018.json').read_text()),
            intent=json.loads((ROOT/'preparation/actor50-grpo-prefix20to40-human-intent-20261018.json').read_text()))
        if a.parent_handle:payload['parent']=json.loads(a.parent_handle.read_text())
        if a.restart_handle:
            assert not a.parent_handle
            payload['restart_parent']=json.loads(a.restart_handle.read_text())
            payload['intent']=json.loads((ROOT/'preparation/actor50-grpo-mixed30-human-intent-20261018.json').read_text())
    else:h=json.loads(a.handle.read_text())
    payload['handle']=h;r=subprocess.run(route(REMOTE),input=json.dumps(payload).encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=65)
    out=ROOT/'preparation'/('actor50-grpo-'+a.action+'-'+str(time.time_ns())+'.json')
    out.write_text(json.dumps(dict(checked_unix=time.time(),exit_code=r.returncode,stdout=r.stdout.decode(),stderr=r.stderr.decode()),indent=2)+'\n')
    if r.returncode:raise RuntimeError('Inspect actual once-operation receipt: '+str(out))
    print(json.dumps(dict(receipt=str(out),result=json.loads(r.stdout))))


if __name__=='__main__':main()
