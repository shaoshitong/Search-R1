"""One bounded, read-only epoch2 facts capture; no model/rollout payload hashing."""
from pathlib import Path
import sys,json,hashlib,datetime
sys.path.insert(0,'/Users/shaoshitong/Documents/Search-G1/qprobe-prefix20-online1000-h1008-20261016')
from reproduction.h1008_qprobe.management_transport import remote
W=Path(__file__).parent
script=r'''
from pathlib import Path
import json,hashlib,datetime
B=Path('/mnt/data/shaoshitong/search-g1-prefix20-online1000-20261016-ca1c60f');E=B/'epochs/epoch-0002';T=B/'train/epochs/epoch-0002'
MAX=16*1024*1024
files=[]
def small(p,jsonl=False):
 p=Path(p);before=p.stat();assert before.st_size<=MAX,('Refuse large-file read',str(p),before.st_size)
 data=p.read_bytes();after=p.stat();assert(before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)
 proof=dict(path=str(p),bytes=len(data),independent_small_file_full_sha256=hashlib.sha256(data).hexdigest())
 files.append(proof)
 return ([json.loads(x)for x in data.decode().splitlines()if x]if jsonl else json.loads(data)),proof
sel,selproof=small(E/'selection.json');qs,qproof=small(E/'questions.json');roll,rproof=small(E/'complete.json');complete,cproof=small(T/'complete.json');metrics,mproof=small(T/'metrics.json');pool,pproof=small(E/'pool-state-at-head-completion.json');bundle,bproof=small(B/'train/latest-bundle.json');stage,stageproof=small(B/'logs/pipeline-status.json')
assert roll['epoch']==complete['epoch']==metrics['epoch']==sel['epoch']==qs['epoch']==2
assert roll['questions']==5120 and roll['trajectories']==25600 and complete['actual_adam_steps_per_rank']==[200]*8
assert qs['selection_sha256']==selproof['independent_small_file_full_sha256']==roll['selection_sha256']
assert roll['questions_sha256']==qproof['independent_small_file_full_sha256']
qids=[str(x['qid'])for x in sel['rows']];assert len(qids)==len(set(qids))==5120
prior,_=small(B/'epochs/epoch-0001/selection.json');priorq={str(x['qid'])for x in prior['rows']}
rankfacts=[];assets=[]
for rank in range(8):
 rr,rp=small(E/'rollout'/('rank-%d-complete.json'%rank));updates,up=small(T/('rank-%d-updates.jsonl'%rank),True)
 assert rr['rank']==rank and rr['epoch']==2 and rr['questions']==640 and rr['trajectories']==3200
 counters=[v['actual_adam_steps']for v in updates];assert counters==list(range(101,201)) and all(v['epoch']==2 and v['local_trajectories']==32 for v in updates)
 rankfacts.append(dict(rank=rank,rollout_questions=rr['questions'],rollout_trajectories=rr['trajectories'],strict_correct_trajectories=rr['strict_correct_trajectories'],rollout_elapsed_seconds=rr['elapsed_seconds'],complete_receipt=rp,actual_adam_before=100,actual_adam_after=200,actual_adam_increment=100,actual_update_records=100,update_ledger=up,first_update=updates[0],last_update=updates[-1]))
 for role,key in [('reusable_compact_prefix','prefix'),('original_full_rollout','full')]:
  a=rr[key];p=Path(a['path']);st=p.stat();assert st.st_size==a['bytes']
  assets.append(dict(role=role,rank=rank,path=str(p),bytes=st.st_size,count=a['count'],producer_receipt_sha256=a['sha256'],independent_payload_full_sha_performed=False,sha_semantics='Producer receipt hash only; payload not independently reread by this audit.'))
phase=[]
for log in [B/'logs/native-3408.log']:
 if log.exists():
  assert log.stat().st_size<=4*1024*1024
  for line in log.read_text(errors='replace').splitlines():
   if 'PREFIX20_PIPELINE='not in line:continue
   try:v=json.loads(line.split('PREFIX20_PIPELINE=',1)[1])
   except ValueError:continue
   if v.get('epoch')==2 or(v.get('epoch')==3 and v.get('stage')=='e0003-rollout'):phase.append(v)
phasecost={}
for a,b in zip(phase,phase[1:]):
 if a.get('epoch')==2 and a['stage'].startswith('e0002-'):
  phasecost[a['stage']]=b['checked_unix']-a['checked_unix']
feature_receipts=[]
for rank in range(8):
 p=E/'features'/('shard-%d-complete.json'%rank)
 if p.exists():
  sh,proof=small(p);feature_receipts.append(dict(rank=rank,count=sh['count'],total_tokens=sh['total_tokens'],elapsed_seconds=sh['elapsed_seconds'],receipt=proof))
pruned=None
if(E/'derived-cache-pruned.json').exists():pruned,_=small(E/'derived-cache-pruned.json')
latest_head_metadata=dict(epoch=bundle['epoch'],path=bundle['head_path'],producer_checkpoint_sha256=bundle['head_sha256'],independent_model_hash_performed=False)
result=dict(role='FIRST_FRESH_EPOCH2_ACTUAL_FACTS',observed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),job_id=3408,run_id=roll['run_id'],runtime_source_commit='ca1c60f49a29c30ca299c1d6073ee9a4bb86655b',base=str(B),epoch=2,fresh_rollout_complete=True,questions=5120,unique_question_ids=5120,question_overlap_with_cached_epoch1=len(set(qids)&priorq),trajectories=25600,rank_facts=rankfacts,actual_adam_before_per_rank=[100]*8,actual_adam_after_per_rank=[200]*8,actual_adam_increment_per_rank=[100]*8,rank_Adam_not_summed=True,questions_bundle=qproof,selection=selproof,rollout_complete=rproof,train_complete=cproof,metrics=metrics,metrics_proof=mproof,retained_rollout_assets=assets,pool_boundary=dict(path=pproof['path'],sha256=pproof['independent_small_file_full_sha256'],cursor=pool.get('cursor'),next_epoch=pool.get('next_epoch'),cycle=pool.get('cycle')),latest_bundle=bundle,latest_bundle_proof=bproof,head_checkpoint_metadata=latest_head_metadata,model_bytes_downloaded=False,model_independent_full_hash_performed=False,console_phase_records=phase,phase_wall_seconds_including_startup_cleanup_polling_and_publication=phasecost,training_loop_seconds=metrics['train_seconds'],fixed_diagnostic_seconds=metrics['diagnostic_seconds'],encoder_rank_receipts=feature_receipts,derived_cache_pruned=pruned,current_stage=stage,epoch3_start_observed=stage.get('epoch',0)>=3,small_file_readback_proofs=files,GPU_computation_performed=False,remote_files_written=False,training_state_or_automation_modified=False)
assert metrics['raw']['count']==2560 and metrics['sampling']=='natural_no_replacement'and metrics['train_unique_trajectories']==25600
print(json.dumps(result),flush=True)
'''
e=W/'prefix20-online-epoch2-actual-facts-transport-v1.json'
out=remote(script,e,control_path=None,target='h100-8');facts=json.loads(out);facts['transport_proof']=dict(path=str(e.resolve()),sha256=hashlib.sha256(e.read_bytes()).hexdigest())
p=W/'prefix20-online-epoch2-actual-facts-v1.json'
with p.open('x')as f:json.dump(facts,f,indent=2,sort_keys=True);f.write('\n')
print(json.dumps(dict(path=str(p.resolve()),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),epoch=facts['epoch'],questions=facts['questions'],trajectories=facts['trajectories'],actual_adam_after=facts['actual_adam_after_per_rank'],metrics=facts['metrics'],phase_costs=facts['phase_wall_seconds_including_startup_cleanup_polling_and_publication'],epoch3_start_observed=facts['epoch3_start_observed']),indent=2))
