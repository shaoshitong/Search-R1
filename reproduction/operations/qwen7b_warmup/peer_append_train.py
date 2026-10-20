"""Honest one-current-rank CP256 continuation: only682 unseen-tail updates.

The old256 batches are immutable history, never a new epoch. No E1 counter
rewrites, optimizer resets, ranking losses, actor updates, or live-code edits.
Preparation is metadata-only; actual fullstate restore/replay happens on GPU.
"""
import argparse,copy,fcntl,hashlib,json,os,platform,shlex,signal,subprocess,sys,time,traceback
from pathlib import Path
from . import peer_append_only_plan as planlib

B=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019')
A=B/'algorithm-improvement-20261021';CONTROL=A/'peer-newseed8192-g10-train-v1'
TARGET=A/'peer-expanded-append-only-train-v1';PKG='reproduction/operations/qwen7b_warmup/'
D8='d8f74420aa09fa56e8ec59207c6d1453fa86f95e';CALLBACK='48eb944281d2e3d6fa5ad66e4d6f0f9eaf18aeec'
SOURCE_SHA='36d9bcc3be71ba94eaca3a1e1b292c0ed9a65297fb96a746ea5295432f09c000'
SOURCE_CONFIG_SHA='16b2fd34e3e8907307b3866298b33216478fd77c2dc1b9ca80638e0929800130'
SELECTION=dict(path=str(A/'peer-expanded-append-only-plan-v1/source-selection.json'),bytes=4166,sha256='aa22a83fba470f9f1b014765eb5016ed1a5d116c9d1e97af637e27b7728e3e0b')
HISTORY,END,SOURCE_ADAM=256,938,10720
STOP=False

def read(p):return json.loads(Path(p).read_text())
def proof(p):
 p=Path(p);h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())
def checked(v):
 p=Path(v['path'])
 if not p.is_absolute() or p.resolve()!=p or not str(p).startswith(str(B)+'/'):raise ValueError('Remote nonsymlink shared input required')
 actual=proof(p)
 if actual['sha256']!=v['sha256'] or ('bytes' in v and actual['bytes']!=v['bytes']):raise ValueError('Input SHA/bytes changed: '+str(p))
 return p
def atomic(p,v):
 p=Path(p);tmp=p.with_suffix('.tmp')
 with tmp.open('w') as f:json.dump(v,f,sort_keys=True,allow_nan=False);f.flush();os.fsync(f.fileno())
 tmp.replace(p)
def immutable(p,v):
 if Path(p).exists():
  if read(p)!=v:raise ValueError('Immutable output differs: '+str(p))
 else:atomic(p,v)

def accepted_selection():
 s=read(checked(SELECTION))
 if (s['source_checkpoint']['sha256']!=SOURCE_SHA or s['source_config']['sha256']!=SOURCE_CONFIG_SHA
  or s['source_cursor']!=256 or s['source_Adam']!=10720 or s['new_updates']!=682
  or s['cumulative_target_cursor']!=938 or s['cumulative_target_Adam']!=11402
  or s['source_world_size']!=1 or s['current_world_size']!=1 or s['original_E1_archive_world_size']!=4
  or s['loss_kwargs']!=dict(trajectory_weight=1.,three_event_weight=1.)
  or any(s[k] is not False for k in ('optimizer_reset','readout_reset','block_reset','final_pool_read'))
  or any(x['joint_pass'] for x in s['ranking_finals'].values())):raise ValueError('Only precommitted no-rank peer CP256 fallback accepted')
 for key in ('source_config','source_complete','source_plan','append_plan','new_bank','new_bank_independent_audit'):checked(s[key])
 cp=Path(s['source_checkpoint']['path'])
 if cp.resolve()!=cp or cp.stat().st_size!=s['source_checkpoint']['bytes']:raise ValueError('Source CP absent/truncated')
 return s

def configured(old,pub,s):
 if (old['source_adam_steps']!=10464 or old['update_budget']!=256 or old['global_questions']!=32
  or old['fit_questions']!=8192 or old['fit_real_trajectories']!=81920
  or old['recipe']['loss_kwargs']!=s['loss_kwargs']
  or old['grouping']!=dict(mode='random',views_per_question=4,seed=1729,on_unbalance='raise')
  or old['optimizer_reset'] is not False):raise ValueError('Exact unchanged original peer protocol required')
 c=copy.deepcopy(old)
 c.update(source_checkpoint=s['source_checkpoint'],source_config=s['source_config'],source_adam_steps=10720,
  original_E1_checkpoint=old['source_checkpoint'],original_E1_config=old['source_config'],
  source_selection=SELECTION,source_complete=s['source_complete'],original_plan=s['source_plan'],
  g10_complete=s['new_bank'],expanded_bank_independent_audit=s['new_bank_independent_audit'],plan=s['append_plan'],
  fit_questions=30016,fit_real_trajectories=300160,additional_fit_questions=21824,additional_fit_real_trajectories=218240,
  historical_updates=256,new_update_budget=682,update_budget=938,source_cursor=256,
  source_world_size=1,current_world_size=1,hardware_topology_changed=False,original_E1_archive_world_size=4,
  historical_prefix_retrained=False,equal_budget_to_peer256_claim=False,
  evaluation_cumulative_cursors=[512,768,938],
  source_commit=pub['commit'],code_hashes={n:v['sha256'] for n,v in pub['files'].items()})
 c['domain_scope']['question_counts']={'nq':30016}
 c['recipe']['id']='peer-NQ30016-append-only-CP256-plus682-no-rank'
 return c

def accepted_inputs(c):
 """Check own metadata and exact old-prefix/new-tail plan; never read tensors."""
 s=accepted_selection();bank=read(checked(c['g10_complete']))
 if (bank['status']!='EXPANDED_REAL_G10_LAYER26_REUSED8192_PLUS_ADDITIONAL21824_ACCEPTED'
  or bank['questions']!=30016 or bank['trajectories']!=300160 or bank['prior_questions_reused']!=8192
  or bank['new_questions_captured']!=21824 or bank['capture_layer']!=26 or bank['prefix_tokens']!=20
  or bank['question_max_tokens']!=273 or bank['actor_updates']!=0 or bank['final_pool_read'] is not False
  or bank['labels_consulted_for_identity_selection'] is not False):raise ValueError('Actual accepted expanded L26/P20 bank required')
 data={k:read(checked(v)) for k,v in bank['outputs'].items()}
 groups=data['groups.json']['train'];qids=[g['qid'] for g in groups];hashes=[g['question_hash'] for g in groups];tids=[]
 for g in groups:
  if (len(g['trajectory_ids'])!=10 or len(g['labels'])!=10 or len(set(g['trajectory_ids']))!=10
   or any(type(y) is not int or y not in (0,1) for y in g['labels'])):raise ValueError('Ten distinct real IDs and own binary EM required')
  tids.extend(g['trajectory_ids'])
 if len(groups)!=30016 or len(set(qids))!=30016 or len(set(hashes))!=30016 or len(set(tids))!=300160:raise ValueError('Question/hash/trajectory identities repeat')
 identity={'groups':[{k:g[k] for k in ('qid','question_hash','trajectory_ids')} for g in groups]}
 plan=read(checked(c['plan']));oldplan=read(checked(c['original_plan']))
 if planlib.extend_plan(oldplan,identity)!=plan:raise ValueError('Frozen938 plan or historical256 prefix changed')
 # Read the already accepted additional manifests only; old8192 never forwards again.
 allrefs=data['features-layer26.json'];newroot=str(A/'peer-expanded-L26-replay-v1')+'/'
 additional=[v for v in allrefs if v['path'].startswith(newroot)]
 if not additional or len({v['path'] for v in allrefs})!=len(allrefs):raise ValueError('Missing/repeated actual expanded feature descriptors')
 return groups,additional,plan,bank

def validate_evaluations(c,groups):
 from .probe_g10_train import frozen_validation
 out=[];ids=set();hashes=set()
 for e in c['evaluations']:
  if e['role']!='validation' or e['backend']!='huggingface_frozen_causal_prefill':raise ValueError('Same frozen HF validation required')
  bank=read(checked(e['bank']))[e['split']];frozen=frozen_validation(bank,read(checked(e['fixed_g5'])));checked(e['features'])
  own={g['qid'] for g in bank};h={g['question_hash'] for g in bank}
  if ids&own or hashes&h or len(h)!=len(bank):raise ValueError('Validation scopes overlap')
  ids|=own;hashes|=h;out.append((e,frozen))
 if (len(out)!=2 or {g['qid'] for g in groups}&ids or {g['question_hash'] for g in groups}&hashes):raise ValueError('Train/evaluation question identity overlap')
 return out

def render(code,sha):
 return '''#!/usr/bin/env bash
#SBATCH --job-name=q7b-peer-append682
#SBATCH --partition=acd_u
#SBATCH --account=mou176
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --signal=USR1@180
#SBATCH --output=@@BASE@@/logs/train-%j.log
#SBATCH --chdir=@@BASE@@
set -euo pipefail
export QWEN7B_CODE_ROOT=@@CODE@@
export FLASH_ATTENTION_DETERMINISTIC=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
srun --kill-on-bad-exit=1 bash "$QWEN7B_CODE_ROOT/reproduction/operations/qwen7b_warmup/runtime.sh" -u -m reproduction.operations.qwen7b_warmup.peer_append_train --action train --config-sha256 @@SHA@@
'''.replace('@@BASE@@',str(TARGET)).replace('@@CODE@@',shlex.quote(str(code))).replace('@@SHA@@',sha)

def prepare():
 s=accepted_selection();old=read(checked(s['source_config']));pub=read(TARGET/'source-publication.json');root=Path(pub['code_root'])
 for n,v in pub['files'].items():checked(dict(path=str(root/n),**v))
 c=configured(old,pub,s);groups,refs,plan,bank=accepted_inputs(c);validate_evaluations(c,groups)
 immutable(TARGET/'config.json',c);text=render(root,proof(TARGET/'config.json')['sha256']);p=TARGET/'training.sbatch'
 if p.exists() and p.read_text()!=text:raise ValueError('Immutable append-only sbatch differs')
 if not p.exists():p.write_text(text)
 done=dict(status='APPEND_ONLY_CP256_PLUS682_METADATA_AND_PLAN_ACCEPTED_NO_GPU_SUBMISSION',config=proof(TARGET/'config.json'),sbatch=proof(p),source_selection=SELECTION,
  source_checkpoint=s['source_checkpoint'],source_cursor=256,source_Adam=10720,cumulative_cursor=938,cumulative_Adam=11402,
  historical_updates_not_retrained=256,additional_updates=682,additional_questions=21824,additional_feature_manifests=len(refs),
  source_world_size=1,current_world_size=1,hardware_topology_changed=False,original_E1_archive_world_size=4,
  source_CP_full_SHA_already_bound_by_selection=True,CP_tensor_loaded_in_preparation=False,full_GPU_restore_and_replay_still_required=True)
 immutable(TARGET/'preparation.json',done);return done

def full_blob(model,ema,opt,device,cursor,binding,archive):
 from . import probe_ablation_train as u
 return dict(role='APPEND_ONLY_EXISTING_SINGLE_RANK_FULLSTATE_CONTINUATION',phase='ACCEPTED_UPDATE_SAVED_BEFORE_EXTERNAL_EVALUATION',
  cursor=cursor,steps=u.steps(opt),raw=model.state_dict(),ema=ema.state_dict(),optimizer=opt.state_dict(),
  raw_buffers=u.buffers(model),ema_buffers=u.buffers(ema),rng=u.rng_state(device),accepted_hashes=u.state_hashes(model,ema,opt,device),
  source_all_rank_states=archive,source_world_size=1,current_world_size=1,hardware_topology_changed=False,
  original_E1_archive_world_size=4,four_rank_bitexact_resume_claim=False,
  actor_updates=0,optimizer_reset=False,readout_reset=False,block_reset=False,**binding)

def train_update(model,ema,opt,reader,lookup,qids,c,device,epoch):
 from . import probe_g10_train as t
 if c['recipe']['loss_kwargs']!=dict(trajectory_weight=1.,three_event_weight=1.):raise ValueError('No failed ranking recipe may enter append-only baseline')
 return t.run_update(model,ema,opt,reader,lookup,qids,c,device,epoch)

def evaluate_at(cursor,ema,model,opt,device,evaluations,binding):
 """Checkpoint-first fixed evaluation, idempotent receipts, no state writes."""
 from . import probe_ablation_train as u
 from .replay_feature_reader import ReplayFeatureReader
 base=TARGET/'evaluations'/('cursor-%06d'%cursor);base.mkdir(parents=True,exist_ok=True)
 complete=base/'complete.json'
 if complete.exists():
  saved=read(complete)
  if (saved['cursor']!=cursor or saved['config_sha256']!=binding['config_sha256']
   or saved['training_six_components_unchanged'] is not True):raise ValueError('Existing evaluation receipt lineage differs')
  checked(saved['checkpoint'])
  for v in saved['predictions'].values():checked(v)
  return saved['metrics']
 cp=read(TARGET/'latest.json')
 if cp['cursor']!=cursor:raise ValueError('Exact current full checkpoint must precede this evaluation')
 before=u.state_hashes(model,ema,opt,device);metrics={};predictions={}
 for e,frozen in evaluations:
  er=ReplayFeatureReader(read(checked(e['features'])),26);rng=u.rng_state(device);native=u.buffers(ema)
  try:metrics[e['name']]=u.evaluate(ema,er,frozen,device,base/(e['name']+'.jsonl.gz'))
  finally:u.restore_rng(rng,device);u.restore_buffers(ema,native);er.close()
  predictions[e['name']]=proof(base/(e['name']+'.jsonl.gz'))
 after=u.state_hashes(model,ema,opt,device)
 if before!=after:raise ValueError('Fixed evaluation changed training model/EMA/Adam/buffers/RNG')
 immutable(complete,dict(status='APPEND_CHECKPOINT_FIRST_FIXED_HF_EVALUATION_ACCEPTED',cursor=cursor,steps=u.steps(opt),
  config_sha256=binding['config_sha256'],checkpoint=cp,metrics=metrics,predictions=predictions,
  before_six_component_hashes=before,after_six_component_hashes=after,training_six_components_unchanged=True,
  actor_updates=0,extra_optimizer_updates=0,final_pool_read=False,time=time.time()))
 return metrics

def audit_first():
 import torch
 from . import probe_ablation_train as u
 deadline=time.time()+3600
 while not (TARGET/'progress.json').exists() or read(TARGET/'progress.json')['cursor']<257:
  if (TARGET/'failure.json').exists():raise ValueError('Append trainer failed before accepted CP257')
  if time.time()>deadline:raise ValueError('Bounded first-update audit wait expired')
  time.sleep(1)
 p=TARGET/'checkpoints/cursor-000257.pt';v=torch.load(p,map_location='cpu',mmap=True,weights_only=False);replay=read(TARGET/'next-update-replay.json')
 if (v['cursor']!=257 or v['steps']!=10721 or v['config_sha256']!=proof(TARGET/'config.json')['sha256']
  or replay['status']!='PASSED' or replay['actual']!=replay['expected'] or v['source_world_size']!=1
  or v['hardware_topology_changed'] is not False or len(v['source_all_rank_states'])!=4):raise ValueError('First append checkpoint/replay lineage differs')
 actual=dict(raw=u.model_mapping_hash(v['raw']),ema=u.model_mapping_hash(v['ema']),adam=u.optimizer_blob_hash(v['optimizer']),raw_buffers=u.tensor_hash(v['raw_buffers']),ema_buffers=u.tensor_hash(v['ema_buffers']),rng=u.rng_hash(v['rng']))
 if actual!=v['accepted_hashes'] or actual!=replay['actual']:raise ValueError('Independent CP257 six-component reload differs')
 result=dict(status='APPEND_CP257_INDEPENDENT_FULL_SHA_AND_SIX_COMPONENT_CPU_RELOAD_PASSED',checkpoint=proof(p),cursor=257,steps=10721,component_hashes=actual,source_world_size=1,current_world_size=1,original_E1_archive_world_size=4,CUDA_used=False,extra_optimizer_updates=0,time=time.time())
 immutable(TARGET/'first-cp-independent-audit.json',result);return result

def run(expected_sha):
 import torch
 from . import probe_ablation_train as u,probe_g10_train as t
 from .replay_feature_reader import ReplayFeatureReader
 from .model import Probe
 c=read(TARGET/'config.json');s=accepted_selection();pub=read(TARGET/'source-publication.json');root=Path(pub['code_root'])
 if proof(TARGET/'config.json')['sha256']!=expected_sha or configured(read(checked(s['source_config'])),pub,s)!=c:raise ValueError('Immutable append-only config changed')
 for n,v in pub['files'].items():checked(dict(path=str(root/n),**v))
 if (int(os.environ.get('WORLD_SIZE','1'))!=1 or torch.cuda.device_count()!=1
  or os.environ.get('FLASH_ATTENTION_DETERMINISTIC')!='1' or os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8'):raise ValueError('Deterministic single-current-rank GPU allocation required')
 os.environ['QWEN7B_QUESTION_MAX_TOKENS']='273';os.environ['SEARCH_G1_B_PREFIX_TOKENS']='20'
 device=torch.device('cuda',0);torch.cuda.set_device(device);props=torch.cuda.get_device_properties(device)
 if 'H100' not in props.name or props.total_memory<75*1024**3:raise ValueError('One dedicated >=75GiB H100 required')
 torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
 if Path(c['model_dir'])/'config.json'!=checked(c['model_config']):raise ValueError('Official model config changed')
 groups,refs,plan,bank=accepted_inputs(c);evaluations=validate_evaluations(c,groups);lookup=t.validate_plan(plan,groups,c)
 source=torch.load(checked(s['source_checkpoint']),map_location='cpu',mmap=True,weights_only=False)
 planlib.validate_source_checkpoint_metadata(source,s['source_checkpoint'],source_config_sha=s['source_config']['sha256'],source_plan_sha=s['source_plan']['sha256'])
 archive=copy.deepcopy(source['source_all_rank_states']);archive_hashes=[u.rng_hash(r['rng']) for r in archive]
 binding=dict(config_sha256=expected_sha,matched_binding_sha256=u.digest(c),matched_binding=c,source_checkpoint=c['source_checkpoint'],source_selection=SELECTION,g10_complete=c['g10_complete'],recipe=c['recipe'],plan_sha256=c['plan']['sha256'],source_cursor=256,additional_update_budget=682,historical_prefix_retrained=False)
 model=Probe(c['model_dir'],273,20).to(device);ema=copy.deepcopy(model).eval().requires_grad_(False);opt=torch.optim.Adam(model.parameters(),lr=5e-6,weight_decay=0)
 reader=ReplayFeatureReader(refs,26);needed={tid for q in plan['cohort_qids'][8192:] for tid in lookup[q]['trajectory_ids']}
 if set(reader.records)!=needed:raise ValueError('Additional21824 ownG10 feature coverage differs')
 atomic(TARGET/'controller.json',dict(pid=os.getpid(),job_id=os.environ.get('SLURM_JOB_ID'),status='APPEND_INPUTS_ACCEPTED_FULL_CP256_RESTORE_NEXT',source_steps=10720,global_original_questions=32,source_cursor=256,source_world_size=1,current_world_size=1,original_E1_archive_world_size=4,device=props.name))
 for sig in (signal.SIGTERM,signal.SIGUSR1,signal.SIGINT):signal.signal(sig,request_stop)
 env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
 log=(TARGET/'first-cp-independent-audit.log').open('ab',0)
 child=subprocess.Popen([sys.executable,'-u','-m','reproduction.operations.qwen7b_warmup.peer_append_train','--action','audit-first'],stdout=log,stderr=subprocess.STDOUT,env=env)
 atomic(TARGET/'first-cp-audit-controller.json',dict(pid=child.pid,job_id=os.environ.get('SLURM_JOB_ID'),inside_allocated_job=True,time=time.time()))
 cursor=256
 def accept(metrics):
  row=dict(cursor=cursor,additional_completed=cursor-256,steps=u.steps(opt),time=time.time(),**metrics)
  atomic(TARGET/'progress.json',row)
  with (TARGET/'curve.jsonl').open('a') as f:f.write(json.dumps(row)+'\n');f.flush()
  print(json.dumps(row),flush=True)
 try:
  if (TARGET/'latest.json').exists():
   saved=torch.load(u.checked(read(TARGET/'latest.json')),map_location='cpu',mmap=True,weights_only=False);cursor=saved['cursor']
   highest=max([read(TARGET/n)[k] for n,k in [('progress.json','cursor'),('update-intent.json','next_cursor')] if (TARGET/n).exists()] or [cursor])
   planlib.validate_cursor(cursor,saved['steps'],new_config_sha=expected_sha,saved_config_sha=saved['config_sha256'],highest_intent=highest)
   if saved['source_world_size']!=1 or saved['hardware_topology_changed'] is not False:raise ValueError('Resume topology metadata changed')
   u.restore_blob(saved,model,ema,opt,device)
   if u.state_hashes(model,ema,opt,device)!=saved['accepted_hashes']:raise ValueError('Full append state restore differs')
   archive=saved['source_all_rank_states'];del saved
  else:
   if any((TARGET/n).exists() for n in ('progress.json','update-intent.json')):raise ValueError('Unresolved later update without accepted CP; no rollback')
   u.restore_blob(source,model,ema,opt,device)
   if u.state_hashes(model,ema,opt,device)!=source['accepted_hashes']:raise ValueError('Actual CP256 six-component restore differs')
   atomic(TARGET/'update-intent.json',dict(next_cursor=257,state='CP256_FIRST_ADDITIONAL_UPDATE_DOUBLE_REPLAY'))
   train_update(model,ema,opt,reader,lookup,plan['batches'][256],c,device,plan['batch_epochs'][256]);expected=u.state_hashes(model,ema,opt,device)
   u.restore_blob(source,model,ema,opt,device)
   metrics=train_update(model,ema,opt,reader,lookup,plan['batches'][256],c,device,plan['batch_epochs'][256]);actual=u.state_hashes(model,ema,opt,device)
   atomic(TARGET/'next-update-replay.json',dict(status='PASSED' if actual==expected else 'FAILED',expected=expected,actual=actual,source_cursor=256,candidate_cursor=257,source_steps=10720,candidate_steps=10721,source_world_size=1,current_world_size=1,hardware_topology_changed=False))
   if actual!=expected:raise ValueError('Actual append next-update double replay differs')
   cursor=257;u.save_checkpoint(TARGET,full_blob(model,ema,opt,device,cursor,binding,archive));accept(metrics)
  if [u.rng_hash(r['rng']) for r in archive]!=archive_hashes:raise ValueError('Historical original four-rank RNG archive changed')
  del source
  for milestone in c['evaluation_cumulative_cursors']:
   if milestone<cursor and not (TARGET/'evaluations'/('cursor-%06d'%milestone)/'complete.json').exists():raise ValueError('A required earlier evaluation receipt is absent; do not silently skip')
   if milestone<=cursor:evaluate_at(milestone,ema,model,opt,device,evaluations,binding)
  while cursor<938 and not STOP:
   atomic(TARGET/'update-intent.json',dict(next_cursor=cursor+1,state='APPEND_ONLY_OPTIMIZER_TRANSACTION_IN_PROGRESS'))
   metrics=train_update(model,ema,opt,reader,lookup,plan['batches'][cursor],c,device,plan['batch_epochs'][cursor]);cursor+=1;accept(metrics)
   if cursor%c['save_every']==0 or STOP or cursor in c['evaluation_cumulative_cursors']:u.save_checkpoint(TARGET,full_blob(model,ema,opt,device,cursor,binding,archive))
   if cursor in c['evaluation_cumulative_cursors']:evaluate_at(cursor,ema,model,opt,device,evaluations,binding)
  if STOP:
   if read(TARGET/'latest.json')['cursor']!=cursor:u.save_checkpoint(TARGET,full_blob(model,ema,opt,device,cursor,binding,archive))
   atomic(TARGET/'interrupted.json',dict(status='EXACT_ACCEPTED_CUMULATIVE_CURSOR_SAVED',cursor=cursor,additional_completed=cursor-256));return
  if read(TARGET/'latest.json')['cursor']!=938 or u.steps(opt)!=11402:raise ValueError('Final CP938/Adam11402 must be durable before evaluation')
  if child.wait(timeout=60)!=0 or not (TARGET/'first-cp-independent-audit.json').exists():raise ValueError('Independent first full CP257 acceptance failed')
  metrics=evaluate_at(938,ema,model,opt,device,evaluations,binding)
  atomic(TARGET/'complete.json',dict(status='APPEND_ONLY682_NEW_UPDATES_CUMULATIVE938_AND_FIXED_HF_VALIDATION_COMPLETE',checkpoint=read(TARGET/'latest.json'),cursor=938,steps=11402,additional_updates=682,metrics=metrics,evaluation_receipts={str(i):proof(TARGET/'evaluations'/('cursor-%06d'%i)/'complete.json') for i in c['evaluation_cumulative_cursors']},source_world_size=1,current_world_size=1,hardware_topology_changed=False,original_E1_archive_world_size=4,actor_updates=0,optimizer_reset=False,**binding))
 finally:
  reader.close()
  if child.poll() is None:child.terminate();child.wait(timeout=30)

def request_stop(_sig,_frame):
 global STOP
 STOP=True

def main():
 p=argparse.ArgumentParser();p.add_argument('--action',choices=('prepare','train','audit-first'),required=True);p.add_argument('--config-sha256');a=p.parse_args()
 if platform.system()!='Linux':p.error('Actual actions are remote only')
 TARGET.mkdir(parents=True,exist_ok=True);(TARGET/'logs').mkdir(exist_ok=True)
 if a.action=='train' and (not os.environ.get('SLURM_JOB_ID') or not os.environ.get('CUDA_VISIBLE_DEVICES')):p.error('Actual training requires an owned GPU allocation')
 with (TARGET/('first-cp-audit.lock' if a.action=='audit-first' else 'append-only.lock')).open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:result=prepare() if a.action=='prepare' else audit_first() if a.action=='audit-first' else run(a.config_sha256);print(json.dumps(result),flush=True)
  except BaseException as e:atomic(TARGET/('first-cp-audit-failure.json' if a.action=='audit-first' else 'failure.json'),dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc(),time=time.time(),job_id=os.environ.get('SLURM_JOB_ID')));raise
if __name__=='__main__':main()
