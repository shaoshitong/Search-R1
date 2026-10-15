#!/usr/bin/env python3
"""Audit one completed immutable prefix20 checkpoint on h100-8; never start training.

Example (run only after the real savepoint exists):
  python3 work/read_prefix20_immutable_checkpoint_cpu_once.py --epoch 20

--dry-run validates the local state/CLI and compiles the remote CPU payload without SSH.
No checkpoint, trajectory, feature, or complete remote log is downloaded.
"""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import sys

CPU_PYTHON = '/dataset/shaoshitong/Search-G1-common8/env/bin/python'
ROOT = Path(__file__).resolve().parent.parent

REMOTE_BODY = r'''
import datetime,fcntl,hashlib,json,os,pathlib,random,stat,time
p=PAYLOAD;b=pathlib.Path(p['base']);epoch=p['epoch'];expected_adam=epoch*p['adam_per_epoch'];started=time.monotonic()
def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def signature(s):return(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
def statinfo(s):return dict(dev=s.st_dev,inode=s.st_ino,bytes=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)
def file_sha(q):
 h=hashlib.sha256()
 with q.open('rb') as f:
  before=os.fstat(f.fileno());assert stat.S_ISREG(before.st_mode)
  for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
  after=os.fstat(f.fileno())
 assert signature(before)==signature(after)==signature(q.stat()),str(q)+' changed while read'
 return h.hexdigest(),statinfo(before)
def metadata(q):
 with q.open('rb') as f:
  before=os.fstat(f.fileno());data=f.read();after=os.fstat(f.fileno())
 assert signature(before)==signature(after)==signature(q.stat()),str(q)+' changed while read'
 return json.loads(data),dict(path=str(q),sha256=hashlib.sha256(data).hexdigest(),stat=statinfo(before),stat_stable=True)
def once(q,v):
 with q.open('x') as f:json.dump(v,f,sort_keys=True,allow_nan=False);f.flush();os.fsync(f.fileno())
 fd=os.open(q.parent,os.O_RDONLY)
 try:os.fsync(fd)
 finally:os.close(fd)
def output(v):print(json.dumps(v,sort_keys=True,allow_nan=False),flush=True)
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
cp=b/'train'/('head-epoch-%04d.pt'%epoch)
bundle_path=b/'train'/('head-epoch-%04d-bundle.json'%epoch)
complete_path=b/'train/epochs'/('epoch-%04d'%epoch)/'complete.json'
proof_path=b/'preparation'/('checkpoint%04d-independent-full-sha-cpu-once-v1.json'%epoch)
# CP10's already completed audit predates the generic zero-padded name.
proof_candidates=[proof_path]
if epoch==10:proof_candidates.append(b/'preparation/checkpoint10-independent-full-sha-cpu-once-v1.json')
missing=[str(q)for q in(cp,bundle_path,complete_path)if not q.is_file()]
if missing:
 output(dict(status='PENDING_IMMUTABLE_SAVEPOINT',epoch=epoch,run_id=p['run_id'],base=str(b),missing=missing,once_proof_created=False,independent_full_file_sha_performed=False,cpu_load_performed=False,large_payload_transferred_to_Mac=False));raise SystemExit
cfg,cfg_info=metadata(b/'preparation/launch-config.json')
assert cfg['run_id']==p['run_id'] and cfg['base']==str(b),'State/config run identity differs'
assert cfg['source_commit']==p['runtime_commit'],'State/runtime commit differs'
assert cfg['global_batch']==p['global_batch'] and cfg['learning_rate']==p['learning_rate']
assert cfg['epochs']==p['total_epochs'] and cfg['save_every']==p['save_every']
assert epoch<=cfg['epochs'] and(epoch%cfg['save_every']==0 or epoch==cfg['epochs'])
bundle,bundle_info=metadata(bundle_path);complete,complete_info=metadata(complete_path)
if complete.get('status')!='FRESH_ONLINE20_EPOCH_COMPLETE' or complete.get('immutable_saved')is not True:
 output(dict(status='PENDING_IMMUTABLE_COMPLETE_RECEIPT',epoch=epoch,complete_status=complete.get('status'),immutable_saved=complete.get('immutable_saved'),once_proof_created=False,independent_full_file_sha_performed=False,cpu_load_performed=False));raise SystemExit
assert bundle['run_id']==p['run_id'] and bundle['epoch']==complete['epoch']==epoch
assert bundle['head_path']==str(cp) and bundle['config_sha256']==cfg_info['sha256']
producer_sha=complete['sha256'];producer_bytes=complete['bytes']
assert bundle['head_sha256']==producer_sha and len(producer_sha)==64
assert complete['actual_adam_steps_per_rank']==bundle['actual_adam_steps_per_rank']==[expected_adam]*8
assert complete['actual_learning_rate']==cfg['learning_rate'] and complete['actual_optimizer_learning_rates']==[cfg['learning_rate']]
assert complete['train_trajectories']==p['epoch_trajectories'] and complete['test_trajectories']==p['diagnostic_trajectories'] and complete['actor_updates']==0
# A nonblocking task-local lock prevents two callers from doing the expensive read concurrently.
lock_path=b/'preparation'/('checkpoint%04d-independent-cpu-audit.lock'%epoch)
lock=lock_path.open('a')
try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError:
 output(dict(status='CPU_AUDIT_ALREADY_RUNNING',epoch=epoch,run_id=p['run_id'],lock_path=str(lock_path),once_proof_created=False,independent_full_file_sha_performed=False,cpu_load_performed=False));raise SystemExit
existing=[]
for q in proof_candidates:
 if not q.exists():continue
 prior,info=metadata(q)
 assert prior.get('status')in('IMMUTABLE_CP_INDEPENDENT_FULL_SHA_CPU_STATE_PASS','IMMUTABLE_CP10_INDEPENDENT_FULL_SHA_CPU_STATE_PASS'),'Existing once proof is not a successful CPU audit'
 assert prior.get('run_id')==p['run_id'] and prior.get('runtime_commit')==p['runtime_commit'],'Existing once proof identity differs'
 assert prior.get('epoch')==epoch and prior.get('checkpoint_path')==str(cp),'Existing once proof checkpoint differs'
 assert prior.get('checkpoint_sha256')==producer_sha and prior.get('checkpoint_bytes')==producer_bytes,'Existing once proof bytes differ'
 assert prior.get('actual_adam_steps_per_rank')==[expected_adam]*8
 assert prior.get('actual_optimizer_learning_rates')==[cfg['learning_rate']]
 assert prior.get('independent_full_file_sha_performed')is True and prior.get('cpu_load_performed')is True
 assert prior.get('binding_sha256')==complete['binding_sha256']
 assert prior.get('bundle_actual_file',{}).get('sha256')==bundle_info['sha256'],'Existing proof bundle binding differs'
 existing.append((prior,info))
if existing:
 prior,info=existing[0]
 output(dict(status='EXISTING_ONCE_PROOF_INSPECTED_NO_REPLAY',epoch=epoch,run_id=p['run_id'],proof_path=info['path'],proof_sha256=info['sha256'],proof_status=prior['status'],checkpoint_path=str(cp),checkpoint_sha256=prior['checkpoint_sha256'],checkpoint_bytes=prior['checkpoint_bytes'],actual_adam_steps_per_rank=prior['actual_adam_steps_per_rank'],actual_optimizer_learning_rates=prior['actual_optimizer_learning_rates'],existing_proof_candidates=len(existing),independent_full_file_sha_performed_this_call=False,cpu_load_performed_this_call=False,prior_independent_full_file_sha_performed=True,prior_cpu_load_performed=True,once_proof_created=False,large_payload_transferred_to_Mac=False));raise SystemExit
with cp.open('rb') as f:
 before=os.fstat(f.fileno());assert stat.S_ISREG(before.st_mode) and before.st_size==producer_bytes
 h=hashlib.sha256()
 for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
 cp_sha=h.hexdigest();assert cp_sha==producer_sha,'Independent full checkpoint SHA differs from producer receipt'
 f.seek(0)
 import torch
 import numpy as np
 saved=torch.load(f,map_location='cpu',weights_only=False)
 after=os.fstat(f.fileno())
assert signature(before)==signature(after)==signature(cp.stat()),'Immutable checkpoint stat changed'
assert saved['schema_version']==1 and saved['epoch']==epoch and saved['actual_adam_steps_per_rank']==[expected_adam]*8
binding=saved['binding'];binding_sha=hashlib.sha256(canonical(binding)).hexdigest()
assert saved['binding_sha256']==binding_sha==complete['binding_sha256']
assert binding['role']=='online20_fresh_trajectory_correctness_head'
assert binding['feature_layer']==27 and binding['feature_boundary']=='post_block' and binding['prefix_tokens']==20
assert binding['encoder_frozen']is True and binding['question_hidden_passed_to_head']is False
assert binding['global_batch']==cfg['global_batch'] and binding['world_size']==8 and binding['accumulation']==1
assert binding['learning_rate']==cfg['learning_rate'] and binding['seed']==cfg['seed'] and binding['epochs']==cfg['epochs'] and binding['save_every']==cfg['save_every']
assert binding['test_manifest_sha256']==saved['test_manifest_sha256']==cfg['cached_features_manifest_sha256']
assert saved['actual_optimizer_learning_rates']==[cfg['learning_rate']]
transition=saved.get('learning_rate_transition');assert transition==complete.get('learning_rate_transition')
if transition is not None:
 assert transition['run_id']==p['run_id'] and transition['authorized_change_only']=='learning_rate'
 assert transition['start_epoch']==transition['source_complete_epoch']+1 and epoch>=transition['start_epoch']
 assert transition['target_learning_rate']==cfg['learning_rate'] and transition['target_binding_sha256']==binding_sha
shapes={
 'block.self_attn.q_proj.weight':(2048,2048),'block.self_attn.q_proj.bias':(2048,),
 'block.self_attn.k_proj.weight':(256,2048),'block.self_attn.k_proj.bias':(256,),
 'block.self_attn.v_proj.weight':(256,2048),'block.self_attn.v_proj.bias':(256,),
 'block.self_attn.o_proj.weight':(2048,2048),
 'block.mlp.gate_proj.weight':(11008,2048),'block.mlp.up_proj.weight':(11008,2048),
 'block.mlp.down_proj.weight':(2048,11008),
 'block.input_layernorm.weight':(2048,),'block.post_attention_layernorm.weight':(2048,),
 'output.weight':(1,40960),'output.bias':(1,)}
model=saved['model'];assert set(model)==set(shapes)
for key,value in model.items():
 assert isinstance(value,torch.Tensor) and tuple(value.shape)==shapes[key]
 assert value.dtype==torch.float32 and value.device.type=='cpu'
optimizer=saved['optimizer'];groups=optimizer['param_groups'];states=optimizer['state']
assert len(groups)==1 and groups[0]['lr']==cfg['learning_rate']
ids=[key for group in groups for key in group['params']]
assert len(ids)==len(set(ids))==len(states)==len(model)==14 and set(ids)==set(states)
steps=[];moment_dtypes=set();parameter_state_summary=[]
for key,(name,value) in zip(ids,model.items()):
 state=states[key];assert set(state)=={'step','exp_avg','exp_avg_sq'}
 step=state['step'];step_value=int(step.item())if isinstance(step,torch.Tensor)else int(step)
 assert step_value==expected_adam;steps.append(step_value)
 for field in('exp_avg','exp_avg_sq'):
  tensor=state[field];assert isinstance(tensor,torch.Tensor) and tensor.shape==value.shape
  assert tensor.dtype==torch.float32 and tensor.device.type=='cpu';moment_dtypes.add(str(tensor.dtype))
 parameter_state_summary.append(dict(parameter=name,shape=list(value.shape),dtype=str(value.dtype),adam_step=step_value,moment_fields=['exp_avg','exp_avg_sq']))
rng=saved['rank_rng'];assert len(rng)==8 and[r['rank']for r in rng]==list(range(8))
rng_summary=[]
for row in rng:
 assert row['actual_adam_steps']==expected_adam and all(key in row for key in('python','numpy','torch_cpu','torch_cuda'))
 random.Random().setstate(row['python']);np.random.RandomState().set_state(row['numpy'])
 assert isinstance(row['torch_cpu'],torch.Tensor)and isinstance(row['torch_cuda'],torch.Tensor)
 assert row['torch_cpu'].dtype==row['torch_cuda'].dtype==torch.uint8
 assert row['torch_cpu'].device.type==row['torch_cuda'].device.type=='cpu'
 assert row['torch_cpu'].numel()>0 and row['torch_cuda'].numel()>0
 torch.Generator(device='cpu').set_state(row['torch_cpu'])
 rng_summary.append(dict(rank=row['rank'],actual_adam_steps=row['actual_adam_steps'],python_state_valid=True,numpy_state_valid=True,torch_cpu_state_valid=True,torch_cuda_state_present=True,torch_cpu_state_bytes=row['torch_cpu'].numel(),torch_cuda_state_bytes=row['torch_cuda'].numel()))
root=b/'epochs'/('epoch-%04d'%epoch)
pool_path=pathlib.Path(bundle['pool_state_path']);assert pool_path==root/'pool-state-at-head-completion.json'
pool,pool_info=metadata(pool_path);assert pool_info['sha256']==bundle['pool_state_sha256']
size=p['pool_size'];order=pool['order'];cursor=pool['cursor'];cycle=pool['cycle']
assert pool['seed']==cfg['seed'] and pool['next_epoch']==epoch+1 and pool['epoch_questions']==p['epoch_questions']
assert pool['consumed_questions']==p['epoch_questions']*epoch
assert type(cycle)is int and cycle>=0 and type(cursor)is int and 0<=cursor<=size
assert len(order)==len(set(order))==size and set(order)==set(range(size))
assert cycle*size+cursor==pool['consumed_questions'],'Pool cycle/cursor differ from total consumption'
pool_sha,pool_stat=file_sha(b/'question-pool/pool.json');assert pool_sha==bundle['pool_json_sha256']
selection,selection_info=metadata(root/'selection.json')
assert selection_info['sha256']==bundle['selection_sha256'] and selection['epoch']==epoch
assert selection['questions']==p['epoch_questions'] and selection['trajectories']==p['epoch_trajectories'] and selection['trajectory_group_size']==5
rows=selection['rows'];assert len(rows)==p['epoch_questions']
assert len({row['qid']for row in rows})==len(rows)and len({row['pool_index']for row in rows})==len(rows)
assert selection['question_order_sha256']==hashlib.sha256(canonical(rows)).hexdigest()
assert selection['pool_cycle_start']==rows[0]['pool_cycle'] and selection['pool_cycle_end']==rows[-1]['pool_cycle']==cycle
# At a reshuffle boundary, only the tail belongs to the newly current order.
current_rows=[row for row in rows if row['pool_cycle']==cycle]
assert rows[-len(current_rows):]==current_rows and all(row['pool_cycle']in(cycle-1,cycle)for row in rows)
assert[row['pool_index']for row in current_rows]==order[cursor-len(current_rows):cursor]
assert all(0<=row['pool_index']<size for row in rows)
rollout_receipt_sha,rollout_receipt_stat=file_sha(root/'complete.json');assert rollout_receipt_sha==bundle['rollout_complete_sha256']
assert signature(before)==signature(cp.stat()),'Immutable checkpoint changed after metadata audit'
proof=dict(schema_version=1,status='IMMUTABLE_CP_INDEPENDENT_FULL_SHA_CPU_STATE_PASS',observed_utc=utc(),run_id=p['run_id'],runtime_commit=p['runtime_commit'],checkpoint_path=str(cp),checkpoint_sha256=cp_sha,checkpoint_bytes=before.st_size,checkpoint_stat_before=statinfo(before),checkpoint_stat_after=statinfo(after),checkpoint_stat_stable=True,independent_full_file_sha_performed=True,producer_sha_matched_independent_read=True,cpu_load_performed=True,cpu_load_map_location='cpu',gpu_forward_performed=False,gpu_semantic_reload_performed=False,training_environment_modified=False,audit_cpu_subprocess_cuda_visible_devices='',epoch=epoch,actual_adam_steps_per_rank=[expected_adam]*8,actual_optimizer_learning_rates=[g['lr']for g in groups],optimizer_parameter_states=len(states),optimizer_step_values=sorted(set(steps)),model_parameter_tensors=len(model),model_parameter_count=sum(v.numel()for v in model.values()),model_parameter_dtypes=['torch.float32'],optimizer_moment_dtypes=sorted(moment_dtypes),parameter_state_summary=parameter_state_summary,rank_rng=rng_summary,binding=binding,binding_sha256=binding_sha,learning_rate_transition=transition,train_manifest_sha256=saved['train_manifest_sha256'],test_manifest_sha256=saved['test_manifest_sha256'],checkpoint_bundle=bundle,bundle_actual_file=bundle_info,epoch_complete_actual_file=complete_info,launch_config_actual_file=cfg_info,pool_state_actual_file=pool_info,pool_next_epoch=pool['next_epoch'],pool_cursor=cursor,pool_cycle=cycle,pool_consumed_questions=pool['consumed_questions'],pool_order_unique_size=len(order),immutable_pool_json_actual_file=dict(path=str(b/'question-pool/pool.json'),sha256=pool_sha,stat=pool_stat,stat_stable=True),selection_actual_file=selection_info,selection_question_order_sha256=selection['question_order_sha256'],selection_current_cycle_order_matches=True,rollout_completion_receipt_actual_file=dict(path=str(root/'complete.json'),sha256=rollout_receipt_sha,stat=rollout_receipt_stat,stat_stable=True),rollout_payload_read=False,official_model_files_read=False,mutable_latest_checkpoint_read=False,mutable_pool_state_read=False,backup_copy_performed=False,large_payload_transferred_to_Mac=False,control_state_path=p['state_path'],control_state_sha256=p['state_sha256'],elapsed_seconds=time.monotonic()-started)
once(proof_path,proof);proof_sha,proof_stat=file_sha(proof_path)
output(dict(status=proof['status'],observed_utc=proof['observed_utc'],proof_path=str(proof_path),proof_sha256=proof_sha,proof_bytes=proof_stat['bytes'],checkpoint_path=str(cp),checkpoint_sha256=cp_sha,checkpoint_bytes=before.st_size,checkpoint_stat_stable=True,independent_full_file_sha_performed=True,cpu_load_performed=True,epoch=epoch,actual_adam_steps_per_rank=[expected_adam]*8,actual_optimizer_learning_rates=[g['lr']for g in groups],optimizer_parameter_states=len(states),optimizer_step_values=sorted(set(steps)),model_parameter_tensors=len(model),model_parameter_dtypes=['torch.float32'],optimizer_moment_dtypes=sorted(moment_dtypes),rank_rng_complete_count=len(rng),pool_bundle_actual_hashes_matched=True,pool_next_epoch=pool['next_epoch'],pool_cursor=cursor,pool_cycle=cycle,pool_consumed_questions=pool['consumed_questions'],bundle_path=bundle_info['path'],bundle_sha256=bundle_info['sha256'],once_proof_created=True,gpu_forward_performed=False,backup_copy_performed=False,large_payload_transferred_to_Mac=False,elapsed_seconds=proof['elapsed_seconds']))
'''


def positive_int(value):
    result = int(value)
    if result < 1:
        raise argparse.ArgumentTypeError('must be a positive integer')
    return result


def prepare(args):
    path = args.state.resolve()
    raw = path.read_bytes()
    state = json.loads(raw)
    if state['role'] != 'fresh_rollout_prefix20_correctness_probe1000':
        raise ValueError('State is not the current prefix20 online probe goal')
    if state['node'] != 'h100-8':
        raise ValueError('Only the assigned h100-8 training node is allowed')
    runtime = state.get('runtime_source_commit', state['source_commit'])
    if len(runtime) != 40 or any(c not in '0123456789abcdef' for c in runtime):
        raise ValueError('State must pin a complete runtime commit')
    epoch = args.epoch
    total, cadence = state['total_epochs'], state['save_every']
    if epoch > total or (epoch % cadence != 0 and epoch != total):
        raise ValueError('Requested epoch is not an immutable cadence/final savepoint')
    if (state['global_batch'], state['world_size'], state['accumulation'],
            state['epoch_train_questions'], state['epoch_train_trajectories'],
            state['planned_adam_per_epoch']) != (256, 8, 1, 5120, 25600, 100):
        raise ValueError('Current state no longer matches this probe checkpoint format')
    base = state['base']
    if not Path(base).is_absolute() or not base.startswith('/mnt/data/shaoshitong/'):
        raise ValueError('Checkpoint base must remain in the actual remote task storage')
    payload = dict(base=base, run_id=state['run_id'], runtime_commit=runtime,
                   epoch=epoch, total_epochs=total, save_every=cadence,
                   global_batch=state['global_batch'], world_size=state['world_size'],
                   learning_rate=state['learning_rate'], adam_per_epoch=100,
                   epoch_questions=5120, epoch_trajectories=25600,
                   diagnostic_trajectories=state['fixed_diagnostic_trajectories'],
                   pool_size=168510, state_path=str(path),
                   state_sha256=hashlib.sha256(raw).hexdigest())
    body = REMOTE_BODY.replace('PAYLOAD', repr(payload))
    compile(body, '<immutable checkpoint CPU audit>', 'exec')
    wrapper = ('import os; body=' + repr(body) +
               "; env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1'); " +
               'os.execve(' + repr(CPU_PYTHON) + ',[' + repr(CPU_PYTHON) + ",'-c',body],env)")
    compile(wrapper, '<CPU audit environment wrapper>', 'exec')
    return state, payload, wrapper


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epoch', required=True, type=positive_int)
    parser.add_argument('--state', type=Path, default=ROOT/'work/prefix20-online1000-state.json')
    parser.add_argument('--output-dir', type=Path, default=ROOT/'work',
                        help='local code/small transport and audit summaries only')
    parser.add_argument('--dry-run', action='store_true', help='local validation/compile only; no SSH')
    args = parser.parse_args(argv)
    state, payload, wrapper = prepare(args)
    if args.dry_run:
        print(json.dumps(dict(status='LOCAL_DRY_RUN_COMPILED_NO_SSH', payload=payload,
                              remote_body_compiled=True, cpu_wrapper_compiled=True,
                              remote_execution_performed=False), sort_keys=True))
        return 0
    # Import transport only after CLI validation, and only from the state's current code checkout.
    sys.path.insert(0, state['worktree'])
    from reproduction.h1008_qprobe.management_transport import remote
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    stem = 'prefix20-cp%04d-independent-cpu-' % args.epoch
    transport = output_dir/(stem+'transport-'+stamp+'.json')
    output = remote(wrapper, transport, control_path=None, target='h100-8')
    summary = json.loads(output)
    summary.update(transport_receipt=str(transport),
                   transport_receipt_sha256=hashlib.sha256(transport.read_bytes()).hexdigest(),
                   local_adapter_path=str(Path(__file__).resolve()),
                   local_adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                   state_path=payload['state_path'], state_sha256=payload['state_sha256'])
    path = output_dir/(stem+'summary-'+stamp+'.json')
    with path.open('x') as stream:
        json.dump(summary, stream, indent=2, sort_keys=True, allow_nan=False)
    print(json.dumps(dict(local_summary=str(path),
                          local_summary_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                          summary=summary), sort_keys=True), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
