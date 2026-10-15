"""One immutable CP10 full-file independent SHA and CPU state audit; no GPU use."""
import datetime
import hashlib
import json
from pathlib import Path
import sys

CWD = Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control')
REPO = Path('/Users/shaoshitong/Documents/Search-G1/qprobe-prefix20-online1000-h1008-20261016')
payload = dict(
    base='/mnt/data/shaoshitong/search-g1-prefix20-online1000-20261016-4bc6838-lr3e5',
    run_id='8a6b24dc19664cb3a3fb97101f21c54e',
    runtime_commit='4bc683883a569d7feb9d5da87307612b872e8f3a',
    producer_checkpoint_sha256='9cadb713c580874e4c7bb2d19a05bb3d1f0148e0e6ece2a988cbbf1e2c4db43a',
    expected_bytes=925542230,
    producer_bundle_sha256='530da792cbe8d249e53fc085ae1ef8e0e85a41543091d7889c2ba4f50920b079',
    expected_test_manifest_sha256='86c100961ae9bb75a11da3d39f1026c36e88dfd29ae68dc34a0786c5d185cd1b')
body = r'''
import datetime,hashlib,json,os,pathlib,random,stat,time
p=PAYLOAD;b=pathlib.Path(p['base']);started=time.monotonic()
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
proof_path=b/'preparation/checkpoint10-independent-full-sha-cpu-once-v1.json'
if proof_path.exists():
 prior,info=metadata(proof_path)
 print(json.dumps(dict(status='EXISTING_ONCE_PROOF_INSPECTED_NO_REPLAY',proof_path=str(proof_path),proof_sha256=info['sha256'],proof_status=prior.get('status'),checkpoint_sha256=prior.get('checkpoint_sha256'),epoch=prior.get('epoch'),actual_adam_steps_per_rank=prior.get('actual_adam_steps_per_rank'),cpu_load_performed=prior.get('cpu_load_performed'),large_payload_transferred_to_Mac=False)));raise SystemExit
assert os.environ.get('CUDA_VISIBLE_DEVICES')==''
cfg,cfg_info=metadata(b/'preparation/launch-config.json')
assert cfg['run_id']==p['run_id'] and cfg['base']==str(b)
assert cfg['source_commit']==p['runtime_commit']
cp=b/'train/head-epoch-0010.pt'
bundle,bundle_info=metadata(b/'train/head-epoch-0010-bundle.json')
complete,complete_info=metadata(b/'train/epochs/epoch-0010/complete.json')
assert bundle_info['sha256']==p['producer_bundle_sha256']
assert bundle['run_id']==p['run_id'] and bundle['epoch']==10 and bundle['head_path']==str(cp)
assert bundle['config_sha256']==cfg_info['sha256']
assert complete['status']=='FRESH_ONLINE20_EPOCH_COMPLETE' and complete['epoch']==10 and complete['immutable_saved'] is True
assert complete['actual_adam_steps_per_rank']==bundle['actual_adam_steps_per_rank']==[1000]*8
assert complete['actual_learning_rate']==3e-5 and complete['actual_optimizer_learning_rates']==[3e-5]
assert complete['train_trajectories']==25600 and complete['test_trajectories']==2560 and complete['actor_updates']==0
with cp.open('rb') as f:
 before=os.fstat(f.fileno());assert stat.S_ISREG(before.st_mode) and before.st_size==p['expected_bytes']
 h=hashlib.sha256()
 for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
 cp_sha=h.hexdigest()
 assert cp_sha==p['producer_checkpoint_sha256']==bundle['head_sha256']==complete['sha256']
 f.seek(0)
 import torch
 import numpy as np
 saved=torch.load(f,map_location='cpu',weights_only=False)
 after=os.fstat(f.fileno())
assert signature(before)==signature(after)==signature(cp.stat()),'Immutable CP10 stat changed'
assert saved['schema_version']==1 and saved['epoch']==10 and saved['actual_adam_steps_per_rank']==[1000]*8
binding=saved['binding'];binding_sha=hashlib.sha256(canonical(binding)).hexdigest()
assert saved['binding_sha256']==binding_sha==complete['binding_sha256']
assert binding['role']=='online20_fresh_trajectory_correctness_head'
assert binding['feature_layer']==27 and binding['feature_boundary']=='post_block' and binding['prefix_tokens']==20
assert binding['encoder_frozen'] is True and binding['question_hidden_passed_to_head'] is False
assert binding['global_batch']==256 and binding['world_size']==8 and binding['accumulation']==1
assert binding['learning_rate']==3e-5 and binding['seed']==1729 and binding['epochs']==1000 and binding['save_every']==10
assert binding['test_manifest_sha256']==saved['test_manifest_sha256']==p['expected_test_manifest_sha256']
assert saved['actual_optimizer_learning_rates']==[3e-5]
transition=saved['learning_rate_transition']
assert transition==complete['learning_rate_transition']
assert transition['run_id']==p['run_id'] and transition['authorized_change_only']=='learning_rate'
assert transition['source_complete_epoch']==7 and transition['start_epoch']==8
assert transition['source_learning_rate']==3e-4 and transition['target_learning_rate']==3e-5
assert transition['target_binding_sha256']==binding_sha
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
assert len(groups)==1 and groups[0]['lr']==3e-5
ids=[key for group in groups for key in group['params']]
assert len(ids)==len(set(ids))==len(states)==len(model)==14 and set(ids)==set(states)
steps=[];moment_dtypes=set();parameter_state_summary=[]
for key,(name,value) in zip(ids,model.items()):
 state=states[key];assert set(state)=={'step','exp_avg','exp_avg_sq'}
 step=state['step'];step_value=int(step.item()) if isinstance(step,torch.Tensor) else int(step)
 assert step_value==1000;steps.append(step_value)
 for field in('exp_avg','exp_avg_sq'):
  tensor=state[field];assert isinstance(tensor,torch.Tensor) and tensor.shape==value.shape
  assert tensor.dtype==torch.float32 and tensor.device.type=='cpu';moment_dtypes.add(str(tensor.dtype))
 parameter_state_summary.append(dict(parameter=name,shape=list(value.shape),dtype=str(value.dtype),adam_step=step_value,moment_fields=['exp_avg','exp_avg_sq']))
rng=saved['rank_rng'];assert len(rng)==8 and[r['rank'] for r in rng]==list(range(8))
rng_summary=[]
for row in rng:
 assert row['actual_adam_steps']==1000 and all(key in row for key in('python','numpy','torch_cpu','torch_cuda'))
 random.Random().setstate(row['python']);np.random.RandomState().set_state(row['numpy'])
 assert isinstance(row['torch_cpu'],torch.Tensor) and isinstance(row['torch_cuda'],torch.Tensor)
 assert row['torch_cpu'].dtype==row['torch_cuda'].dtype==torch.uint8
 assert row['torch_cpu'].device.type==row['torch_cuda'].device.type=='cpu'
 assert row['torch_cpu'].numel()>0 and row['torch_cuda'].numel()>0
 torch.Generator(device='cpu').set_state(row['torch_cpu'])
 rng_summary.append(dict(rank=row['rank'],actual_adam_steps=row['actual_adam_steps'],python_state_valid=True,numpy_state_valid=True,torch_cpu_state_valid=True,torch_cuda_state_present=True,torch_cpu_state_bytes=row['torch_cpu'].numel(),torch_cuda_state_bytes=row['torch_cuda'].numel()))
pool_path=pathlib.Path(bundle['pool_state_path']);assert pool_path==b/'epochs/epoch-0010/pool-state-at-head-completion.json'
pool,pool_info=metadata(pool_path)
assert pool_info['sha256']==bundle['pool_state_sha256']
assert pool['seed']==1729 and pool['next_epoch']==11 and pool['consumed_questions']==51200 and pool['epoch_questions']==5120
assert pool['cycle']==0 and pool['cursor']==51200 and len(pool['order'])==len(set(pool['order']))==168510
assert set(pool['order'])==set(range(168510))
# Only immutable pool metadata is read; never current mutable state or rollout payloads.
pool_sha,pool_stat=file_sha(b/'question-pool/pool.json');assert pool_sha==bundle['pool_json_sha256']
# The small selection/rollout completion receipts are bound by their hashes, without reading any trajectory shard.
root=b/'epochs/epoch-0010'
selection_sha,selection_stat=file_sha(root/'selection.json')
rollout_receipt_sha,rollout_receipt_stat=file_sha(root/'complete.json')
assert selection_sha==bundle['selection_sha256'] and rollout_receipt_sha==bundle['rollout_complete_sha256']
assert signature(before)==signature(cp.stat()),'Immutable CP10 changed after metadata audit'
proof=dict(schema_version=1,status='IMMUTABLE_CP10_INDEPENDENT_FULL_SHA_CPU_STATE_PASS',observed_utc=utc(),run_id=p['run_id'],runtime_commit=p['runtime_commit'],checkpoint_path=str(cp),checkpoint_sha256=cp_sha,checkpoint_bytes=before.st_size,checkpoint_stat_before=statinfo(before),checkpoint_stat_after=statinfo(after),checkpoint_stat_stable=True,independent_full_file_sha_performed=True,producer_sha_matched_independent_read=True,cpu_load_performed=True,cpu_load_map_location='cpu',gpu_forward_performed=False,gpu_semantic_reload_performed=False,training_environment_modified=False,audit_cpu_subprocess_cuda_visible_devices='',epoch=10,actual_adam_steps_per_rank=[1000]*8,actual_optimizer_learning_rates=[g['lr']for g in groups],optimizer_parameter_states=len(states),optimizer_step_values=sorted(set(steps)),model_parameter_tensors=len(model),model_parameter_count=sum(v.numel()for v in model.values()),model_parameter_dtypes=['torch.float32'],optimizer_moment_dtypes=sorted(moment_dtypes),parameter_state_summary=parameter_state_summary,rank_rng=rng_summary,binding=binding,binding_sha256=binding_sha,learning_rate_transition=transition,train_manifest_sha256=saved['train_manifest_sha256'],test_manifest_sha256=saved['test_manifest_sha256'],checkpoint_bundle=bundle,bundle_actual_file=bundle_info,epoch_complete_actual_file=complete_info,launch_config_actual_file=cfg_info,pool_state_actual_file=pool_info,pool_next_epoch=pool['next_epoch'],pool_cursor=pool['cursor'],pool_cycle=pool['cycle'],pool_consumed_questions=pool['consumed_questions'],pool_order_unique_size=len(pool['order']),immutable_pool_json_actual_file=dict(path=str(b/'question-pool/pool.json'),sha256=pool_sha,stat=pool_stat,stat_stable=True),selection_actual_file=dict(path=str(root/'selection.json'),sha256=selection_sha,stat=selection_stat,stat_stable=True),rollout_completion_receipt_actual_file=dict(path=str(root/'complete.json'),sha256=rollout_receipt_sha,stat=rollout_receipt_stat,stat_stable=True),rollout_payload_read=False,official_model_files_read=False,mutable_latest_checkpoint_read=False,mutable_pool_state_read=False,backup_copy_performed=False,large_payload_transferred_to_Mac=False,elapsed_seconds=time.monotonic()-started)
once(proof_path,proof);proof_sha,proof_stat=file_sha(proof_path)
print(json.dumps(dict(status=proof['status'],observed_utc=proof['observed_utc'],proof_path=str(proof_path),proof_sha256=proof_sha,proof_bytes=proof_stat['bytes'],checkpoint_path=str(cp),checkpoint_sha256=cp_sha,checkpoint_bytes=before.st_size,checkpoint_stat_stable=True,independent_full_file_sha_performed=True,cpu_load_performed=True,epoch=10,actual_adam_steps_per_rank=[1000]*8,actual_optimizer_learning_rates=[g['lr']for g in groups],optimizer_parameter_states=len(states),optimizer_step_values=sorted(set(steps)),model_parameter_tensors=len(model),model_parameter_dtypes=['torch.float32'],optimizer_moment_dtypes=sorted(moment_dtypes),rank_rng_complete_count=len(rng),pool_bundle_actual_hashes_matched=True,pool_next_epoch=pool['next_epoch'],pool_cursor=pool['cursor'],pool_consumed_questions=pool['consumed_questions'],bundle_path=bundle_info['path'],bundle_sha256=bundle_info['sha256'],gpu_forward_performed=False,backup_copy_performed=False,large_payload_transferred_to_Mac=False,elapsed_seconds=proof['elapsed_seconds'])),flush=True)
'''.replace('PAYLOAD', repr(payload))
compile(body, '<immutable CP10 CPU proof>', 'exec')
wrapper="import os; body="+repr(body)+"; env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',HF_DATASETS_OFFLINE='1'); os.execve('/dataset/shaoshitong/Search-G1-common8/env/bin/python',['/dataset/shaoshitong/Search-G1-common8/env/bin/python','-c',body],env)"
sys.path.insert(0,str(REPO))
from reproduction.h1008_qprobe.management_transport import remote
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
transport=CWD/'work'/('prefix20-cp10-independent-cpu-transport-'+stamp+'.json')
output=remote(wrapper,transport,control_path=None,target='h100-8')
summary=json.loads(output)
summary.update(transport_receipt=str(transport),transport_receipt_sha256=hashlib.sha256(transport.read_bytes()).hexdigest(),local_adapter_path=str(Path(__file__)),local_adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
path=CWD/'work'/('prefix20-cp10-independent-cpu-summary-'+stamp+'.json')
with path.open('x')as f:json.dump(summary,f,indent=2,sort_keys=True)
print(json.dumps(dict(local_summary=str(path),local_summary_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),summary=summary)),flush=True)
