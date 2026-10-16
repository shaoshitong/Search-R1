"""Once CPU/full-file-SHA audit of an actual immutable joint100 savepoint; never train."""
import argparse,fcntl,gc,hashlib,json,math,os,random,sys,time
from pathlib import Path

BASE='/mnt/data/shaoshitong/search-g1-question20-groupjoint100-20261016-cee20bcd'
RUN='cee20bcd24194b79a9a8a88809bd5886'
SOURCE='2fd5c19fa0d9c5d6c4eb514a7e9a308eb6bc6479'
BINDING='791ab0fd04af07e4655f364462454ea9bcfecf1e8d21101b1a41313cc60eda3b'
TEST='a125fc5af90e34fcd728c82077d9d62070a78d5ef703ac11f3748687d088e64f'

def read(path):return json.loads(Path(path).read_text())
def digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def signature(info):return(info.st_dev,info.st_ino,info.st_size,info.st_mtime_ns,info.st_ctime_ns)
def small_sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def once(path,value):
 with path.open('x')as handle:json.dump(value,handle,sort_keys=True,allow_nan=False);handle.flush();os.fsync(handle.fileno())
def summary(proof,path,existing=False):
 return dict(status='EXISTING_JOINT_SAVEPOINT_PROOF_NO_REPLAY'if existing else proof['status'],epoch=proof['epoch'],
  run_id=RUN,proof_path=str(path),proof_sha256=small_sha(path),checkpoint_sha256=proof.get('checkpoint_sha256'),
  actual_adam_steps_per_rank=proof.get('actual_adam_steps_per_rank'),actual_learning_rate=proof.get('actual_learning_rate'),
  proof_status=proof['status'],audit_error=proof.get('error'),
  independent_full_sha_performed_this_call=not existing and proof.get('independent_full_file_sha_performed',False),
  cpu_load_performed_this_call=not existing and proof.get('cpu_load_performed',False),
  gpu_forward_performed=False,cuda_rng_applied=False,large_payload_transferred_to_Mac=False)

def audit(base,epoch):
 b=Path(base);assert str(b)==BASE and epoch in range(10,101,10)
 assert os.environ.get('CUDA_VISIBLE_DEVICES')=='','CPU-only audit requires empty CUDA_VISIBLE_DEVICES'
 prep=b/'preparation';proof_path=prep/('checkpoint%04d-independent-full-sha-cpu-once-v1.json'%epoch)
 if proof_path.exists():
  prior=read(proof_path);assert prior['run_id']==RUN and prior['epoch']==epoch and prior['binding_sha256']==BINDING
  return summary(prior,proof_path,True)
 cp=b/'train'/('head-epoch-%04d.pt'%epoch)
 root=b/'train/epochs'/('epoch-%04d'%epoch);bundle_path=b/'epochs'/('epoch-%04d'%epoch)/'data-bundle.json'
 needed=[cp,root/'complete.json',root/'metrics.json',bundle_path]
 missing=[str(path)for path in needed if not path.is_file()]
 if missing:return dict(status='PENDING_IMMUTABLE_SAVEPOINT_NO_INTENT',epoch=epoch,missing=missing,once_intent_created=False)
 c=read(prep/'launch-config.json');claim=read(b/'train/run-claim.json');binding=claim['binding']
 complete,metrics,bundle=read(root/'complete.json'),read(root/'metrics.json'),read(bundle_path)
 steps=epoch*100;source_epoch=(list(range(1,71))+list(range(1,31)))[epoch-1]
 assert c['run_id']==RUN and c['source_commit']==SOURCE and c['base']==BASE and c['fresh_head']is True
 assert c['epochs']==100 and c['total_optimizer_updates']==10000 and c['data_schedule']==list(range(1,71))+list(range(1,31))
 assert claim['binding_sha256']==digest(binding)==BINDING==complete['binding_sha256']
 assert binding['run_id']==RUN and binding['test_manifest_sha256']==TEST and binding['source_data_schedule']==c['data_schedule']
 assert binding['question_max_tokens']==273 and binding['head_max_tokens']==293 and binding['feature_layer']==27
 assert binding['old_probe_imported']is False and binding['old_optimizer_imported']is False and binding['fresh_run_initial_adam_steps']==0
 assert binding['global_batch_average']==256 and binding['world_size']==8 and binding['accumulation']==1 and binding['group_weight']==1
 assert complete['status']=='FRESH_QUESTION20_G5_JOINT_EPOCH_COMPLETE'and complete['immutable_saved']is True
 assert complete['epoch']==epoch and complete['source_data_epoch']==source_epoch and complete['actual_adam_steps_per_rank']==[steps]*8
 assert complete['fresh_run_initial_adam_steps']==0 and complete['new_adam_steps_per_rank']==100
 assert bundle['epoch']==epoch and bundle['source_data_epoch']==source_epoch and bundle['checkpoint']=={key:complete[key]for key in bundle['checkpoint']}
 assert bundle['question_max_tokens']==273 and bundle['head_max_tokens']==293 and bundle['test_manifest_sha256']==TEST
 assert metrics['actual_adam_steps_per_rank']==[steps]*8 and metrics['test_manifest_sha256']==TEST
 assert metrics['train_inputs_scan_sha256']==bundle['train_scan_sha256'] and metrics['sampler_plan_sha256']
 intent=prep/('checkpoint%04d-independent-full-sha-cpu-once-v1-intent.json'%epoch)
 with (prep/('checkpoint%04d-cpu-audit.lock'%epoch)).open('a')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if intent.exists():return dict(status='EXISTING_CPU_AUDIT_INTENT_NO_BLIND_REPLAY',epoch=epoch,intent_path=str(intent))
  once(intent,dict(epoch=epoch,run_id=RUN,checkpoint_path=str(cp),producer_sha256=complete['sha256'],started_unix=time.time()))
  started=time.monotonic();full_sha_done=False;cpu_load_done=False
  try:
   sys.path.insert(0,str(b/'repo'))
   from reproduction.prefix20_group_joint100.train_epoch import natural_question_plan,cosine_learning_rate,schedule_state
   import numpy as np
   import torch
   torch.set_num_threads(1)
   plan_sha=digest(natural_question_plan(5120,c['seed'],epoch));rate=cosine_learning_rate(steps-1)
   assert metrics['sampler_plan_sha256']==plan_sha
   with cp.open('rb')as handle:
    before=os.fstat(handle.fileno());assert before.st_size==complete['bytes'];h=hashlib.sha256()
    for block in iter(lambda:handle.read(8*1024**2),b''):h.update(block)
    full_sha=h.hexdigest();assert full_sha==complete['sha256'];full_sha_done=True;handle.seek(0)
    saved=torch.load(handle,map_location='cpu',weights_only=False);cpu_load_done=True;after=os.fstat(handle.fileno())
   assert signature(before)==signature(after)==signature(cp.stat())
   assert saved['epoch']==epoch and saved['source_data_epoch']==source_epoch and saved['run_id']==RUN
   assert saved['binding']==binding and saved['binding_sha256']==BINDING and saved['actual_adam_steps_per_rank']==[steps]*8
   assert saved['fresh_run_initial_adam_steps']==0 and saved['test_manifest_sha256']==TEST
   assert saved['sampler_plan_sha256']==plan_sha and saved['train_inputs_scan_sha256']==bundle['train_scan_sha256']
   assert saved['cosine_schedule']==complete['cosine_schedule']==schedule_state(steps,10000,3e-5,3e-6)
   assert saved['actual_optimizer_learning_rates']==complete['actual_optimizer_learning_rates']==[rate]
   shapes={'block.self_attn.q_proj.weight':(2048,2048),'block.self_attn.q_proj.bias':(2048,),
    'block.self_attn.k_proj.weight':(256,2048),'block.self_attn.k_proj.bias':(256,),
    'block.self_attn.v_proj.weight':(256,2048),'block.self_attn.v_proj.bias':(256,),
    'block.self_attn.o_proj.weight':(2048,2048),'block.mlp.gate_proj.weight':(11008,2048),
    'block.mlp.up_proj.weight':(11008,2048),'block.mlp.down_proj.weight':(2048,11008),
    'block.input_layernorm.weight':(2048,),'block.post_attention_layernorm.weight':(2048,),
    'output.weight':(1,600064),'output.bias':(1,)}
   model,optimizer=saved['model'],saved['optimizer'];assert set(model)==set(shapes)
   for name,value in model.items():assert tuple(value.shape)==shapes[name]and value.dtype==torch.float32 and value.device.type=='cpu'
   groups=optimizer['param_groups'];ids=[key for group in groups for key in group['params']];states=optimizer['state']
   assert len(ids)==len(set(ids))==len(states)==14 and set(ids)==set(states)and all(group['lr']==rate for group in groups)
   for key,parameter in zip(ids,model.values()):
    state=states[key];assert int(state['step'])==steps
    for field in('exp_avg','exp_avg_sq'):
     value=state[field];assert value.shape==parameter.shape and value.dtype==torch.float32 and value.device.type=='cpu'
   rng=saved['rank_rng'];assert len(rng)==8 and[row['rank']for row in rng]==list(range(8))
   for row in rng:
    assert row['actual_adam_steps']==steps;random.Random().setstate(row['python']);np.random.RandomState().set_state(row['numpy'])
    torch.Generator(device='cpu').set_state(row['torch_cpu'])
    for key in('torch_cpu','torch_cuda'):
     value=row[key];assert value.dtype==torch.uint8 and value.device.type=='cpu'and value.numel()>0
   proof=dict(status='JOINT100_CHECKPOINT_INDEPENDENT_FULL_SHA_CPU_PASS',epoch=epoch,run_id=RUN,base=BASE,
    source_commit=SOURCE,binding_sha256=BINDING,checkpoint_path=str(cp),checkpoint_sha256=full_sha,checkpoint_bytes=before.st_size,
    stable_original_inode=True,dev=before.st_dev,inode=before.st_ino,actual_adam_steps_per_rank=[steps]*8,actual_learning_rate=rate,
    model_fp32_tensors=14,adam_fp32_states=14,rank_rng_count=8,fresh_run_initial_adam_steps=0,source_data_epoch=source_epoch,
    sampler_plan_sha256=plan_sha,test_manifest_sha256=TEST,config_sha256=small_sha(prep/'launch-config.json'),
    complete_receipt_sha256=small_sha(root/'complete.json'),data_bundle_sha256=small_sha(bundle_path),
    independent_full_file_sha_performed=True,cpu_load_performed=True,gpu_forward_performed=False,cuda_rng_applied=False,
    backup_performed=False,training_modified=False,large_payload_transferred_to_Mac=False,elapsed_seconds=time.monotonic()-started)
   del saved,model,optimizer,states,rng;gc.collect()
  except BaseException as error:
   proof=dict(status='JOINT100_CPU_AUDIT_FAILED_TRAINING_UNCHANGED',epoch=epoch,run_id=RUN,binding_sha256=BINDING,
    error_type=type(error).__name__,error=str(error),training_modified=False,
    independent_full_file_sha_performed=full_sha_done,cpu_load_performed=cpu_load_done,elapsed_seconds=time.monotonic()-started)
  once(proof_path,proof);return summary(proof,proof_path)

def main():
 parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--base',required=True);parser.add_argument('--epoch',required=True,type=int)
 args=parser.parse_args();print(json.dumps(audit(args.base,args.epoch),sort_keys=True,allow_nan=False))

if __name__=='__main__':main()
