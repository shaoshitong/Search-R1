"""Independent terminal-only CPU restore for corrected explicit-seed2 window2018-v2.

An epoch may be accepted after the whole job is COMPLETED. This validates CPU
restoration; CUDA RNG bytes and GPU next-update equivalence are not executed.
"""
import argparse,fcntl,hashlib,importlib,inspect,json,os,pathlib,random,resource,subprocess,time
A=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021')
COMMON_LOCK=A/'online-mixed-P80-handoff-CPU-acceptance-20261023-v1/acceptance.lock'
SEED2_PARENTS={
 'online-packed-window2018-seed19137-plan2026102302-L15P80-8192-20261023-v2':('online_packed_window_topk_seeded_v2','window2018',64,2018),
}
EXPECTED_SEEDS=dict(init_seed=19137,train_plan_seed=2026102302,diagnostic_seed=2026102228)
def seed_recipe(base,binding):
 module,cell,batch,capacity=SEED2_PARENTS[base.parent.name]
 assert base.name==cell and base.parent.parent==A
 assert all(binding[k]==v for k,v in EXPECTED_SEEDS.items())and binding['seed']==EXPECTED_SEEDS['train_plan_seed']and binding['seed_source']=='explicit_CLI'
 assert binding['fixed_m']==5 and binding['global_batch']==batch and binding['head_max_positions']==2048
 assert binding.get('cache_window_capacity')==capacity
 assert binding['objective']==('mse'if capacity is None else'topk_ramp')
 head=binding['initial_head_sha256'];assert isinstance(head,str)and len(head)==64 and all(c in'0123456789abcdef'for c in head)
 return dict(module='reproduction.operations.qwen7b_warmup.'+module,train_plan_seed=EXPECTED_SEEDS['train_plan_seed'],initial_head_sha256=head)
def verify_seed_stamp(metadata,recipe,head_required=False):
 assert all(metadata[k]==v for k,v in EXPECTED_SEEDS.items())
 if head_required:assert metadata['initial_head_sha256']==recipe['initial_head_sha256']
def actual_epoch_plan(groups,epoch,binding,recipe):
 module=importlib.import_module(recipe['module'])
 return module.epoch_plan(groups,epoch,binding['fixed_m'],train_plan_seed=recipe['train_plan_seed'])
def read(p):return json.loads(p.read_bytes())
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()
def proof(p):
 h=hashlib.sha256()
 with p.open('rb')as f:
  for chunk in iter(lambda:f.read(8388608),b''):h.update(chunk)
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())
def atomic(p,v):
 q=p.with_suffix('.tmp');q.write_text(json.dumps(v,sort_keys=True,allow_nan=False)+'\n');q.replace(p)
def finite(t,torch):
 assert torch.is_tensor(t)and t.device.type=='cpu'and t.layout==torch.strided
 if t.is_floating_point():
  flat=t.detach().reshape(-1)
  for i in range(0,flat.numel(),1048576):assert bool(torch.isfinite(flat[i:i+1048576]).all())
 return t.numel()
def tensor_mapping_sha(mapping,torch):
 h=hashlib.sha256()
 for name,t in mapping.items():
  h.update(name.encode());h.update(str(t.dtype).encode());h.update(repr(tuple(t.shape)).encode());raw=t.detach().contiguous().reshape(-1).view(torch.uint8)
  for i in range(0,raw.numel(),1048576):h.update(raw[i:i+1048576].numpy().tobytes())
 return h.hexdigest()
def serialized_cuda_rng_bytes(t,torch):
 if not(torch.is_tensor(t)and t.device.type=='cpu'and t.dtype==torch.uint8 and t.ndim==1 and t.numel()>=16):raise ValueError('Expected serialized CUDA RNG state: CPU uint8 vector with at least16bytes')
 return t.numel()
def validate(base,epoch,out):
 import numpy as np
 import torch
 from transformers import Qwen2Config
 from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer,Qwen2RotaryEmbedding
 from reproduction.operations.qwen7b_warmup.packed_variance_probe import PackedVarianceHead,PackedVarianceProbe
 started=time.time();submission=read(base/'root-submission.json');job=submission['job_id']
 accounting=subprocess.check_output(['sacct','-X','-n','-P','-j',job,'--format=JobIDRaw,State,ExitCode'],universal_newlines=True,timeout=15)
 if not any(row.split('|')[:3]==[job,'COMPLETED','0:0']for row in accounting.splitlines()):return dict(status='WAITING_FOR_ACTUAL_COMPLETED_JOB_CHECKPOINT_NOT_OPENED',job_id=job,accounting=accounting.strip())
 complete=read(base/'complete.json');controller=read(base/'controller.json');binding=complete['binding'];window='cache_window_capacity'in binding;recipe=seed_recipe(base,binding)
 assert complete['epochs']==10 and complete['training_questions']==8192 and not complete['final4096_scored']
 assert binding['layer']==15 and binding['prefix']==80 and binding['head_max_positions']==2048 and binding['EMA_decay']==.9995 and binding['lr']==5e-6
 assert all(controller.get(k)==v for k,v in binding.items())and controller['job_id']==job
 attempts_per_epoch=128 if window else 64;assert complete['cursor']==complete['batch_attempts']==10*attempts_per_epoch
 pub=read(base.parent/'source-publication.json');assert pub['commit']==submission['source_commit']
 for rel,ref in pub['files'].items():assert proof(pathlib.Path(pub['code_root'])/rel)==dict(path=str(pathlib.Path(pub['code_root'])/rel),**ref)
 fixture_path=pathlib.Path(binding['fixture']['path']);assert proof(fixture_path)==binding['fixture'];fixture=read(fixture_path);groupref=fixture['train_groups'];assert proof(pathlib.Path(groupref['path']))==groupref;groups=read(pathlib.Path(groupref['path']));assert len(groups)==8192
 plans={}
 for e in range(1,11):
  plan=actual_epoch_plan(groups,e,binding,recipe);coverage=read(base/('coverage-epoch-%02d.json'%e));ps=digest(plan);verify_seed_stamp(coverage,recipe);saved_plan=read(base/('sampling-plan-epoch-%02d.json'%e));verify_seed_stamp(saved_plan,recipe);assert saved_plan['rows']==plan and saved_plan['plan_sha256']==ps
  assert coverage['questions']==coverage['unique_questions']==8192 and coverage['sampling_independent_of_EM']and coverage['plan_sha256']==ps
  assert len({v['qid']for v in plan})==8192 and coverage['trajectory_visits']==sum(v['M']for v in plan)
  for view in plan:
   g=groups[view['qid']];own=dict(zip(g['trajectory_ids'],g['labels']));assert view['labels']==[own[t]for t in view['trajectory_ids']]and view['true_mixed']==(0<sum(view['labels'])<len(view['labels']))
  plans[e]=ps
 assert len(set(plans.values()))==10
 path=base/('checkpoint-epoch-%02d.pt'%epoch);before=path.stat();fresh=proof(path)
 if epoch==10:assert read(base/'latest.json')==fresh
 atomic(out/'progress.json',dict(stage='CPU_MMAP_STRICT_RESTORE',checkpoint=fresh,time=time.time()))
 blob=torch.load(path,map_location='cpu',mmap=True,weights_only=False);verify_seed_stamp(blob,recipe,head_required=True);steps=blob['actual_Adam_updates'];assert blob['binding']==binding and blob['actor_updates']==0
 assert blob['cursor']==blob['batch_attempts']==epoch*attempts_per_epoch and blob['EMA_updates']==steps and 0<steps<=blob['cursor']and blob['plan_sha256']==plans[epoch]and blob['epoch']==epoch
 assert steps+blob['margin_satisfied_batches']+blob.get('cached_only_zero_current_gradient_batches',0)==blob['cursor']
 config=Qwen2Config.from_pretrained(str(A.parent/'assets/qwen25-7b-instruct-a09a35458c70'),local_files_only=True);config._attn_implementation='flash_attention_2'
 with torch.device('meta'):
  block=Qwen2DecoderLayer(config,layer_idx=15).float();rotary=Qwen2RotaryEmbedding(config=config).float();head=PackedVarianceHead(max_positions=2048);probe=PackedVarianceProbe(block,rotary,'cache_position'in inspect.signature(block.forward).parameters,head)
 geometry={n:tuple(t.shape)for n,t in probe.state_dict().items()};assert geometry['group_head.positions.weight']==(2048,256)
 models={}
 for state in ('raw','EMA'):
  mapping=blob[state];assert set(mapping)==set(geometry)and all(tuple(mapping[n].shape)==s for n,s in geometry.items());elements=sum(finite(t,torch)for t in mapping.values());probe.load_state_dict(mapping,strict=True,assign=True)
  assert all(torch.equal(t,mapping[n])for n,t in probe.state_dict().items());models[state]=dict(tensors=len(mapping),elements=elements,strict_load=True,sha256=tensor_mapping_sha(mapping,torch))
 assert models['raw']['sha256']!=models['EMA']['sha256'];probe.load_state_dict(blob['raw'],strict=True,assign=True);opt=torch.optim.Adam(probe.parameters(),lr=5e-6);opt.load_state_dict(blob['Adam']);ids=[i for group in blob['Adam']['param_groups']for i in group['params']];params=list(probe.parameters());assert len(ids)==len(params)and set(ids)==set(blob['Adam']['state'])
 for sid,param in zip(ids,params):
  saved=blob['Adam']['state'][sid];loaded=opt.state[param];assert int(saved['step'])==int(loaded['step'])==steps
  for name in ('exp_avg','exp_avg_sq'):
   assert saved[name].shape==param.shape and torch.equal(saved[name],loaded[name]);finite(saved[name],torch)
  assert bool(saved['exp_avg_sq'].ge(0).all())
 assert all(g['lr']==5e-6 for g in opt.param_groups)
 rngs=[];assert len(blob['RNG_by_rank'])==2
 for rank,rng in enumerate(blob['RNG_by_rank']):
  assert set(rng)=={'python','numpy','torch','cuda'};random.Random().setstate(rng['python']);np.random.RandomState().set_state(rng['numpy']);torch.Generator(device='cpu').set_state(rng['torch']);rngs.append(dict(rank=rank,CPU_states_restored=True,CUDA_serialized_bytes=serialized_cuda_rng_bytes(rng['cuda'],torch),CUDA_runtime_restore_tested=False))
 queue=None
 if window:
  from detached_score_window import DetachedScoreWindow
  restored=DetachedScoreWindow.from_state_dict(blob['score_window']);state=restored.state_dict();assert not restored.has_pending and digest(state)==blob['score_window_sha256']and state['last_model_version']==blob['score_model_version']==steps and state['committed_batches']==blob['cursor']
  assert state['capacity']==binding['cache_window_capacity']and state['target_k']==500 and state['reference_population']==2018
  for entry in state['rows']+blob['last_window_entries']:
   own=dict(zip(groups[entry['qid']]['trajectory_ids'],groups[entry['qid']]['labels']));assert entry['labels']==[own[t]for t in entry['trajectory_ids']]and entry['scored_step']<=steps
  queue=dict(roundtrip_exact=True,rows=len(state['rows']),capacity=state['capacity'],sha256=blob['score_window_sha256'],model_version=steps,graph_and_hidden_features_absent=True)
 evaluation=read(base/('evaluation-epoch-%02d.json'%epoch));assert evaluation['cursor']==blob['cursor']and evaluation['binding']==binding and not evaluation['final4096_scored']
 after=path.stat();assert (before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
 result=dict(status='TERMINAL_MATCHED_WINDOW_CPU_SHA_MODEL_ADAM_RNG_PLAN_ACCEPTED',time=time.time(),job_id=job,checkpoint=fresh,epoch=epoch,batch_attempts=blob['cursor'],actual_Adam_updates=steps,EMA_updates=blob['EMA_updates'],margin_satisfied_batches=blob['margin_satisfied_batches'],cached_only_zero_current_gradient_batches=blob.get('cached_only_zero_current_gradient_batches',0),binding=binding,explicit_seed_recipe=recipe,seed_stamps=EXPECTED_SEEDS,models=models,Adam_parameter_states=len(ids),Adam_CPU_restore=True,RNG_by_rank=rngs,epoch_plan_SHAs=plans,queue=queue,source_publication=proof(base.parent/'source-publication.json'),source_complete=proof(base/'complete.json'),source_evaluation=proof(base/('evaluation-epoch-%02d.json'%epoch)),CPU_only=True,GPU_forward_or_next_update_restore_tested=False,final4096_read=False,shared_global_CPU_lock=str(COMMON_LOCK),seconds=time.time()-started,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
 atomic(out/'accepted.json',result);return result
def main():
 p=argparse.ArgumentParser();p.add_argument('--base',type=pathlib.Path,required=True);p.add_argument('--epoch',type=int,default=10);p.add_argument('--source-sha',required=True);a=p.parse_args();assert os.uname().sysname=='Linux'and os.environ.get('CUDA_VISIBLE_DEVICES')==''and 1<=a.epoch<=10 and proof(pathlib.Path(__file__))['sha256']==a.source_sha
 assert a.base.parent.name in SEED2_PARENTS and a.base.parent.parent==A
 out=A/'seed2-window2018-v2-terminal-CPU-acceptance-20261023-v1'/a.base.parent.name/a.base.name/('epoch-%02d'%a.epoch);out.mkdir(parents=True,exist_ok=True)
 with COMMON_LOCK.open('a+')as lock:
  fcntl.flock(lock,fcntl.LOCK_EX);atomic(out/'controller.json',dict(pid=os.getpid(),time=time.time(),source_sha=a.source_sha));result=validate(a.base,a.epoch,out);atomic(out/'run-complete.json',result);print(json.dumps(dict(status=result['status'],receipt=str(out/'accepted.json')if result['status'].startswith('TERMINAL_')else str(out/'run-complete.json'))))
if __name__=='__main__':main()
