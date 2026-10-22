"""Terminal-only CPU SHA/model/Adam/RNG acceptance of packed variance pilot.

Stage independently; never opens running checkpoints, allocates GPUs or changes
Slurm/training. CPU restore is not GPU forward/next-update correctness.
"""
import argparse,fcntl,gc,hashlib,inspect,json,os,resource,subprocess,time,traceback
from pathlib import Path
B=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019')
A=B/'algorithm-improvement-20261021'
BASE=A/'online-packed-full169028-M3to10-P20-20261022-v1'
OUT=A/'packed-full169028-variance-CPU-acceptance-20261022-v1'
def read(p):return json.loads(Path(p).read_bytes())
def atomic(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.%d.tmp'%os.getpid())
 with tmp.open('w')as f:json.dump(v,f,sort_keys=True,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
 tmp.replace(p)
def sha(p,progress=None,**binding):
 p=Path(p);h=hashlib.sha256();n=0
 with p.open('rb')as f:
  for chunk in iter(lambda:f.read(8*1024**2),b''):
   h.update(chunk);n+=len(chunk)
   if progress and(n%(256*1024**2)==0 or n==p.stat().st_size):
    atomic(progress,dict(stage='ACTUAL_CHECKPOINT_SHA_READING',bytes_read=n,total_bytes=p.stat().st_size,time=time.time(),**binding))
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())
def stamp(p):
 s=Path(p).stat();return dict(device=s.st_dev,inode=s.st_ino,bytes=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns)
def finite_tensor(t,torch):
 if not torch.is_tensor(t):raise ValueError('Actual tensor required')
 if t.device.type!='cpu' or t.layout!=torch.strided:raise ValueError('CPU dense checkpoint tensor required')
 if t.is_floating_point() or t.is_complex():
  flat=t.detach().reshape(-1)
  for offset in range(0,flat.numel(),1048576):
   if not bool(torch.isfinite(flat[offset:offset+1048576]).all()):raise ValueError('Nonfinite tensor')
 return t.numel()
def mapping_sha(mapping,torch):
 h=hashlib.sha256()
 for name,t in mapping.items():
  h.update(name.encode());h.update(str(t.dtype).encode());h.update(repr(tuple(t.shape)).encode())
  raw=t.detach().contiguous().reshape(-1).view(torch.uint8)
  for offset in range(0,raw.numel(),1048576):h.update(raw[offset:offset+1048576].numpy().tobytes())
 return h.hexdigest()


def validate(loss):
 base=BASE/loss;dest=OUT/loss;dest.mkdir(exist_ok=True)
 import numpy as np
 import random
 import torch
 from transformers import Qwen2Config
 from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer,Qwen2RotaryEmbedding
 from reproduction.operations.qwen7b_warmup.packed_variance_probe import PackedVarianceHead,PackedVarianceProbe,plan
 variant='full169028-'+loss+'-one-question-M3to10-P20-variance';receipt=base/'root-submission.json'
 if not receipt.exists()or not(base/'complete.json').exists():return dict(status='WAITING_FOR_PUBLISHED_COMPLETE_NOT_OPENING_CHECKPOINT',variant=variant)
 submission=read(receipt);job=str(submission['job_id']);actual=subprocess.check_output(['sacct','-X','-n','-P','-j',job,'--format=JobIDRaw,State,ExitCode'],universal_newlines=True,timeout=10)
 if not any(row.split('|')[:3]==[job,'COMPLETED','0:0']for row in actual.splitlines()):return dict(status='WAITING_FOR_ACTUAL_COMPLETED_0_0_NOT_OPENING_CHECKPOINT',variant=variant,job_id=job,sacct=actual.strip())
 complete=read(base/'complete.json');controller=read(base/'controller.json');latest=read(base/'latest.json');ready=read(base/'ready.json')
 if complete!=dict(status='PACKED_VARIANCE_FULL_SWEEP_COMPLETE',epochs=10,cursor=13210,training_questions=169028,pilot_not_full169028=False,final4096_scored=False):raise ValueError('Actual completed8192 packed3epoch192 budget differs')
 for key,value in dict(job_id=job,layer=21,capture_layer=20,prefix=20,M_range=[3,10],question_prompt_occurrences_per_group=1,actor_rows_per_group=1,target='4*selected_M_population_EM_variance',loss=loss.upper()+'_only',selection_score='predicted_population_EM_variance',selection_order='descending',candidates_same_M=True,variance_G5_diagnostic_threshold=.08,EMA_decay=.9995,anchor_weight=1.0,normalization_epsilon=1e-6,normalization_scope='actual_global_optimizer_batch'if loss=='z_mse_anchor'else None,raw_updates_EMA=True,training_questions=169028,pilot_not_full169028=False,final4096_read=False).items():
  if controller.get(key)!=value:raise ValueError('Actual packed controller field differs: '+key)
 if ready['loss']!=loss or ready['training']['loss']!=loss or ready['training']['updates']!=13210 or ready['training']['epochs']!=10 or ready['training']['EMA_decay']!=.9995 or ready['training']['pilot_checkpoint_resumed']is not False:raise ValueError('Full10epoch fresh-official loss/EMA/budget binding differs')
 pubref=ready['source_publication'];publication=read(pubref['path'])
 if sha(pubref['path'])!=pubref:raise ValueError('Packed source publication SHA changed')
 for name in('packed_variance_probe.py','online_packed_variance_scaling.py','runtime.sh'):
  rel='reproduction/operations/qwen7b_warmup/'+name;p=Path(publication['code_root'])/rel;ref=publication['files'][rel]
  if sha(p)!=dict(path=str(p),**ref):raise ValueError('Packed immutable source file SHA differs')
 fixturebase=BASE/'fixture';fixture=read(fixturebase/'fixture-complete.json');groups=read(fixture['train_groups']['path']);expectedplans={};coverage={}
 if sha(fixture['train_groups']['path'])!=fixture['train_groups']or len(groups)!=169028:raise ValueError('Source actual8192 G10 groups differ')
 for epoch in range(1,11):
  rows=plan(groups,epoch,2026102227);c=read(base/('coverage-epoch-%02d.json'%epoch));ps=hashlib.sha256(json.dumps(rows,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest();hist={str(m):sum(r['M']==m for r in rows)for m in range(3,11)}
  if (len(rows)!=169028 or len({r['qid']for r in rows})!=169028 or c['questions']!=169028 or c['unique_questions']!=169028 or c['M_counts']!=hist or c['plan_sha256']!=ps or c['sampling_independent_of_EM']is not True or c['trajectory_visits']!=sum(r['M']for r in rows)):raise ValueError('Actual epoch own-M plan or coverage differs')
  for r in rows:
   g=groups[r['qid']];idx=[g['trajectory_ids'].index(t)for t in r['trajectory_ids']]
   if not 3<=r['M']<=10 or len(set(idx))!=r['M']or r['labels']!=[g['labels'][i]for i in idx]:raise ValueError('Selected M trajectories or own EM labels differ')
   mean=sum(r['labels'])/r['M'];v=mean*(1-mean)
   if r['target_variance']!=v or r['target4variance']!=4*v:raise ValueError('Actual population variance target differs')
  expectedplans[epoch]=ps;coverage[str(epoch)]=dict(plan_sha256=ps,questions=169028,unique_questions=169028,M_counts=hist,trajectory_visits=c['trajectory_visits']);del rows
 if len(set(expectedplans.values()))!=10:raise ValueError('Epoch sample plans unexpectedly identical')
 path=Path(latest['path']);expected=base/'checkpoint-epoch-10.pt'
 if path!=expected or path.is_symlink()or path.stat().st_size!=latest['bytes']:raise ValueError('Immutable final epoch3 checkpoint path required')
 accepted=dest/'accepted.json'
 if accepted.exists():
  old=read(accepted);current=stamp(path)
  if old['checkpoint']!=latest or any(old['file_stat'][k]!=current[k]for k in('inode','bytes','mtime_ns','ctime_ns')):raise ValueError('Previously accepted packed checkpoint changed')
  return dict(status='EXISTING_PACKED_CPU_ACCEPTANCE_REUSED',variant=variant,accepted=str(accepted),checkpoint=latest)
 started=time.time();before=stamp(path);progress=OUT/'progress.json';binding=dict(variant=variant,job_id=job);actualproof=sha(path,progress,**binding)
 if actualproof!=latest:raise ValueError('Fresh entire final packed checkpoint SHA differs')
 atomic(progress,dict(stage='CPU_MMAP_TORCH_LOAD',checkpoint=actualproof,time=time.time(),**binding));blob=torch.load(path,map_location='cpu',mmap=True,weights_only=False)
 if (blob['cursor']!=13210 or blob['epoch']!=10 or blob['layer']!=21 or blob['prefix']!=20 or blob['training_questions']!=169028 or blob['architecture']!='one_question_M3to10_policy20_packed_variance_v1'or blob['loss']!=loss.upper()+'_only_target4variance'or blob['loss_name']!=loss or blob['EMA_decay']!=.9995 or blob['anchor_weight']!=1.0 or blob['normalization_epsilon']!=1e-6 or blob['fixture']!=sha(fixturebase/'fixture-complete.json')or blob['lr']!=5e-6 or blob['raw_updates_EMA']is not True or len(blob['RNG_by_rank'])!=2 or blob['plan_sha256']!=expectedplans[10]):raise ValueError('Actual packed architecture/counters/objective or sampler plan differs')
 config=Qwen2Config.from_pretrained(str(B/'assets/qwen25-7b-instruct-a09a35458c70'),local_files_only=True);config._attn_implementation='flash_attention_2'
 with torch.device('meta'):
  block=Qwen2DecoderLayer(config,layer_idx=21).float();rotary=Qwen2RotaryEmbedding(config=config).float();head=PackedVarianceHead();probe=PackedVarianceProbe(block,rotary,'cache_position'in inspect.signature(block.forward).parameters,head)
 required={n:tuple(t.shape)for n,t in probe.state_dict().items()};names=[n for n,_ in probe.named_parameters()]
 if required.get('group_head.positions.weight')!=(1024,256)or required.get('group_head.output.weight')!=(1,256):raise ValueError('Real packed regression head geometry differs')
 geometry={}
 for key in('raw','EMA'):
  mapping=blob[key]
  if set(mapping)!=set(required)or any(tuple(mapping[n].shape)!=shape for n,shape in required.items()):raise ValueError('Checkpoint incompatible with exact packed native block/head')
  total=sum(finite_tensor(t,torch)for t in mapping.values());probe.load_state_dict(mapping,strict=True,assign=True)
  if any(not torch.equal(t,mapping[n])for n,t in probe.state_dict().items()):raise ValueError('Actual strict-loaded packed model differs')
  geometry[key]=dict(tensors=len(mapping),parameter_elements=total,strict_load_state_dict_passed=True,independent_dtype_shape_tensor_SHA256=mapping_sha(mapping,torch))
 if geometry['raw']['independent_dtype_shape_tensor_SHA256']==geometry['EMA']['independent_dtype_shape_tensor_SHA256']:raise ValueError('Final raw/EMA unexpectedly identical')
 probe.load_state_dict(blob['raw'],strict=True,assign=True);opt=torch.optim.Adam(probe.parameters(),lr=5e-6);opt.load_state_dict(blob['Adam']);source=blob['Adam'];ids=[i for g in source['param_groups']for i in g['params']]
 if len(ids)!=len(names)or len(set(ids))!=len(ids)or set(ids)!=set(source['state']):raise ValueError('Complete packed native/head Adam state required')
 moments=0
 for name,sid,param in zip(names,ids,probe.parameters()):
  saved=source['state'][sid];loaded=opt.state[param]
  if (not {'step','exp_avg','exp_avg_sq'}<=set(saved)or int(saved['step'])!=13210 or int(loaded['step'])!=13210 or tuple(saved['exp_avg'].shape)!=tuple(param.shape)or tuple(saved['exp_avg_sq'].shape)!=tuple(param.shape)):raise ValueError('Packed Adam192 clock or moment geometry differs: '+name)
  for key in('exp_avg','exp_avg_sq'):
   moments+=finite_tensor(saved[key],torch)
   if not torch.equal(saved[key],loaded[key]):raise ValueError('Actual packed Adam restore differs: '+name)
  if not bool(saved['exp_avg_sq'].ge(0).all()):raise ValueError('Negative packed Adam second moment')
 if any(g['lr']!=5e-6 for g in source['param_groups']):raise ValueError('Actual packed Adam LR differs')
 rngproof=[]
 for rank,rng in enumerate(blob['RNG_by_rank']):
  if set(rng)!={'python','numpy','torch','cuda'}:raise ValueError('Complete two-rank RNG required')
  random.Random().setstate(rng['python']);np.random.RandomState().set_state(rng['numpy']);torch.Generator(device='cpu').set_state(rng['torch']);cuda=rng['cuda']
  if not torch.is_tensor(cuda)or cuda.dtype!=torch.uint8 or cuda.ndim!=1 or cuda.numel()<16 or cuda.device.type!='cpu':raise ValueError('Actual CUDA RNG byte-state required')
  rngproof.append(dict(rank=rank,python_state_accepted=True,numpy_state_accepted=True,CPU_generator_state_accepted=True,CUDA_RNG_serialized_bytes=int(cuda.numel()),CUDA_runtime_restore_tested=False))
 for epoch in range(11):
  ev=read(base/('evaluation-epoch-%02d.json'%epoch))
  if ev['cursor']!=1321*epoch or ev['final4096_scored']is not False or ev['threshold_predeclared_G5_variance']!=.08 or ev['loss']!=loss.upper()+'(4*population_variance)':raise ValueError('Actual epoch evaluation/objective differs')
  for role,states in ev['metrics'].items():
   for metrics in states.values():
    if metrics['selection_score']!='predicted_population_EM_variance'or metrics['selection_order']!='descending'or metrics['strict_EM_used_for_ranking']is not False:raise ValueError('Selection is not label-blind predicted variance')
 normalization_proof=None
 if loss=='z_mse_anchor':
  count=0;tail_epochs=set()
  with(base/'curve.jsonl').open()as f:
   for line in f:
    row=json.loads(line);al=row['alignment'];expected=68 if row['epoch_questions_seen']==84514 else 128
    if al['normalization_global_questions']!=expected or row['actual_global_batch']!=expected or al['normalization_scope']!='actual_global_optimizer_batch_autograd_all_gather'or al['normalization_population_std']is not True or al['normalization_epsilon']!=1e-6 or al['anchor_weight']!=1.0:raise ValueError('Actual global-batch normalization evidence differs')
    if any(not __import__('math').isfinite(al[k])for k in('normalized_MSE','anchor4variance_MSE','total_loss','prediction_mean','prediction_std','target_mean','target_std')):raise ValueError('Nonfinite normalization evidence')
    if expected==68:tail_epochs.add(row['epoch'])
    count+=1
  if count!=13210 or tail_epochs!=set(range(1,11)):raise ValueError('Complete10epoch normalization cursor/tail evidence required')
  p0=read(base/'progress-rank-0.json');p1=read(base/'progress-rank-1.json')
  if p0['cursor']!=13210 or p1['cursor']!=13210 or p0['alignment']!=p1['alignment']:raise ValueError('Final two-rank global normalization evidence differs')
  normalization_proof=dict(scope='actual_global_optimizer_batch',full_batch=128,tail_batch=68,actual_updates=count,tail_epochs=sorted(tail_epochs),final_rank_statistics_exactly_equal=True,mean_and_population_std_gradient_path='autograd_all_gather_in_published_source',STD_epsilon_floor=1e-6,anchor_weight=1.0)
 if before!=stamp(path):raise ValueError('Final packed checkpoint changed during CPU validation')
 result=dict(status='TERMINAL_FULL169028_PACKED_VARIANCE_CPU_SHA_MODEL_ADAM_RNG_PLAN_ACCEPTED',variant=variant,job_id=job,checkpoint=actualproof,file_stat=before,cursor=13210,epochs=10,loss=loss,EMA_decay=.9995,layer=21,prefix=20,M_range=[3,10],geometry=geometry,Adam=dict(actual_steps=13210,parameter_states=len(ids),finite_moment_elements=moments,actual_optimizer_state_CPU_load_passed=True),RNG_by_rank=rngproof,epoch_coverage=coverage,normalization=normalization_proof,data_cursor=dict(completed_epoch=10,completed_optimizer_updates=13210,next_epoch=11,within_next_epoch_question_position=0,restoration='deterministic_own_M_plan_from_seed2026102227_and_epoch_boundary'),source_training_publication=sha(pubref['path']),source_controller=sha(base/'controller.json'),source_complete=sha(base/'complete.json'),source_fixture=sha(fixturebase/'fixture-complete.json'),selection_score='predicted_population_EM_variance',single_question_and_one_prefill_row_per_group_controller_verified=True,raw_and_EMA_separate=True,actual_GPU_forward_or_next_update_restore_tested=False,CPU_restore_only=True,no_final4096_read=True,probe_updates=0,actor_updates=0,peak_RSS_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,seconds=time.time()-started,time=time.time());atomic(accepted,result);return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--source-sha',required=True);args=p.parse_args()
 if os.uname().sysname!='Linux'or os.environ.get('CUDA_VISIBLE_DEVICES','')!='':raise RuntimeError('Remote CPU-only acceptance required')
 if sha(__file__)['sha256']!=args.source_sha:raise ValueError('Immutable validator source SHA differs')
 OUT.mkdir(exist_ok=True);guard=(OUT/'acceptance.lock').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);atomic(OUT/'controller.json',dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],source_sha=args.source_sha,time=time.time()))
 try:
  results=[]
  for loss in('mse','mae','z_mse_anchor'):
   result=validate(loss);results.append(result);gc.collect()
  atomic(OUT/'run-complete.json',dict(results=results,time=time.time()));print(json.dumps(dict(status='FULL_VARIANCE_TERMINAL_CPU_CHECKS_FINISHED',results=[dict(variant=r['variant'],status=r['status'])for r in results])))
 except BaseException as e:atomic(OUT/'failure.json',dict(error=repr(e),traceback=traceback.format_exc(),time=time.time()));raise
if __name__=='__main__':main()
