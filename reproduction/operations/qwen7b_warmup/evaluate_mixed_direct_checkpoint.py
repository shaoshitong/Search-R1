"""Frozen CP128 raw/EMA direct-Mixed snapshot, separate from live training.

Restore and hash every saved component, then score only accepted fixed512/G5
and2048/G5 features. The276GB training feature bank is never opened. No optimizer
step, threshold fitting, final-pool scoring, or mutation of the main namespace.
"""
import argparse,fcntl,gc,json,os,random,time,traceback
from pathlib import Path
import numpy as np
import torch
from . import probe_ablation_train as u,mixed_group_checkpoint as ck
from . import joint_mixed_direct_pretrain as source
from .balanced_hard_g10_sampler import BalancedHardG10Sampler
from .mixed_direct_evaluation import live_state_hashes,score_reader
from .native_mixed_group_classifier import OFFICIAL_REVISION
BASE=source.B/'algorithm-improvement-20261021/mixed-direct-official-cp128-eval-v1'
SOURCE_CONFIG_SHA='2a7f039f4172f5284e3b8282e8ca6200e566bd92bb3013e9802306eca84e7393'
CURSOR=128


def proof(path):
 p=Path(path);return dict(path=str(p),bytes=p.stat().st_size,sha256=u.sha(p))


def validate_receipts(v,base=BASE):
 """Metadata guard; fresh checkpoint bytes/SHA happen on the owned GPU job."""
 cp=v['checkpoint'];original=v['source_checkpoint_original'];cfg=v['source_config'];main=v['source_main_config']
 if (v.get('mode')!='read_only_checkpoint_snapshot'or cfg.get('path')!=str(base/'source-config.json')or cfg.get('sha256')!=SOURCE_CONFIG_SHA or cfg.get('bytes')!=9615 or main!=dict(cfg,path=str(source.MAIN/'config.json'))or cp.get('cursor')!=CURSOR or cp.get('native_steps')!=CURSOR or cp.get('head_steps')!=CURSOR or cp.get('config_sha256')!=SOURCE_CONFIG_SHA or cp.get('full_CPU_native_head_two_Adam_RNG_sampler_source_reload_passed')is not True or set(cp.get('accepted_hashes',{}))!={'raw','ema','native_adam','head_adam','raw_buffers','ema_buffers','rng','sampler','official_source'}):raise ValueError('Exact accepted official CP128 and original config receipts required')
 if not Path(cp['path']).is_absolute()or not str(Path(cp['path'])).startswith(str(base/'inputs')+'/')or not str(Path(original['path'])).startswith(str(source.MAIN/'checkpoints')+'/')or{key:value for key,value in cp.items()if key!='path'}!={key:value for key,value in original.items()if key!='path'}:raise ValueError('Pinned snapshot changes only pathname, never source receipt/SHA/state')
 return True


def source_binding(c,config_sha):
 return dict(config_sha256=config_sha,matched_binding_sha256=u.digest(c),training_groups=c['groups'],features=c['features'],sampler_source=source.SAMPLER,initialization_source='official_initial',official_revision=OFFICIAL_REVISION,official_inventory=c['official_inventory'])


def full_live_hashes(model,ema,block_opt,head_opt,device,sampler,provenance):
 return dict(live_state_hashes(model,ema,block_opt,head_opt,device),sampler=u.digest(sampler.state_dict()),official_source=ck.tree_hash(provenance))


def score_pair(model,ema,block_opt,head_opt,device,sampler,provenance,pools,folder,progress,expected_hashes):
 """Both models, two Adam clocks, all RNG/native buffers/sampler frozen."""
 before=full_live_hashes(model,ema,block_opt,head_opt,device,sampler,provenance)
 if before!=expected_hashes:raise ValueError('Snapshot evaluation input differs from restored complete checkpoint')
 clocks=(ck.optimizer_steps(block_opt.state_dict(),True),ck.optimizer_steps(head_opt.state_dict(),True));rng=u.rng_state(device);buffers=(u.buffers(model),u.buffers(ema));modes=[[(m,m.training)for m in probe.modules()]for probe in(model,ema)];metrics={};folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
 try:
  for label,probe in(('raw',model),('EMA',ema)):
   metrics[label]={}
   for name,(groups,reader)in pools.items():metrics[label][name]=score_reader(probe,reader,groups,device,folder/(label+'-'+name+'.jsonl.gz'),progress,label+'-'+name,micro_groups=1)
 finally:
  u.restore_rng(rng,device);u.restore_buffers(model,buffers[0]);u.restore_buffers(ema,buffers[1])
  for values in modes:
   for module,training in values:module.training=training
 after=full_live_hashes(model,ema,block_opt,head_opt,device,sampler,provenance)
 if before!=after or clocks!=(ck.optimizer_steps(block_opt.state_dict(),True),ck.optimizer_steps(head_opt.state_dict(),True)):raise ValueError('Read-only CP snapshot changed parameters/EMA/twoAdam/RNG/buffers/sampler/source')
 return dict(metrics=metrics,full_components_before=before,full_components_after=after,native_Adam_clock=clocks[0],group_head_Adam_clock=clocks[1],all_saved_training_components_unchanged=True)


def run(base):
 if not os.environ.get('SLURM_JOB_ID')or not torch.cuda.is_available()or torch.cuda.device_count()!=1 or'H100'not in torch.cuda.get_device_properties(0).name:raise ValueError('Own oneH100 read-only snapshot allocation required')
 v=u.read(base/'evaluation-config.json');validate_receipts(v,base)
 for name,sha in v['code_hashes'].items():
  if u.sha(Path(__file__).resolve().parents[3]/name)!=sha:raise ValueError('Published snapshot-worker code differs')
 if(base/'complete.json').exists():
  prior=u.read(base/'complete.json')
  if prior['evaluation_config']!=proof(base/'evaluation-config.json')or prior['checkpoint']!=v['checkpoint']:raise ValueError('Existing immutable snapshot complete has other bindings')
  for items in prior['metrics'].values():
   for item in items.values():u.checked(item['predictions'])
  return prior
 cpath=u.checked(v['source_config']);mpath=u.checked(v['source_main_config'])
 if cpath.read_bytes()!=mpath.read_bytes():raise ValueError('Snapshot must retain exact original config bytes/base/code hashes')
 c=u.read(cpath);source.typed_config(source.MAIN,c)
 if c['official_revision']!=OFFICIAL_REVISION or c['initialization_source']!='official_initial':raise ValueError('Only official-start direct Mixed source accepted')
 device=torch.device('cuda',0);torch.cuda.set_device(device);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
 groups,blocked,hashes=source.own_groups(c);sampler=BalancedHardG10Sampler(groups,seed=c['seed'],batch_questions=32,steps_per_epoch=c['steps_per_epoch'],forbidden_qids=blocked,forbidden_hashes=hashes);binding=source_binding(c,u.sha(cpath))
 u.atomic(base/'controller.json',dict(pid=os.getpid(),job_id=os.environ['SLURM_JOB_ID'],stage='FRESH_OFFICIAL_AND_PINNED_CP_SHA_RESTORE',checkpoint=v['checkpoint'],config_sha256=u.sha(base/'evaluation-config.json'),time=time.time(),source_config=v['source_config'],training_features_reader_constructed=False))
 # This verifies all official model files but loads only the last native block.
 model,ema,block_opt,head_opt,fresh_provenance=source.initialize(c,device)
 # First actual full pinned checkpoint SHA belongs here, never a login load.
 cp=v['checkpoint'];cp_path=u.checked(cp);saved=torch.load(cp_path,map_location='cpu',mmap=True,weights_only=False);cursor=ck.validate(saved,binding)
 if cursor!=CURSOR or saved['accepted_hashes']!=cp['accepted_hashes']or saved['matched_binding_sha256']!=cp['matched_binding_sha256']:raise ValueError('Pinned actual fullstate/hash/cursor differs from accepted source receipt')
 if ck.tree_hash(fresh_provenance)!=ck.tree_hash(saved['official_initialization']):raise ValueError('Fresh official inventory/block/head lineage differs from original initialization')
 cursor,seen=ck.restore(saved,model,ema,block_opt,head_opt,device,sampler,binding);provenance=saved['official_initialization'];accepted=dict(saved['accepted_hashes']);sampler_hash=saved['sampler_state_sha256'];del saved;gc.collect()
 if ck.optimizer_steps(block_opt.state_dict(),True)!=CURSOR or ck.optimizer_steps(head_opt.state_dict(),True)!=CURSOR or sampler.accepted_updates!=CURSOR:raise ValueError('Full native/head Adam128 and sampler128 restoration required')
 restored=dict(cursor=cursor,accepted_hashes=accepted,sampler_state_sha256=sampler_hash,training_seen_questions=len(seen),checkpoint=cp,full_CP_fresh_SHA_verified=True,actual_complete_restore_passed=True,training_features_reader_constructed=False);u.atomic(base/'restore-complete.json',restored)
 from .joint_balanced_native_evaluation import accepted_HF_validation
 from .replay_feature_reader import ReplayFeatureReader
 spec=c['evaluation'];diagnostic=u.read(u.checked(spec['diagnostic_groups']))['diagnostic'];validation,vr,vproof=accepted_HF_validation(spec);dr=None
 if len(diagnostic)!=512 or len(validation)!=2048:vr.close();raise ValueError('Exactly fixed512 and ownG5 fixed2048 only')
 excluded={g['qid']:g['question_hash']for g in u.read(u.checked(c['exclusions']))['groups']}
 if any(g['qid']not in excluded or(g.get('question_hash')and excluded[g['qid']]!=g['question_hash'])for g in diagnostic+validation):vr.close();raise ValueError('Fixed scoring identities must be excluded from full169028 training')
 try:
  dr=ReplayFeatureReader(u.read(u.checked(spec['diagnostic_features'])),26)
  value=score_pair(model,ema,block_opt,head_opt,device,sampler,provenance,dict(diagnostic512=(diagnostic,dr),fixed_selection2048=(validation,vr)),base/'predictions',base/'evaluation-progress.json',accepted)
 finally:
  vr.close()
  if dr is not None:dr.close()
 result=dict(status='OFFICIAL_DIRECT_MIXED_CP128_RAW_EMA_FIXED512_2048_FROZEN_EVALUATION_COMPLETE',cursor=cursor,checkpoint=cp,source_checkpoint_original=v['source_checkpoint_original'],source_config=v['source_config'],source_main_config=v['source_main_config'],evaluation_config=proof(base/'evaluation-config.json'),source_config_base_preserved=str(source.MAIN),native_Adam_updates_during_evaluation=0,group_head_Adam_updates_during_evaluation=0,actor_updates=0,score_semantics='sigmoid_direct_Mixed_group_logit',trajectory_probability_or_accuracy_produced=False,training_features_payload_read=False,final4096_scored=False,threshold_fitted=False,validation_proof=vproof,restoration=restored,completed_time=time.time(),**value);u.atomic(base/'complete.json',result);return result


def main():
 p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);a=p.parse_args();base=a.base.resolve()
 if base!=BASE:p.error('Only independent official CP128 snapshot namespace')
 base.mkdir(parents=True,exist_ok=True)
 with(base/'evaluation.lock').open('a+')as guard:
  fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:print(json.dumps(run(base)),flush=True)
  except BaseException as e:u.atomic(base/'failure.json',dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc(),time=time.time()));raise
if __name__=='__main__':main()
