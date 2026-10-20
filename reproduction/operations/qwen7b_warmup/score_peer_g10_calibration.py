"""Frozen E4/peer256 scoring of authentic seen-training NQ8192 G10.
Standalone overlay: all model/reader imports use qualified d8f74420 runtime.
No optimizer, response generation, evaluation/final scoring or weight updates.
"""
import argparse,fcntl,gzip,hashlib,json,math,os,platform,shlex,sys,time,traceback
from pathlib import Path
B=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019')
A=B/'algorithm-improvement-20261021';TRAIN=A/'peer-newseed8192-g10-train-v1';TARGET=A/'peer8192-frozen-calibration-scores-v2'
PIN='d8f74420aa09fa56e8ec59207c6d1453fa86f95e';ROLE='SEEN_NQ8192_TRAINING_CALIBRATION_ONLY'
STATUS='FROZEN_E4_AND_PEER256_REAL_G10_CALIBRATION_SCORES_COMPLETE'
CP_SHA='36d9bcc3be71ba94eaca3a1e1b292c0ed9a65297fb96a746ea5295432f09c000'
E4_SHA='e29168c36de152a57294f055e7d420cbba25cae31aeace32e793ba1098bf97a8'
def read(p):return json.loads(Path(p).read_text())
def digest(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def proof(p):
 p=Path(p);h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())
def checked(r):
 p=Path(r['path'])
 if not p.is_absolute() or p.resolve()!=p or not str(p).startswith(str(B)+'/'):raise ValueError('Remote shared-storage source required')
 actual=proof(p)
 if actual['sha256']!=r['sha256'] or ('bytes' in r and actual['bytes']!=r['bytes']):raise ValueError('Source SHA/size differs: '+str(p))
 return p
def atomic(p,v):
 p=Path(p);tmp=p.with_suffix(p.suffix+'.tmp')
 with tmp.open('w') as f:json.dump(v,f,sort_keys=True,allow_nan=False);f.flush();os.fsync(f.fileno())
 tmp.replace(p)
def immutable(p,v):
 if Path(p).exists():
  if read(p)!=v:raise ValueError('Immutable output differs: '+str(p))
 else:atomic(p,v)
def runtime(pub):
 if pub['commit']!=PIN:raise ValueError('Only qualified d8f74420 runtime may be imported')
 root=Path(pub['code_root'])
 if root!=TRAIN/('code-'+PIN[:12]):raise ValueError('Unexpected pinned runtime path')
 for name,item in pub['files'].items():
  if Path(name).is_absolute() or '..' in Path(name).parts:raise ValueError('Unsafe source name')
  checked(dict(path=str(root/name),**item))
 sys.path.insert(0,str(root));return root

def identities(groups):
 if len(groups)!=8192:raise ValueError('Exact original8192 training questions required')
 qids=set();hashes=set();tids=set()
 for g in groups:
  if (g['qid'] in qids or g['question_hash'] in hashes or len(g['trajectory_ids'])!=10 or len(set(g['trajectory_ids']))!=10 or tids.intersection(g['trajectory_ids']) or len(g['labels'])!=10 or any(type(y) is not int or y not in (0,1) for y in g['labels'])):raise ValueError('Unique own authentic G10/strict EM required')
  qids.add(g['qid']);hashes.add(g['question_hash']);tids.update(g['trajectory_ids'])
 return dict(questions=len(qids),trajectories=len(tids),ordered_qids_sha256=digest([g['qid'] for g in groups]),ordered_own_trajectory_ids_sha256=digest([g['trajectory_ids'] for g in groups]),ordered_own_strict_em_sha256=digest([g['labels'] for g in groups]))

def score_row(g,z,p):
 if len(z)!=10 or len(p)!=10 or any(not math.isfinite(v) for v in z) or any(not math.isfinite(v) or not 0<=v<=1 for v in p):raise ValueError('Ten finite own logits/probabilities required')
 for logit,prob in zip(z,p):
  want=1/(1+math.exp(-logit)) if logit>=0 else math.exp(logit)/(1+math.exp(logit))
  if abs(want-prob)>2e-7:raise ValueError('Probability/logit order or sigmoid differs')
 ac=aw=1.
 for v in p:ac*=v;aw*=1-v
 return dict(qid=g['qid'],question_hash=g['question_hash'],trajectory_ids=list(g['trajectory_ids']),strict_em=list(g['labels']),logits=list(z),p_j=list(p),predicted_mixed_G10=max(0.,1-ac-aw))

def render_sbatch(code,config_sha):
 code=Path(code)
 if code.parent!=TARGET or not code.name.startswith('code-') or len(config_sha)!=64:raise ValueError('Separate immutable scorer code/config required')
 return '''#!/usr/bin/env bash
#SBATCH --job-name=q7b-peer8192-frozen-cal
#SBATCH --partition=acd_u
#SBATCH --account=mou176
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=@@CAL_BASE@@/logs/scoring-%j.log
#SBATCH --chdir=@@CAL_BASE@@
set -euo pipefail
export QWEN7B_CODE_ROOT=@@PINNED_ROOT@@
export FLASH_ATTENTION_DETERMINISTIC=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
srun --kill-on-bad-exit=1 bash "$QWEN7B_CODE_ROOT/reproduction/operations/qwen7b_warmup/runtime.sh" -u @@SCORER@@ --action run --config-sha256 @@CONFIG_SHA@@
'''.replace('@@CAL_BASE@@',str(TARGET)).replace('@@PINNED_ROOT@@',shlex.quote(str(TRAIN/('code-'+PIN[:12])))).replace('@@SCORER@@',shlex.quote(str(code/Path(__file__).name))).replace('@@CONFIG_SHA@@',config_sha)

def prepare():
 if platform.system()!='Linux':raise ValueError('Preparation metadata stays remote')
 TARGET.mkdir(parents=True,exist_ok=True);(TARGET/'logs').mkdir(exist_ok=True)
 pubref=proof(TRAIN/'source-publication.json');pub=read(pubref['path']);runtime(pub)
 from reproduction.operations.qwen7b_warmup import peer_newseed_train as fit
 c=read(TRAIN/'config.json');done=read(TRAIN/'complete.json');latest=read(TRAIN/'latest.json')
 if (done['status']!='REAL_G10_QUESTION_WEIGHTED_PROBE_BUDGET_AND_VALIDATION_COMPLETE' or done['cursor']!=256 or done['steps']!=10720 or done['checkpoint']!=latest or latest['sha256']!=CP_SHA or latest['config_sha256']!=proof(TRAIN/'config.json')['sha256'] or c['feature_backend']!='huggingface_frozen_causal_prefill'):raise ValueError('Actual accepted completed peerCP256 required')
 groups,manifests,acceptance=fit.accepted_peer_g10(c['g10_complete']);ids=identities(groups)
 truth=[{k:g[k] for k in ('qid','question_hash','trajectory_ids','labels')} for g in groups];immutable(TARGET/'groups-8192.json',truth)
 source=read(TRAIN/'inputs/E4-only-config.json')['same_HF_frozen_validation_E4_baseline']
 if source['sha256']!=E4_SHA or source['steps']!=14418:raise ValueError('Sealed E4 source differs')
 for item in (source,latest):
  if Path(item['path']).stat().st_size!=item['bytes']:raise ValueError('Checkpoint bytes differ')
 peer={k:latest[k] for k in ('path','bytes','sha256','steps','cursor','accepted_hashes')};peer.update(name='peer256',source_config=proof(TRAIN/'config.json'),source_receipt=proof(TRAIN/'latest.json'))
 publication=read(TARGET/'source-publication.json');code=Path(publication['code_root'])
 for name,item in publication['files'].items():checked(dict(path=str(code/name),**item))
 config=dict(role=ROLE,base=str(TARGET),pinned_runtime_publication=pubref,source_publication=proof(TARGET/'source-publication.json'),source_training_config=proof(TRAIN/'config.json'),training_input=c['g10_complete'],source_plan=c['plan'],source_training_complete=proof(TRAIN/'complete.json'),groups=proof(TARGET/'groups-8192.json'),checkpoints=[source,peer],model_dir=c['model_dir'],model_config=c['model_config'],micro_questions=4,progress_questions=16,identities=ids,actor_updates=0,probe_updates=0,optimizer_constructed=False,seen_training_data=True,unseen_performance_claim=False,evaluation_or_final_scored=False,feature_backend='qualified_frozen_HF_batch32_cap26',same_cohort_as_peer_fit=True,labels_affect_inference=False)
 immutable(TARGET/'config.json',config);sha=proof(TARGET/'config.json')['sha256'];script=render_sbatch(code,sha);p=TARGET/'scoring.sbatch'
 if p.exists() and p.read_text()!=script:raise ValueError('Immutable sbatch differs')
 if not p.exists():p.write_text(script)
 result=dict(status='FROZEN_PEER8192_CALIBRATION_SCORER_PREPARED',config=proof(TARGET/'config.json'),sbatch=proof(p),source_publication=proof(TARGET/'source-publication.json'),identities=ids,checkpoint_payloads_loaded=False,feature_payloads_loaded=False,optimizer_constructed=False)
 immutable(TARGET/'preparation.json',result);return result

def evaluate(model,reader,groups,device,destination,progress,name,micro=4):
 import torch
 before=time.monotonic();io=compute=0.;tmp=destination.with_suffix('.tmp');n=0
 with gzip.open(tmp,'wt') as stream,torch.inference_mode():
  for start in range(0,len(groups),micro):
   part=groups[start:start+micro];tick=time.monotonic();batch=reader.get([tid for g in part for tid in g['trajectory_ids']],device)
   torch.cuda.synchronize(device);io+=time.monotonic()-tick;tick=time.monotonic()
   with torch.autocast('cuda',dtype=torch.bfloat16):z=model(batch).reshape(len(part),10)
   logits=z.cpu().tolist();probabilities=z.sigmoid().cpu().tolist()
   for g,zz,pp in zip(part,logits,probabilities):stream.write(json.dumps(score_row(g,zz,pp),allow_nan=False)+'\n')
   torch.cuda.synchronize(device);compute+=time.monotonic()-tick;n+=len(part)
   if n%16==0 or n==len(groups):
    stream.flush();atomic(progress,dict(checkpoint=name,completed_questions=n,expected_questions=len(groups),completed_trajectories=10*n,elapsed_seconds=time.monotonic()-before,feature_io_seconds=io,compute_seconds=compute,actor_updates=0,probe_updates=0,observed_epoch=time.time()))
   del batch,z
 tmp.replace(destination);return dict(questions=n,trajectories=10*n,elapsed_seconds=time.monotonic()-before,feature_io_seconds=io,compute_seconds=compute)

def verify_predictions(path,groups):
 count=0
 with gzip.open(path,'rt') as f:
  for g in groups:
   line=f.readline()
   if not line:raise ValueError('Incomplete remote prediction coverage')
   row=json.loads(line)
   if row!=score_row(g,row['logits'],row['p_j']):raise ValueError('Own IDs/strict EM/probabilities changed in remote score output')
   count+=1
  if f.readline():raise ValueError('Unexpected extra prediction rows')
 return dict(questions=count,trajectories=10*count,all_logits_and_p_finite=True,own_IDs_strict_EM_and_order_exact=True,labels_used_for_score=False)

def run(config_sha):
 if platform.system()!='Linux' or not os.environ.get('SLURM_JOB_ID') or not os.environ.get('CUDA_VISIBLE_DEVICES'):raise ValueError('Only actual owned remote Slurm GPU may read tensor assets')
 cfgref=proof(TARGET/'config.json')
 if cfgref['sha256']!=config_sha:raise ValueError('Config SHA differs')
 c=read(cfgref['path']);pub=read(checked(c['pinned_runtime_publication']));runtime(pub);publication=read(checked(c['source_publication']))
 for name,item in publication['files'].items():checked(dict(path=str(Path(publication['code_root'])/name),**item))
 from reproduction.operations.qwen7b_warmup import peer_newseed_train as fit
 from reproduction.operations.qwen7b_warmup import probe_ablation_train as u
 from reproduction.operations.qwen7b_warmup import evaluate_frozen_validation as legacy
 from reproduction.operations.qwen7b_warmup.model import Probe
 import torch
 if (c['role']!=ROLE or c['actor_updates']!=0 or c['probe_updates']!=0 or c['optimizer_constructed'] is not False or c['evaluation_or_final_scored'] is not False or c['seen_training_data'] is not True or int(os.environ.get('WORLD_SIZE','1'))!=1 or torch.cuda.device_count()!=1 or os.environ.get('FLASH_ATTENTION_DETERMINISTIC')!='1' or os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8'):raise ValueError('Strict frozen one-GPU training-calibration contract required')
 checked(c['source_plan']);checked(c['source_training_complete']);checked(c['model_config']);checked(c['source_training_config'])
 groups,manifests,accepted=fit.accepted_peer_g10(c['training_input'])
 if read(checked(c['groups']))!=[{k:g[k] for k in ('qid','question_hash','trajectory_ids','labels')} for g in groups]:raise ValueError('Calibration truth group projection changed')
 if identities(groups)!=c['identities']:raise ValueError('Frozen8192/G10 identities changed')
 os.environ['QWEN7B_QUESTION_MAX_TOKENS']='273';os.environ['SEARCH_G1_B_PREFIX_TOKENS']='20';device=torch.device('cuda',0);torch.cuda.set_device(device)
 if 'H100' not in torch.cuda.get_device_name(device):raise ValueError('Owned H100 required')
 torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.benchmark=False
 atomic(TARGET/'controller.json',dict(pid=os.getpid(),job_id=os.environ['SLURM_JOB_ID'],config=cfgref,status='FROZEN_G10_SCORING_SOURCE_ACCEPTED',identities=c['identities'],optimizer_constructed=False,time=time.time()))
 model=Probe(c['model_dir'],273,20).to(device).eval().requires_grad_(False);reader=fit.replay_reader(manifests);results={}
 try:
  wanted={t:g['qid'] for g in groups for t in g['trajectory_ids']}
  if set(reader.records)!=set(wanted) or any(reader.records[t][1]['qid']!=q for t,q in wanted.items()):raise ValueError('Actual reader contains other/absent training identities')
  for descriptor in c['checkpoints']:
   name=descriptor['name'];out=TARGET/name;out.mkdir(exist_ok=True)
   if (out/'complete.json').exists():
    prior=read(out/'complete.json')
    if prior['checkpoint']!=descriptor or prior['config']!=cfgref:raise ValueError('Existing score binding differs')
    verify_predictions(checked(prior['predictions']),groups);results[name]=prior;continue
   state=torch.load(checked(descriptor),map_location='cpu',mmap=True,weights_only=False);receipt=read(checked(descriptor['source_receipt']));source_config=read(checked(descriptor['source_config']))
   if name=='E4':legacy.validate_checkpoint_metadata(descriptor,receipt,source_config,state)
   elif (name!='peer256' or state.get('role')!='ISOLATED_REAL_G10_QUESTION_WEIGHTED_PROBE' or state['cursor']!=256 or state['steps']!=10720 or state['actor_updates']!=0 or state['config_sha256']!=c['source_training_config']['sha256'] or state['accepted_hashes']!=descriptor['accepted_hashes'] or receipt['sha256']!=descriptor['sha256'] or receipt['steps']!=10720 or state['matched_binding']['g10_complete']!=c['training_input']):raise ValueError('Actual completed peer fullstate lineage differs')
   ema=u.model_mapping_hash(state['ema'])
   if name=='E4':
    if any(s['ema_sha256']!=ema for s in state['rank_states']):raise ValueError('E4 four-rank EMA hashes differ')
   elif state['accepted_hashes']['ema']!=ema:raise ValueError('Peer accepted EMA hash differs')
   native=state['ema_buffers'];model.load_state_dict(state['ema'],strict=True);u.restore_buffers(model,native);before=u.model_hash(model);buffers_before=u.tensor_hash(u.buffers(model))
   if before!=ema:raise ValueError('Actually loaded EMA differs')
   del state
   timing=evaluate(model,reader,groups,device,out/'predictions.jsonl.gz',TARGET/'progress.json',name,c['micro_questions']);verification=verify_predictions(out/'predictions.jsonl.gz',groups)
   if u.model_hash(model)!=before or any(p.requires_grad or p.grad is not None for p in model.parameters()):raise ValueError('Frozen inference modified parameter state or produced gradients')
   u.restore_buffers(model,native)
   if u.tensor_hash(u.buffers(model))!=buffers_before:raise ValueError('Native buffers failed exact restoration')
   if proof(descriptor['path'])!={k:descriptor[k] for k in ('path','bytes','sha256')}:raise ValueError('Source checkpoint file changed during inference')
   report=dict(status='ONE_FROZEN_TRAINING_G10_CALIBRATION_MODEL_COMPLETE',checkpoint=descriptor,config=cfgref,predictions=proof(out/'predictions.jsonl.gz'),ema_sha256=ema,ema_buffers_sha256=buffers_before,timing=timing,verification=verification,identities=c['identities'],role=ROLE,optimizer_constructed=False,actor_updates=0,probe_updates=0,parameters_unchanged=True,evaluation_or_final_scored=False,seen_training_data=True,unseen_performance_claim=False)
   immutable(out/'complete.json',report);results[name]=report;print(json.dumps(dict(model=name,status=report['status'],questions=8192,elapsed_seconds=timing['elapsed_seconds'])),flush=True)
  result=dict(status=STATUS,config=cfgref,checkpoints=results,groups=c['groups'],identities=c['identities'],role=ROLE,actor_updates=0,probe_updates=0,optimizer_constructed=False,evaluation_or_final_scored=False,normal_exit=True,peer_training_updates_before=256,peer_training_updates_after=256,finished_epoch=time.time())
  immutable(TARGET/'complete.json',result);return result
 finally:reader.close()

def main():
 p=argparse.ArgumentParser();p.add_argument('--action',choices=('prepare','run'),required=True);p.add_argument('--config-sha256');a=p.parse_args();TARGET.mkdir(parents=True,exist_ok=True)
 with (TARGET/'scoring.lock').open('a+') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  try:
   result=prepare() if a.action=='prepare' else run(a.config_sha256);print(json.dumps(result if a.action=='prepare' else dict(status=result['status'],identities=result['identities'])),flush=True)
  except BaseException as error:atomic(TARGET/'failure.json',dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc(),time=time.time(),job_id=os.environ.get('SLURM_JOB_ID')));raise
if __name__=='__main__':main()
