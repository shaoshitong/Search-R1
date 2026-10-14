# -*- coding: utf-8 -*-
"""Read original immutable TRAIN200, reuse its still-loaded frozen probe for detached inference.
No trainer/config/manifest changes; raw records stay on the source node."""
from pathlib import Path
import os,sys,json,hashlib,datetime,importlib.util,heapq,math,time
os.environ['CUDA_VISIBLE_DEVICES']='7';os.environ['TOKENIZERS_PARALLELISM']='false';os.environ['OMP_NUM_THREADS']='1'
STEP=int(sys.argv[1]);assert STEP in [100,200]
D=Path(__file__).resolve().parent;r=Path('/mnt/data/shaoshitong/search-g1-speed-resume5-0553ade-20261012');repo=r/'repo';sys.path.insert(0,str(repo))
def publish(n,v):(D/n).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False))
def status(stage,**kw):publish('status.json',{'stage':stage,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw})
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
 return h.hexdigest()
def module(n,p):
 s=importlib.util.spec_from_file_location(n,str(p));m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
try:
 status('CHECK_INPUT_BINDINGS');run=json.load((r/'logs/run-manifest.json').open());cfg=json.load((r/'logs/resolved-training-config.json').open())['resolved_config'];updates=[json.loads(l) for l in (r/'logs/actual-updates.jsonl').open()];u=next(v for v in updates if v['step']==STEP);assert updates[-1]['step']==200 and u['run_id']==run['run_id'];aid=u['attempt_id']
 gatep=r/'logs'/('behavior-probability-attempt-'+aid+'.json');assert sha(gatep)==u['behavior_probability_gate_sha256'];gate=json.load(gatep.open());assert gate['accepted'] is True and gate['trajectory_count']==2560
 sp=r/'artifacts/mixed-selection'/(aid+'.json');selection=json.load(sp.open());assert selection['step']==STEP and selection['selected_groups']==512
 ch=module('C_stream',D/'C_stream.py');core=module('replay_core',D/'replay_core.py');stats=module('aggregator',D/'stats.py')
 for rel in ['search_g1/signals.py','search_r1/qa_protocol.py','verl/utils/reward_score/qa_em.py','verl/trainer/main_ppo_eco.py','probe_server.py','verl/trainer/ppo/ray_trainer.py','verl/utils/seqlen_balancing.py']:assert sha(repo/rel)==run['source_sha256'][rel]
 signals=ch.load_pure_module('signals_bound',repo/'search_g1/signals.py');qa=ch.load_pure_module('qa_bound',repo/'verl/utils/reward_score/qa_em.py')
 expected=u['probe_version'];manifests=[json.load(p.open()) for p in (r/'artifacts').glob('round_*-attempt-*/version-manifest.json')];manifest=next(v for v in manifests if v['version']==expected);assert manifest['round']==STEP-50
 for role in ['commit','boundary']:assert sha(manifest[role])==manifest['head_sha256'][role+'_probe.pkl']
 status('CHECK_FROZEN_ENCODER',step=STEP)
 for name,h in manifest['encoder_sha256'].items():assert sha(Path(manifest['encoder'])/name)==h
 selected=[g for g in selection['groups'] if g['selected']];byid={tid:(g,j) for g in selected for j,tid in enumerate(g['trajectory_ids'])};assert len(byid)==2560 and set(byid)==set(gate['trajectory_ids'])
 import pyarrow.parquet as pq
 wanted=set(gate['upstream_group_sha256'].values());mapping={};train=cfg['data']['train_files'];train=[train] if isinstance(train,str) else train
 for path in train:
  assert sha(path)==run['input_sha256'][Path(path).name]
  for b in pq.ParquetFile(path).iter_batches(batch_size=256,columns=['extra_info','reward_model']):
   m,_=ch.gold_mapping(b.to_pylist())
   for k,v in m.items():
    if k in wanted:mapping.setdefault(k,{}).update(v)
 from tokenizers import Tokenizer
 class Decoder:
  def __init__(self):self.t=Tokenizer.from_file(str(Path(run['initial_model'])/'tokenizer.json'))
  def decode(self,ids):return self.t.decode(ids,skip_special_tokens=False)
 for n,h in run['initial_model_sha256'].items():
  if n.endswith(('.json','.txt','.model')):assert sha(Path(run['initial_model'])/n)==h
 assert json.load((Path(run['initial_model'])/'tokenizer_config.json').open()).get('clean_up_tokenization_spaces',False)==False
 tok=Decoder();trace=r/'artifacts/behavior-traces'/('attempt-'+aid+'.json');ch.stream_trace_sha(trace,u['behavior_trace_sha256']);items=[];C={};lengths=[];digest=hashlib.sha256(b'[')
 status('DECODE_ORIGINAL_ROWS')
 for i,row in enumerate(ch.stream_actor_rows(trace)):
  ch.verify_streamed_actor(row,i,gate)
  if i:digest.update(b',')
  digest.update(ch.canonical(row));tid=row['trajectory_id'];c=ch.reconstruct_row(row,ch.unique_gold(mapping,gate['upstream_group_sha256'][tid]),tok,signals,qa);g,j=byid[tid];assert c['C_raw']==int(g['strict_correctness'][j])
  prompt_ids,response_ids,mask=ch.actor_parts(row);prompt=tok.decode(prompt_ids);response,ranges=signals.decode_segments(tok,response_ids,mask);items.append({'id':tid,'prompt':prompt,'response':response,'model_ranges':ranges});C[tid]=c;lengths.append(sum(row['attention_mask']))
  if (i+1)%256==0:status('DECODE_ORIGINAL_ROWS',rows=i+1)
 digest.update(b']');ch.verify_actor_array_digest(digest.hexdigest(),len(items),gate)
 balance=core.load_balance(repo/'verl/utils/seqlen_balancing.py',run['source_sha256']['verl/utils/seqlen_balancing.py']);world=run['training_world_size'];assert world==8
 parts=balance(lengths,k_partitions=world,equal_size=True);order=[j for part in parts for j in part];assert sorted(order)==list(range(2560));balanced=[items[i] for i in order]
 constants={'PENALTY':.1,'MAX_SEARCH':2,'wrong_floor':.05,'bal_ground':.6,'bal_nosearch_bonus':.5,'w_cost':.02,'wrong_ground_scale':0.,'probe_base':0.,'probe_gate':True,'probe_alpha_cost':False};score=core.load_reward(repo/'verl/trainer/main_ppo_eco.py',run['source_sha256']['verl/trainer/main_ppo_eco.py'],constants)
 assert cfg['actor_rollout_ref']['actor'].get('use_kl_loss',False) or cfg['algorithm']['kl_ctrl']['kl_coef']==0
 os.environ['CUDA_VISIBLE_DEVICES']='7'
 import subprocess,numpy as np,torch,joblib,threading
 from typing import List
 from transformers import AutoModelForCausalLM,AutoTokenizer
 GPU='GPU-cef6b05a-4e86-cab0-b75e-c417c30cf67e';own=os.getpid()
 def resource_guard():
  q=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],text=True,capture_output=True,check=True).stdout
  foreign=[line for line in q.splitlines() if GPU in line and int(line.rsplit(',',1)[1])!=own]
  assert not foreign,('GPU became occupied; stop own replay',foreign)
  free=float(subprocess.run(['nvidia-smi','--id='+GPU,'--query-gpu=memory.free','--format=csv,noheader,nounits'],text=True,capture_output=True,check=True).stdout.strip())
  assert free>16000,('GPU free memory guard',free)
  if torch.cuda.is_initialized():assert torch.cuda.memory_reserved()<24*1024**3,'Own replay exceeded 24GiB cap'
 resource_guard();assert torch.cuda.device_count()==1
 ns=dict(torch=torch,np=np,joblib=joblib,AutoTokenizer=AutoTokenizer,AutoModelForCausalLM=AutoModelForCausalLM,DTYPE=torch.bfloat16,contexts=signals.contexts,List=List)
 funcs=core.extract_functions(repo/'probe_server.py',['_load','_last_hidden','_commit_ctx'],ns)
 status('LOAD_PRIVATE_FROZEN_PROBE',step=STEP,version=expected);started=time.monotonic()
 tok_gpu,model,commit,boundary=funcs['_load'](manifest['encoder'],manifest['commit'],manifest['boundary']);resource_guard()
 assert list(commit['clf'].classes_)==list(boundary['clf'].classes_)==[0,1]
 for m in model.modules():
  if isinstance(m,torch.nn.Dropout):assert not m.training
 previous=[0.]
 def hook(mod,ins):
  if time.monotonic()-previous[0]>2:
   resource_guard();previous[0]=time.monotonic();status('REPLAY_FROZEN_PROBE',step=STEP,phase=phase[0],micro_batches=counter[0],version=expected)
  counter[0]+=1
 phase=['commit'];counter=[0];guard_hook=model.model.register_forward_pre_hook(hook)
 cc=[funcs['_commit_ctx'](it['prompt'],it['response'],it['model_ranges'],commit['position']) for it in balanced];bc=[it['prompt'] for it in balanced]
 Hc=funcs['_last_hidden'](cc,commit['layer'],model,tok_gpu);phase[0]='boundary';counter[0]=0
 Hb=funcs['_last_hidden'](bc,boundary['layer'],model,tok_gpu);guard_hook.remove();torch.cuda.synchronize()
 flip=commit['clf'].predict_proba(Hc)[:,1];bnd=boundary['clf'].predict_proba(Hb)[:,1];assert np.isfinite(flip).all() and np.isfinite(bnd).all()
 assert ((0<=flip)&(flip<=1)).all() and ((0<=bnd)&(bnd<=1)).all()
 pred={it['id']:dict(version=expected,flip=float(flip[i]),boundary=float(bnd[i])) for i,it in enumerate(balanced)}
 assert len(pred)==2560 and set(pred)==set(C)
 del model;import gc;gc.collect();torch.cuda.empty_cache()
 R={};rows=[]
 for item in items:
  tid=item['id'];c=C[tid];p=pred[tid];assert p['version']==expected;rr=core.scalar_reward(score,c['final_action_valid'],c['C_raw'],c['model_authored_search_count'],float(p['boundary']),float(p['flip']));R[tid]=rr['R_reconstructed'];rows.append({'trajectory_id':tid,'C':c['C_raw'],'R_reconstructed':R[tid],'boundary':p['boundary'],'flip':p['flip']})
 import numpy as np
 calculated=float(np.asarray(list(R.values()),dtype=np.float32).mean());native=float(u['metrics']['critic/rewards/mean']);assert abs(calculated-native)<5e-5,(calculated,native)
 groupsC=[[C[tid]['C_raw'] for tid in g['trajectory_ids']] for g in selected];groupsR=[[R[tid] for tid in g['trajectory_ids']] for g in selected];s=stats.stat(groupsC,groupsR)
 with (D/f'original-TRAIN{STEP}-pairs.jsonl').open('w') as f:
  for row in rows:f.write(json.dumps(row,separators=(',',':'))+'\n')
 result={'name':'Easy Selection+Soft Reward','step':STEP,'run_id':run['run_id'],'probe_version':expected,'native_mean_reward':native,'reconstructed_mean_reward_fp32':calculated,'native_mean_check_pass':True,'replay_seconds':time.monotonic()-started,'trace_sha256':u['behavior_trace_sha256'],'selection_sha256':sha(sp),'actor_rows_sha256':digest.hexdigest(),'pairs_sha256':sha(D/f'original-TRAIN{STEP}-pairs.jsonl'),'per_question_std_correction':1,'online_bitwise_reward_recovery':False,'GPU_model_created':True,'trainer_changed':False,'stats':s}
 publish('aggregate.json',result);status('COMPLETE',groups=512,trajectories=2560)
except Exception as e:
 import traceback
 traceback.print_exc();status('FAILED',error_type=type(e).__name__,reason=str(e)[:500]);raise
