"""Resumable full51713 four-replica evaluation of preserved outer200 HF weights."""
import os,sys,pathlib,json,time,hashlib,subprocess,fcntl
BASE=pathlib.Path('/data/user/jxu957/experiments/topd-20261008');ROOT=BASE/'runtime'
SOURCE=ROOT/'tools/topd-instruct300-source-v1';sys.path.insert(0,str(SOURCE))
MODEL=ROOT/'logs/topd-hpc3-713717/evidence/checkpoints/actor/global_step_200'
OUT=ROOT/'logs/topd-alpha1-final51713-recovery-713717'
N=51713
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def atomic(p,d):
 t=p.with_suffix('.tmp')
 with open(t,'w') as f:json.dump(d,f,sort_keys=True);f.flush();os.fsync(f.fileno())
 t.replace(p)
 fd=os.open(p.parent,os.O_RDONLY)
 try:os.fsync(fd)
 finally:os.close(fd)
def worker(rank):
 import torch
 from transformers import AutoTokenizer
 from vllm import LLM,SamplingParams
 from verl import DataProto
 from verl.utils.dataset.rl_dataset import RLHFDataset,collate_fn
 from torch.utils.data import DataLoader,Subset
 from search_r1.llm_agent.generation import LLMGenerationManager,GenerationConfig
 from search_r1.baselines.evaluation import batch_answer_metrics,aggregate_answer_metrics
 from verl.utils.torch_functional import get_eos_mask
 torch.manual_seed(42)
 tok=AutoTokenizer.from_pretrained(str(MODEL),local_files_only=True);assert tok.padding_side=='right'
 engine=LLM(model=str(MODEL),tokenizer=str(MODEL),dtype='bfloat16',tensor_parallel_size=1,gpu_memory_utilization=.4,enforce_eager=True,max_model_len=4596,max_num_batched_tokens=8192,max_num_seqs=64,enable_prefix_caching=False,seed=42)
 params=SamplingParams(n=1,best_of=1,top_p=1.,top_k=-1,min_p=0.,temperature=0.,max_tokens=500,ignore_eos=False,detokenize=False)
 class Adapter:
  def generate_sequences(self,p):
   ids=p.batch['input_ids'];att=p.batch['attention_mask'];pos=p.batch['position_ids']
   prompts=[x[m.bool()].tolist() for x,m in zip(ids,att)]
   requests=engine.generate(prompt_token_ids=prompts,sampling_params=params,use_tqdm=False)
   tokens=[list(x.outputs[0].token_ids) for x in requests];assert len(tokens)==len(prompts)
   response=torch.full((len(tokens),500),tok.pad_token_id,dtype=torch.long)
   for i,t in enumerate(tokens):response[i,:len(t)]=torch.tensor(t,dtype=torch.long)
   seq=torch.cat([ids,response],-1);rp=pos[:,-1:]+torch.arange(1,501).unsqueeze(0)
   mask=get_eos_mask(response_id=response,eos_token=p.meta_info['eos_token_id'],dtype=att.dtype)
   return DataProto.from_dict({'prompts':ids,'responses':response,'input_ids':seq,'attention_mask':torch.cat([att,mask],-1),'position_ids':torch.cat([pos,rp],-1)})
 dataset=RLHFDataset(parquet_files=str(ROOT/'assets/common-instruct-0b002b/final.parquet'),tokenizer=tok,prompt_key='prompt',max_prompt_length=4096,filter_prompts=True,return_raw_chat=False,truncation='error');assert len(dataset)==N
 indexes=list(range(rank*N//4,(rank+1)*N//4))
 manager=LLMGenerationManager(tok,Adapter(),GenerationConfig(max_turns=4,max_start_length=2048,max_prompt_length=4096,max_response_length=500,max_obs_length=500,num_gpus=1,search_url='http://127.0.0.1:18086/retrieve',topk=3,force_final_answer=True),is_validation=True)
 rows=[];started=time.time()
 previous=sorted(OUT.glob('rank%d-batch-*.json'%rank))
 for batch_i,p in enumerate(previous):
  assert p.name=='rank%d-batch-%04d.json'%(rank,batch_i)
  record=json.loads(p.read_text());saved=record['rows'];assert hashlib.sha256(json.dumps(saved,sort_keys=True).encode()).hexdigest()==record['sha256'];assert [x['row_id'] for x in saved]==indexes[batch_i*32:batch_i*32+len(saved)]
  assert len(saved)==min(32,len(indexes)-batch_i*32)
  rows.extend(saved)
 offset=len(rows);assert offset%32==0 or offset==len(indexes)
 for batch_i,bd in enumerate(DataLoader(Subset(dataset,indexes[offset:]),batch_size=32,shuffle=False,drop_last=False,collate_fn=collate_fn)):
  batch_i+=offset//32
  batch=DataProto.from_single_dict(bd);gen=batch.pop(batch_keys=['input_ids','attention_mask','position_ids']);gen.meta_info={'eos_token_id':tok.eos_token_id,'pad_token_id':tok.pad_token_id,'recompute_log_prob':False,'do_sample':False,'validate':True}
  manager.timing_raw={};result=manager.run_llm_loop(gen_batch=gen,initial_input_ids=gen.batch['input_ids'][:,-2048:].clone());batch=batch.union(result)
  from verl.trainer.main_ppo import decode_model_response_spans
  if batch_i==0:
   width=batch.batch['responses'].shape[1];m=batch.batch['attention_mask'][0,-width:].bool() & batch.batch['info_mask'][0,-width:].bool()
   atomic(OUT/('rank%d-first-response-audit.json'%rank),{'padding_side':tok.padding_side,'model_spans':decode_model_response_spans(tok,batch.batch['responses'][0],m)})
  scored=batch_answer_metrics(batch,tok)
  for i,row in enumerate(scored):
   row['row_id']=indexes[batch_i*32+i]
   from search_r1.baselines.bapo import answer_from_spans
   width=batch.batch['responses'].shape[1]
   model_mask=batch.batch['attention_mask'][i,-width:].bool() & batch.batch['info_mask'][i,-width:].bool()
   row['prediction']=answer_from_spans(decode_model_response_spans(tok,batch.batch['responses'][i],model_mask))
  atomic(OUT/('rank%d-batch-%04d.json'%(rank,batch_i)),{'rows':scored,'sha256':hashlib.sha256(json.dumps(scored,sort_keys=True).encode()).hexdigest()})
  rows.extend(scored);atomic(OUT/('rank%d-progress.json'%rank),{'rank':rank,'rows':len(rows),'epoch':time.time(),'seconds':time.time()-started,'metrics':aggregate_answer_metrics(rows)})
  print('RANK_PROGRESS',rank,len(rows),flush=True)
 atomic(OUT/('rank%d-rows.json'%rank),rows)
 atomic(OUT/('rank%d-complete.json'%rank),{'rank':rank,'rows':len(rows),'sha256':sha(OUT/('rank%d-rows.json'%rank)),'seconds':time.time()-started})
def audit_saved_checkpoints():
 import torch,gc
 from safetensors import safe_open
 reports=[]
 for outer in [5,50,100,150,200]:
  directory=(ROOT/'logs/topd-hpc3-712204/evidence/checkpoints/actor/global_step_5') if outer==5 else MODEL.parent/('global_step_%d'%outer)
  print('CHECKPOINT_AUDIT_STARTED',outer,flush=True)
  receipt=json.loads((directory/'resume-complete.json').read_text());assert receipt['outer_step']==outer
  rank_reports=[]
  for rank,item in enumerate(receipt['ranks']):
   p=pathlib.Path(item['path']);assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256']
   print('CHECKPOINT_RANK_SHA_PASS',outer,rank,flush=True)
   state=torch.load(p,map_location='cpu',weights_only=False)
   assert state['rank']==rank and state['world_size']==4 and state['outer_step']==outer
   assert state['scheduler']['last_epoch']==outer and len(state['resource_updates'])==outer
   assert all(len(u['optimizer_step_records'])==20 and all(x['finite'] for x in u['optimizer_step_records']) for u in state['resource_updates'])
   assert state['optimizer']['state'] and state['model_local']
   for k in ['python_rng','numpy_rng','torch_rng','cuda_rng']:assert k in state
   for value in state['model_local'].values():assert torch.isfinite(value).all().item()
   for slot in state['optimizer']['state'].values():
    assert int(slot['step'])==20*outer
    for k in ['exp_avg','exp_avg_sq']:assert torch.isfinite(slot[k]).all().item()
   rank_reports.append({'rank':rank,'sha256':item['sha256'],'optimizer_steps':20*outer,'scheduler_outer':outer,'state_structure_and_finiteness_pass':True})
   del state;gc.collect()
  item=receipt['driver'];p=pathlib.Path(item['path']);assert p.stat().st_size==item['bytes'] and sha(p)==item['sha256']
  state=torch.load(p,map_location='cpu',weights_only=False)
  assert state['outer_step']==outer and state['next_outer_step']==outer+1
  assert sorted(state['data_order'])==list(range(169028)) and state['next_batch']==outer
  del state;gc.collect()
  hf={}
  for p in directory.glob('*.safetensors'):
   count=0
   with safe_open(str(p),framework='pt',device='cpu') as f:
    for key in f.keys():
     value=f.get_tensor(key);assert torch.isfinite(value).all().item();count+=1;del value
   hf[p.name]={'sha256':sha(p),'finite_tensors':count}
  assert hf
  report={'outer':outer,'ranks':rank_reports,'driver_sha256':item['sha256'],'hf':hf,'fresh_checkpoint_read_pass':True,'new_process_GPU_next_step_resume_test':False}
  atomic(OUT/('checkpoint-%d-audit.json'%outer),report);reports.append(report)
 atomic(OUT/'all-scheduled-checkpoints-audit.json',{'steps':[5,50,100,150,200],'reports':reports,'epoch':time.time()})
def main():
 OUT.mkdir(parents=True,exist_ok=True);lock=open(OUT/'eval.lock','w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 audit_saved_checkpoints()
 receipt=json.loads((MODEL/'resume-complete.json').read_text());assert receipt['outer_step']==200
 verified=[]
 for r in receipt['ranks']+[receipt['driver']]:
  p=pathlib.Path(r['path']);assert p.stat().st_size==r['bytes'] and sha(p)==r['sha256'];verified.append({'path':str(p),'sha256':r['sha256']})
 fresh={'verified':verified,'epoch':time.time(),'hf_files':{p.name:sha(p) for p in MODEL.iterdir() if p.suffix in ['.safetensors','.json'] and not p.name.startswith('resume')}}
 data=ROOT/'assets/common-instruct-0b002b/final.parquet'
 manifest=ROOT/'assets/common-instruct-0b002b/manifest.json'
 assert sha(manifest)=='0b002b0038d58d0ce07c073e41a8be785ce47eb4f15a1ee883fe19cc86dbcf24'
 binding={'model_files':fresh['hf_files'],'final_parquet_sha256':sha(data),'source_sha256':sha(pathlib.Path(__file__)),'rows':N,'training_job':713717,'alpha':1,'checkpoint_step':200,'method':'four-vllm-TP1-bf16-greedy-right-padding-CPU-FAISS-GPU-E5'}
 if (OUT/'binding.json').exists():assert json.loads((OUT/'binding.json').read_text())==binding,'Resume binding drift'
 else:atomic(OUT/'binding.json',binding)
 atomic(OUT/('checkpoint200-fresh-sha-'+os.environ['SLURM_JOB_ID']+'.json'),fresh)
 c=json.loads((ROOT/'logs/topd-hpc3-713717/config.json').read_text());cmd=c['retriever_command'].copy();cmd.remove('--faiss_gpu');cmd[cmd.index('--port')+1]='18086';cmd[cmd.index('--threads')+1]='16'
 env=os.environ.copy();visible=env['CUDA_VISIBLE_DEVICES'].split(',');assert len(visible)==4
 env['CUDA_VISIBLE_DEVICES']=visible[3]
 retriever=subprocess.Popen(cmd,env=env,stdout=open(OUT/'retriever.log','a'),stderr=subprocess.STDOUT)
 children=[]
 try:
  import requests
  for _ in range(1200):
   if retriever.poll() is not None:raise RuntimeError('Standalone retriever exited')
   try:
    if requests.get('http://127.0.0.1:18086/docs',timeout=2).status_code==200:break
   except requests.RequestException:pass
   time.sleep(1)
  else:raise TimeoutError('Retriever startup')
  for rank in range(4):
   e=os.environ.copy();e['CUDA_VISIBLE_DEVICES']=visible[rank];e['TRITON_CACHE_DIR']=str(OUT/('cache/triton%d'%rank));e['TMPDIR']=str(OUT/('cache/tmp%d'%rank));pathlib.Path(e['TMPDIR']).mkdir(parents=True,exist_ok=True)
   children.append(subprocess.Popen([sys.executable,'-B',__file__,'--rank',str(rank)],env=e,stdout=open(OUT/('rank%d.log'%rank),'a'),stderr=subprocess.STDOUT))
  while any(p.poll() is None for p in children):
   for p in children:
    if p.poll() not in (None,0):raise RuntimeError('Evaluation worker exited code '+str(p.returncode))
   time.sleep(5)
  assert all(p.returncode==0 for p in children)
  rows=[]
  for rank in range(4): rows.extend(json.loads((OUT/('rank%d-rows.json'%rank)).read_text()))
  assert len(rows)==N and sorted(r['row_id'] for r in rows)==list(range(N))
  from search_r1.baselines.evaluation import aggregate_answer_metrics
  atomic(OUT/'complete.json',{'status':'FINAL51713_ALL_FOUR_GPU_WORKERS_VERIFIED','checkpoint_step':200,'training_job':713717,'alpha':1,'evaluation_job':int(os.environ['SLURM_JOB_ID']),'rows':N,'metrics':aggregate_answer_metrics(rows),'rank_receipts':[json.loads((OUT/('rank%d-complete.json'%r)).read_text()) for r in range(4)],'inference':'fixed-HF-four-independent-vllm-TP1-eager','retrieval':'same-E5-fullWiki-CPU-FAISS-GPU-E5','epoch':time.time()})
 finally:
  for p in children+[retriever]:
   if p.poll() is None:p.terminate()
  for p in children+[retriever]:
   try:p.wait(timeout=30)
   except subprocess.TimeoutExpired:p.kill();p.wait()
  atomic(OUT/('owned-cleanup-'+os.environ['SLURM_JOB_ID']+'.json'),{'all_spawned_children_reaped':all(p.poll() is not None for p in children+[retriever]),'only_exact_spawned_process_handles':True,'job':os.environ['SLURM_JOB_ID'],'epoch':time.time()})
if __name__=='__main__':
 if '--rank' in sys.argv:worker(int(sys.argv[sys.argv.index('--rank')+1]))
 else:main()
