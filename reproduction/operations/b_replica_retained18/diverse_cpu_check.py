"""Focused CPU checks for ten-to-five provenance and exact 4->8 Adam split."""
import argparse,json,tempfile
from pathlib import Path

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--base',required=True);a=parser.parse_args()
 import torch
 from reproduction.prefix_selection200.launch import config,_require_training_assignment
 from reproduction.prefix_selection200.prefix_diversity import select
 from reproduction.prefix_selection200.diverse_resume import split_optimizer,fork_rng,flat_numels
 from reproduction.prefix_selection200.actor import ActorBudget,rank_plan
 from reproduction.prefix_selection200.actor_hidden import NativeFeatureWriter
 from reproduction.prefix_selection200.contract import WORLD,WIDTH,DIM
 c=config(a.base);_require_training_assignment(c);budget=ActorBudget(c['actor_global_micro_batch_size'])
 assert WORLD==8 and budget.local_micro==1 and budget.accumulation==16
 from reproduction.prefix_selection200.actor_safety import frozen_monitor_ids
 plan=rank_plan(1729,51);clear_rows=[dict(response_info_mask=[1]*(i%37+1))for i in range(2560)]
 eight=[frozen_monitor_ids(rank,clear_rows)for rank in plan]
 four=[frozen_monitor_ids([a+b for a,b in zip(plan[2*r],plan[2*r+1])],clear_rows)for r in range(4)]
 assert [i for r in four for i in r]==[i for r in eight for i in r] and len(set(i for r in eight for i in r))==32
 assert len(plan)==8 and all(len(r)==20 for r in plan)
 assert sorted(i for r in plan for b in r for i in b)==list(range(2560))
 for unique in (1,3,5,10):
  rows=[dict(policy_token_ids=[i%unique,8,9],trajectory_id='q:g%d'%i,response_index=i,prefix_sha256=str(i))for i in range(10)]
  selected,proof=select(rows);assert len(selected)==5 and len(set(selected))==5 and proof['unique_retained_prefixes']==min(5,unique)
  assert select(rows)==(selected,proof) and not proof['probe_outputs_used'] and not proof['outcome_labels_used']
 optimizer=dict(param_groups=[dict(params=list(range(37)),lr=1e-6)],state={})
 layout=[[]]
 for i in range(37):
  optimizer['state'][i]=dict(step=torch.tensor(1000.),exp_avg=torch.arange(6,dtype=torch.float32)+i,exp_avg_sq=torch.arange(6,dtype=torch.float32)+2*i)
  layout[0].append(dict(shape=[6],numel=6,dtype='torch.float32'))
 saved=dict(optimizer=optimizer,optimizer_parameter_layout=layout)
 left,_,_=split_optimizer(saved,0,[24]*37);right,_,_=split_optimizer(saved,1,[24]*37)
 for i in range(37):
  for name in ('exp_avg','exp_avg_sq'):assert torch.equal(torch.cat([left['state'][i][name],right['state'][i][name]]),optimizer['state'][i][name])
 import random,numpy as np
 rng=dict(python=random.getstate(),numpy=np.random.get_state(),torch_cpu=torch.Generator().manual_seed(23).get_state(),torch_cuda=torch.zeros(16,dtype=torch.uint8))
 assert torch.equal(fork_rng(rng,0)['torch_cpu'],rng['torch_cpu'])
 assert not torch.equal(fork_rng(rng,1)['torch_cpu'],rng['torch_cpu'])
 with tempfile.TemporaryDirectory(dir=Path(a.base)/'preparation')as directory:
  from safetensors.torch import save_file
  model=Path(directory)/'tiny-hf';model.mkdir();(model/'config.json').write_text(json.dumps(dict(tie_word_embeddings=True)))
  weights={'model.embed_tokens.weight':torch.ones(8),'model.norm.weight':torch.ones(16),'lm_head.weight':torch.ones(8)}
  weights.update({'model.layers.%d.weight'%i:torch.ones(24)for i in range(36)})
  save_file(weights,str(model/'model.safetensors'));assert flat_numels(model)==[24]*37
  writer=NativeFeatureWriter(directory,0,['g%d'%i for i in range(10)],selected_capacity=5)
  for i in [9,2,5,7,0]:
   result=writer.write(dict(trajectory_id='g%d'%i,actor_version='CP50'),dict(features=torch.full((WIDTH,DIM),float(i),dtype=torch.bfloat16)))
   assert result['feature_row']==len(writer.written)-1
  path=writer.finish();array=np.load(path,mmap_mode='r');assert array.shape==(5,WIDTH,DIM)
 assert not torch.cuda.is_initialized()
 print(json.dumps(dict(status='DIVERSE_CP50_FOCUSED_CPU_PASS',world=8,local_micro=1,accumulation=16,exact_moment_concat=True,distinct_prefixes=True,native_selected_rows=5,cuda_initialized=False)))
if __name__=='__main__':main()
