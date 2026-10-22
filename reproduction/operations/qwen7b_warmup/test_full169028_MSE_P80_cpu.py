"""Targeted full sampler, exact2-rank tail mean and epoch-boundary resume tests."""
import copy,json,math,os,pathlib,sys,tempfile,time,types
import torch
from torch import nn
from . import online_packed_full169028_MSE_P80 as m

def sampler_and_resume():
 for rejected in({},dict(fresh_full_P20_prefix_prompt_and_own_EM_exact=False)):
  try:m.full_acceptance_alignment_ref(rejected)
  except ValueError:pass
  else:raise AssertionError('Producer-only acceptance must not start training')
 ref=dict(path='/remote/synthetic-alignment.json',bytes=1,sha256='a'*64);assert m.full_acceptance_alignment_ref(dict(fresh_full_P20_prefix_prompt_and_own_EM_exact=True,P20_alignment=ref))==ref
 groups={str(i):dict(qid=str(i),trajectory_ids=[str(i)+'t'+str(j)for j in range(10)],labels=[j%2 for j in range(10)])for i in range(169028)}
 plan=m.epoch_plan(groups,1,5,train_plan_seed=2026102227);rank={q:i%2 for i,q in enumerate(sorted(groups))};counts=[sum(rank[v['qid']]==r for v in plan)for r in(0,1)];assert counts==[84514,84514]and len({v['qid']for v in plan})==169028
 assert all(v['M']==5 and len(set(v['trajectory_ids']))==5 and set(v['trajectory_ids'])<=set(groups[v['qid']]['trajectory_ids'])for v in plan)
 assert [len(list(range(i,min(i+64,84514))))for i in range(0,84514,64)][-1]==34 and math.ceil(84514/64)==1321
 assert m.actual_local_batch_layout(34)==dict(actual_global_batch=68,local_batch=34,micro=8,gradient_accumulation=5,micro_question_counts=[8,8,8,8,2])
 args=types.SimpleNamespace(init_seed=9137,train_plan_seed=2026102227,epochs=10);blob=dict(binding={},init_seed=9137,train_plan_seed=2026102227,diagnostic_seed=2026102228,initial_head_sha256='head',epoch=1,cursor=1321,batch_attempts=1321,actual_Adam_updates=1321,EMA_updates=1321,margin_satisfied_batches=0,RNG_by_rank=[dict(python=1,numpy=2,torch=3,cuda=4)]*2,epoch_complete=True,next_epoch=2,within_next_epoch_question_position=0,plan_sha256=m.digest(plan),raw={},EMA={},Adam={});m.verify_resume_boundary(blob,{},groups,args,'head')
 for bad in('clock','missing_Adam','wrong_phase'):
  b=copy.deepcopy(blob)
  if bad=='clock':b['actual_Adam_updates']=1320
  elif bad=='missing_Adam':del b['Adam']
  else:b['epoch_complete']=False
  try:m.verify_resume_boundary(b,{},groups,args,'head')
  except ValueError:pass
  else:raise AssertionError('Incomplete resume was accepted: '+bad)
 try:m.prepare(None)
 except RuntimeError:pass
 else:raise AssertionError('Pilot prepare must be disabled')
 return dict(training_questions=len(plan),trajectory_visits=sum(v['M']for v in plan),rank_questions=counts,updates_per_epoch=1321,tail=m.actual_local_batch_layout(34),plan_sha256=m.digest(plan),full_epoch_resume_boundary_passed=True,invalid_Adam_clock_or_phase_rejected=True,pilot_prepare_disabled=True)

def worker(rank,store,out):
 import torch.distributed as dist
 from torch.nn.parallel import DistributedDataParallel as DDP
 torch.set_num_threads(1);os.environ['GLOO_SOCKET_IFNAME']='lo0'if sys.platform=='darwin'else'lo';dist.init_process_group('gloo',init_method='file://'+store,rank=rank,world_size=2);results=[]
 for n in(128,68):
  torch.manual_seed(9137);model=nn.Linear(3,1).double();ddp=DDP(model);x=torch.arange(n*3,dtype=torch.float64).reshape(n,3)/max(1,n)-1;y=torch.tensor([(i%5)/5 for i in range(n)],dtype=torch.float32);local=n//2;data=x[rank*local:(rank+1)*local];target=y[rank*local:(rank+1)*local];total=0.
  for lo in range(0,local,8):loss=m.local_mean_mse(ddp(data[lo:lo+8]).squeeze(-1),target[lo:lo+8],local);loss.backward();total+=float(loss.detach())
  results.append(dict(global_questions=n,local_questions=local,gradients={k:p.grad.tolist()for k,p in model.named_parameters()},loss=total,micro_counts=m.actual_local_batch_layout(local)['micro_question_counts']))
 pathlib.Path(out+str(rank)+'.json').write_text(json.dumps(results));dist.destroy_process_group()

def exact_two_rank_mean():
 import torch.multiprocessing as mp
 with tempfile.TemporaryDirectory(prefix='full-p80-tail-gloo-')as tmp:
  out=str(pathlib.Path(tmp)/'r');mp.spawn(worker,args=(str(pathlib.Path(tmp)/'store'),out),nprocs=2,join=True);actual=[json.loads(pathlib.Path(out+str(r)+'.json').read_text())for r in(0,1)]
 maximum=0.
 for index,n in enumerate((128,68)):
  torch.manual_seed(9137);model=nn.Linear(3,1).double();x=torch.arange(n*3,dtype=torch.float64).reshape(n,3)/max(1,n)-1;y=torch.tensor([(i%5)/5 for i in range(n)],dtype=torch.float32);loss=m.local_mean_mse(model(x).squeeze(-1),y,n);loss.backward()
  for rank in range(2):
   for name,p in model.named_parameters():
    a=torch.tensor(actual[rank][index]['gradients'][name],dtype=torch.float64);error=float((a-p.grad).abs().max());maximum=max(maximum,error);assert torch.allclose(a,p.grad,atol=2e-8,rtol=2e-6),(n,name,error)
 return dict(backend='gloo',world_size=2,actual_global_batches=[128,68],local_batches=[64,34],tail_micro_counts=[8,8,8,8,2],DDP_equal_local_means_match_true_global_mean=True,maximum_absolute_gradient_error=maximum)

def main():
 start=time.time();torch.set_num_threads(1);result=dict(status='FULL169028_SAMPLER_RESUME_BOUNDARY_AND_REAL_TWO_RANK_TAIL_MEAN_CPU_PASS',sampler=sampler_and_resume(),distributed=exact_two_rank_mean(),seconds=time.time()-start,no_GPU=True,no_Slurm=True,no_real_model_checkpoint_or_rollout_read=True,final4096_read=False);print(json.dumps(result))
if __name__=='__main__':main()
