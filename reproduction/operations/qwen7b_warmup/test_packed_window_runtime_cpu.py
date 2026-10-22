"""Targeted tiny CPU integration: detached cache, zero-stop, exact 2-rank math."""
import copy,json,os,pathlib,sys,tempfile,time
import torch
from torch import nn
from .detached_score_window import DetachedScoreWindow
from .group_topk_losses import precision_at_k_ramp

def entries(scores,labels,*,prefix,step,attempt):
 return [dict(qid=prefix+str(i),trajectory_ids=[prefix+str(i)+'-t'+str(j)for j in range(5)],labels=[1,0,0,0,0]if y else[0,0,0,0,0],true_mixed=bool(y),M=5,cached_score=float(s),scored_step=step,scored_batch_attempt=attempt,score_model='raw')for i,(s,y)in enumerate(zip(scores,labels))]

def full_history(negative_score=0.):
 labels=[i%2 for i in range(1954)];scores=[0. if y else negative_score for y in labels];queue=DetachedScoreWindow(2018,500,2018);queue.prepare(entries(scores,labels,prefix='h',step=0,attempt=1),model_version=0);queue.commit(post_update_step=0);return queue

def local_checks():
 labels=[i%2 for i in range(64)];current=nn.Parameter(torch.tensor([-.5 if y else .5 for y in labels],dtype=torch.float64));optimizer=torch.optim.Adam([current],lr=5e-6);ema=current.detach().clone();q=full_history();window=q.prepare(entries(current.detach(),labels,prefix='c',step=0,attempt=2),model_version=0);history=torch.tensor(window['cached_scores'],dtype=torch.float64);scores=torch.cat((history,current));target=torch.tensor([int(r['true_mixed'])for r in window['rows']],dtype=torch.float64);loss=precision_at_k_ramp(scores,target,window['K']);scaled=loss*window['N_eff']/64;scaled.backward();active_gradient_L2=float(current.grad.norm());assert history.requires_grad is False and history.grad is None and current.grad is not None and float(current.grad.norm())>0;optimizer.step();ema.mul_(.9995).add_(current.detach(),alpha=.0005);q.commit(post_update_step=1);state=q.state_dict();restored=DetachedScoreWindow.from_state_dict(state);assert restored.state_dict()==state and restored.ring_sha256==q.ring_sha256 and state['pending']is None
 # Positive optimizer moments exist from the first actual update. A zero-loss
 # branch must not even call Adam.step or EMA, so their state remains bit exact.
 with torch.no_grad():current.copy_(torch.tensor([4. if y else-4. for y in labels],dtype=torch.float64))
 optimizer.zero_grad(set_to_none=True);zero=precision_at_k_ramp(current,torch.tensor(labels,dtype=torch.float64),16);zero.backward();assert float(zero.detach())==0. and float(current.grad.norm())==0.;before=current.detach().clone();old_ema=ema.clone();old_adam=copy.deepcopy(optimizer.state_dict());update=float(zero.detach())!=0. and float(current.grad.norm())!=0.
 if update:raise AssertionError('Satisfied margin must skip Adam and EMA')
 assert torch.equal(current,before)and torch.equal(ema,old_ema)
 for key,value in old_adam['state'][0].items():assert torch.equal(optimizer.state_dict()['state'][0][key],value)
 # A positive loss supported solely by historical constraints must also skip.
 q2=full_history(negative_score=2.);x=torch.full((64,),-100.,dtype=torch.float64,requires_grad=True);w=q2.prepare(entries(x.detach(),labels,prefix='z',step=0,attempt=2),model_version=0);stale=torch.tensor(w['cached_scores'],dtype=torch.float64);l=precision_at_k_ramp(torch.cat((stale,x)),torch.tensor([int(r['true_mixed'])for r in w['rows']],dtype=torch.float64),w['K']);l.backward();assert float(l.detach())>0. and float(x.grad.norm())==0.;q2.commit(post_update_step=0)
 return dict(full_window_N=window['N_eff'],K=window['K'],history_rows=len(history),current_rows=64,historical_requires_grad=history.requires_grad,current_gradient_L2=active_gradient_L2,queue_JSON_roundtrip_exact=True,zero_margin_skips_Adam_and_EMA_preserving_moments=True,cached_only_positive_loss=float(l.detach()),cached_only_current_gradient_L2=float(x.grad.norm()),cached_only_skips_Adam_and_EMA=True)

def worker(rank,store,out):
 import torch.distributed as dist
 from torch.distributed.nn.functional import all_gather
 from torch.nn.parallel import DistributedDataParallel as DDP
 torch.set_num_threads(1);os.environ['GLOO_SOCKET_IFNAME']='lo0'if sys.platform=='darwin'else'lo';dist.init_process_group('gloo',init_method='file://'+store,rank=rank,world_size=2);torch.manual_seed(9137);model=nn.Linear(3,1).double();ddp=DDP(model);features=torch.arange(64*3,dtype=torch.float64).reshape(64,3)/97-1;labels=torch.tensor([i%2 for i in range(64)],dtype=torch.float64);part=features[rank*32:(rank+1)*32];micro=[ddp(part[i:i+8]).squeeze(-1)for i in range(0,32,8)];current=torch.cat(all_gather(torch.cat(micro).contiguous()));current.retain_grad();history=torch.zeros(1954,dtype=torch.float64);historical_labels=torch.tensor([i%2 for i in range(1954)],dtype=torch.float64);score=torch.cat((history,current));target=torch.cat((historical_labels,labels));loss=precision_at_k_ramp(score,target,500)*2018/64;loss.backward();assert not history.requires_grad and history.grad is None and current.grad is not None
 result=dict(rank=rank,loss=float(loss.detach()),gradients={name:p.grad.tolist()for name,p in model.named_parameters()},current_logit_gradient_L2=float(current.grad.norm()),historical_requires_grad=history.requires_grad,micro_forwards_before_one_backward=4)
 pathlib.Path(out+str(rank)+'.json').write_text(json.dumps(result));dist.destroy_process_group()

def distributed_check():
 import torch.multiprocessing as mp
 with tempfile.TemporaryDirectory(prefix='packed-window-gloo-')as tmp:
  store=str(pathlib.Path(tmp)/'store');out=str(pathlib.Path(tmp)/'rank-');mp.spawn(worker,args=(store,out),nprocs=2,join=True);actual=[json.loads(pathlib.Path(out+str(rank)+'.json').read_text())for rank in(0,1)]
 torch.manual_seed(9137);model=nn.Linear(3,1).double();features=torch.arange(64*3,dtype=torch.float64).reshape(64,3)/97-1;labels=torch.tensor([i%2 for i in range(64)],dtype=torch.float64);current=model(features).squeeze(-1);history=torch.zeros(1954,dtype=torch.float64);historical_labels=torch.tensor([i%2 for i in range(1954)],dtype=torch.float64);loss=precision_at_k_ramp(torch.cat((history,current)),torch.cat((historical_labels,labels)),500)*2018/64;loss.backward();maximum=0.
 for rank in actual:
  assert abs(rank['loss']-float(loss.detach()))<1e-10
  for name,p in model.named_parameters():
   observed=torch.tensor(rank['gradients'][name],dtype=torch.float64);error=float((observed-p.grad).abs().max());maximum=max(maximum,error);assert torch.allclose(observed,p.grad,rtol=1e-10,atol=1e-10),(name,observed,p.grad)
 return dict(world_size=2,backend='gloo',global_current=64,local_current=32,micro=8,forwards_before_single_backward=4,historical_scores=1954,window_N=2018,K=500,scale=2018/64,all_gather_backward_sum_and_DDP_gradient_average=True,one_rank_global_reference_gradients_match=True,maximum_absolute_gradient_error=maximum,actual=actual)

def main():
 started=time.time();torch.set_num_threads(1);result=dict(status='TINY_CPU_CACHE_AND_TWO_RANK_GRADIENT_INTEGRATION_PASSED',local_checks=local_checks(),distributed=distributed_check(),seconds=time.time()-started,no_GPU=True,no_Slurm=True,final4096_read=False);print(json.dumps(result))
if __name__=='__main__':main()
