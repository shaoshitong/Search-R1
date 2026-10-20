"""CPU state test: reducer graph warmup must not advance Adam or EMA."""
import copy,json
from contextlib import nullcontext
import torch
from . import mixed_finetune as fit
from .replay_diagnostics import snapshot,compare

class Model(torch.nn.Module):
 def __init__(self):super().__init__();self.output=torch.nn.Linear(3,1)
 def forward(self,data):return self.output(data['x'])
class Wrapper:
 def __init__(self,model):self.model=model
 def no_sync(self):return nullcontext()
 def __call__(self,data):return self.model(data)
class Reader:
 def get(self,ids,device):return dict(x=torch.tensor([[float(i%3),float(i%2),1.] for i in ids],device=device))

def main():
 torch.manual_seed(1729);model=Model();ema=copy.deepcopy(model);opt=torch.optim.Adam(model.parameters(),lr=5e-6)
 model(dict(x=torch.ones(1,3))).sum().backward();opt.step();opt.zero_grad(set_to_none=True)
 y=[1,0,1,0,0];mask=[[bool(y[i] and not y[j]) for j in range(5)] for i in range(5)];groups=[dict(trajectory_ids=list(range(5)),labels=y,pair_mask=mask)]*4
 original_reduce=fit.dist.all_reduce;original_autocast=fit.torch.autocast
 try:
  fit.dist.all_reduce=lambda *a,**k:None;fit.torch.autocast=lambda *a,**k:nullcontext()
  before=snapshot(model,ema,opt)
  fit.update(Wrapper(model),model,ema,opt,Reader(),groups,list(range(4)),0,torch.device('cpu'),dict(pair_weight=.1,pair_margin=1.),commit=False)
  assert compare(before,snapshot(model,ema,opt))['bit_exact'];assert fit.actual_steps(opt)==1
  fit.update(Wrapper(model),model,ema,opt,Reader(),groups,list(range(4)),0,torch.device('cpu'),dict(pair_weight=.1,pair_margin=1.))
  assert fit.actual_steps(opt)==2 and not compare(before,snapshot(model,ema,opt))['bit_exact']
 finally:fit.dist.all_reduce=original_reduce;fit.torch.autocast=original_autocast
 print(json.dumps(dict(status='DRY_DDP_GRAPH_WARMUP_PRESERVES_MODEL_EMA_ADAM_COUNTER_AND_MOMENTS',CPU_state_test=True,four_rank_GPU_not_claimed=True)))
if __name__=='__main__':main()
