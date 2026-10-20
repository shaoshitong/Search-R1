"""Prefix geometry migration preserves every old weight and Adam moment."""
import copy,json
import torch
from .prefix_migration import extend,state_hash,optimizer_state_hash

def main():
 torch.manual_seed(1729);qmax,dim=2,4;old=(qmax+5)*dim
 raw={'block.weight':torch.randn(3,3),'output.weight':torch.randn(1,old),'output.bias':torch.randn(1)}
 ema={k:v*.9 for k,v in raw.items()};state={i:dict(step=torch.tensor(9146.),exp_avg=torch.randn_like(v),exp_avg_sq=torch.rand_like(v)) for i,v in enumerate(raw.values())}
 optimizer=dict(state=state,param_groups=[dict(params=list(state),lr=5e-6,weight_decay=0)])
 saved=dict(raw=raw,ema=ema,optimizer=optimizer,steps=9146,rank_states=[dict(rank=r,rng=dict(marker=r),model_sha256=state_hash(raw),ema_sha256=state_hash(ema),optimizer_sha256=optimizer_state_hash(optimizer)) for r in range(4)])
 before=(state_hash(raw),state_hash(ema),optimizer_state_hash(optimizer));new,proof=extend(saved,list(raw),qmax,dim)
 assert before==(state_hash(raw),state_hash(ema),optimizer_state_hash(optimizer))
 for key in ('raw','ema'):
  assert torch.equal(new[key]['output.weight'][:,:old],saved[key]['output.weight']);assert (new[key]['output.weight'][:,old:]==0).all()
  for name in ('block.weight','output.bias'):assert new[key][name] is saved[key][name]
 for key in ('exp_avg','exp_avg_sq'):
  assert torch.equal(new['optimizer']['state'][1][key][:,:old],saved['optimizer']['state'][1][key]);assert (new['optimizer']['state'][1][key][:,old:]==0).all()
 assert new['optimizer']['state'][0] is saved['optimizer']['state'][0]
 assert all(int(s['step'])==9146 for s in new['optimizer']['state'].values())
 # A zero-expanded readout preserves the function; wider GEMMs may change FP32 reduction order.
 h=torch.randn(10,(qmax+20)*dim)
 a=torch.nn.functional.linear(h[:,:old],raw['output.weight'],raw['output.bias']);b=torch.nn.functional.linear(h,new['raw']['output.weight'],new['raw']['output.bias'])
 assert torch.allclose(a,b,atol=1e-5,rtol=1e-6)
 print(json.dumps(dict(status='PREFIX20_MIGRATION_OLD_WEIGHTS_MOMENTS_STEPS_AND_READOUT_PRESERVED',source_checkpoint_object_unchanged=True,readout_fp32_max_error=float((a-b).abs().max()),existing_weights_and_moments_bit_exact=True,geometry=(old,(qmax+20)*dim),steps=9146)))
if __name__=='__main__':main()
