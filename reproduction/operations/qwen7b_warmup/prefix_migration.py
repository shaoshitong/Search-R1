"""Extend CP2 Prefix5 readout to Prefix20 without resetting any old parameter."""
import copy,hashlib
import torch

def state_hash(state):
 h=hashlib.sha256()
 for name,value in state.items():h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()

def optimizer_state_hash(state):
 h=hashlib.sha256()
 for group in state['param_groups']:
  for parameter in group['params']:
   for key,value in sorted(state['state'][parameter].items()):h.update(key.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes() if torch.is_tensor(value) else repr(value).encode())
 return h.hexdigest()

def extend(saved,parameter_names,qmax=273,dim=3584,old_prefix=5,new_prefix=20):
 assert new_prefix>old_prefix and saved['steps']==9146
 assert list(saved['raw'])==list(parameter_names),'Parameter/order mapping must be explicit'
 ids=[p for g in saved['optimizer']['param_groups'] for p in g['params']];assert len(ids)==len(parameter_names)
 weight_id=dict(zip(parameter_names,ids))['output.weight'];old=(qmax+old_prefix)*dim;new=(qmax+new_prefix)*dim
 def pad(v):
  assert tuple(v.shape)==(1,old)
  out=v.new_zeros((1,new));out[:,:old].copy_(v);assert torch.equal(out[:,:old],v) and not bool(out[:,old:].any());return out
 result=dict(saved)
 for key in ('raw','ema'):
  result[key]=dict(saved[key]);result[key]['output.weight']=pad(saved[key]['output.weight'])
  assert all(result[key][k] is saved[key][k] for k in saved[key] if k!='output.weight')
 result['optimizer']=dict(saved['optimizer']);result['optimizer']['param_groups']=copy.deepcopy(saved['optimizer']['param_groups']);result['optimizer']['state']=dict(saved['optimizer']['state'])
 own=dict(saved['optimizer']['state'][weight_id])
 for key in ('exp_avg','exp_avg_sq','max_exp_avg_sq'):
  if key in own:own[key]=pad(own[key])
 result['optimizer']['state'][weight_id]=own
 assert int(own['step'])==9146
 for i,value in result['optimizer']['state'].items():
  assert int(value['step'])==9146
  if i!=weight_id:assert value is saved['optimizer']['state'][i]
 raw_sha,ema_sha,opt_sha=state_hash(result['raw']),state_hash(result['ema']),optimizer_state_hash(result['optimizer'])
 result['rank_states']=[]
 for rank in saved['rank_states']:
  own=dict(rank);own.update(model_sha256=raw_sha,ema_sha256=ema_sha,optimizer_sha256=opt_sha);assert own['rng'] is rank['rng'];result['rank_states'].append(own)
 report=dict(status='CP2_PREFIX5_TO20_OLD_PARAMETERS_AND_ADAM_COLUMNS_PRESERVED',old_prefix=old_prefix,new_prefix=new_prefix,old_readout_columns=old,new_readout_columns=new,new_columns_initialized_to_zero=True,new_column_moments_zero=True,existing_block_and_bias_unchanged=True,existing_moments_and_steps_preserved=True,steps=9146,raw_sha256=raw_sha,ema_sha256=ema_sha,optimizer_sha256=opt_sha)
 result['prefix_migration']=report;return result,report

@torch.no_grad()
def qualify(model,reader,groups,device):
 # The only new columns are zero. Check the first278 causal outputs remain
 # numerically compatible, rather than claiming BF16 kernel bit identity.
 baseline=copy.deepcopy(model).eval();baseline.prefix=5;baseline.width=model.qmax+5
 baseline.output=torch.nn.Linear(baseline.width*3584,1,device=device,dtype=model.output.weight.dtype)
 baseline.output.weight.copy_(model.output.weight[:,:baseline.width*3584]);baseline.output.bias.copy_(model.output.bias)
 model.eval();data=reader.get([tid for g in groups[:2] for tid in g['trajectory_ids']],device)
 with torch.autocast('cuda',dtype=torch.bfloat16):old=baseline(data);new=model(data)
 logit_error=float((old-new).abs().max());prob_error=float((old.sigmoid()-new.sigmoid()).abs().max())
 assert logit_error<=.05 and prob_error<=.01,('Expanded causal readout deviates at initialization',logit_error,prob_error)
 del baseline
 return dict(status='GPU_PREFIX20_ZERO_EXTENSION_CAUSAL_INITIALIZATION_CHECK_PASSED',max_absolute_logit_error=logit_error,max_absolute_probability_error=prob_error,probability_tolerance=.01,weight_migration_exact=True,compute_bit_exact_not_claimed=True)
