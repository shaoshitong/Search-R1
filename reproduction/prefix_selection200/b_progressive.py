"""Human-authorized B-only CP50 fork: native20/40/80, top768/1024/1536.

Ordinary rounds use the reference actual-mixed-first rerank after continuation.
A and historical five-token execution are unchanged when the mode is absent.
The B optimizer, balancing, EMA, loss and periodic2560 pool are unchanged.
"""
import os
MODE='b_cp50_prefix20_40_80_two_stage_v1'
ENV='SEARCH_G1_B_PREFIX_TOKENS'
def enabled(config):return config.get('b_progressive',{}).get('mode')==MODE
def prefix(outer):
    if not 1<=outer<=200:raise ValueError('Actor outer1..200 required')
    return 5 if outer<=50 else 20 if outer<=100 else 40 if outer<=150 else 80
def retained_prefix():
    value=int(os.environ.get(ENV,'5'))
    if value not in (5,20,40,80):raise ValueError('Unsupported authorized prefix length')
    return value
def continuation_questions(outer):return {20:768,40:1024,80:1536}[prefix(outer)]
def ordinary(config,outer):
    from .periodic import is_probe_update
    return enabled(config) and outer>50 and not is_probe_update(outer) and not config.get('diagnostic_only')
def validate(config):
    if not enabled(config):return
    from .contract import WORLD
    cards=list(range(8)) if WORLD==8 else [4,5,6,7]
    if WORLD==4 and config.get('resume_joint',{}).get('recovery_mode')in ('diverse_eight_to_four_cp181_v1','diverse_same_four_latest_joint_v1'):cards=[0,1,2,3]
    if config.get('selection_probe_experiment')!='balanced_online_ema' or config['physical_gpu_indices']!=cards:raise ValueError('Configured B cards only')
    if config['b_progressive'].get('resume_outer')!=50:raise ValueError('Initial fork is complete CP50')

def expand(saved,new_prefix):
    """Keep raw/EMA block and old Linear coordinates, moments, step and RNG."""
    import copy,torch
    out=copy.deepcopy(saved);old=saved['model']['output.weight'];old_prefix=old.shape[1]//2048-273
    if old_prefix not in (5,20,40,80) or new_prefix not in (20,40,80) or new_prefix<old_prefix:raise ValueError('Only nonshrinking prefix readout expansion')
    if old_prefix==new_prefix:return out
    def extension(value):
        if value.shape!=old.shape:raise ValueError('Linear moment shape differs')
        result=value.new_zeros((1,(273+new_prefix)*2048));result[:,:value.shape[1]].copy_(value);return result
    for key in ('model','ema_model'):out[key]['output.weight']=extension(saved[key]['output.weight'])
    # Saved model state contains exactly the fourteen learned parameters in native
    # model.parameters order, as validated by the actual CP50 CPU proof.
    names=list(saved['model']);params=[v for g in saved['optimizer']['param_groups'] for v in g['params']]
    if len(names)!=len(params) or len(names)!=14:raise ValueError('Saved native optimizer parameter layout differs')
    for name,param in zip(names,params):
        for moment in ('exp_avg','exp_avg_sq'):
            if saved['optimizer']['state'][param][moment].shape!=saved['model'][name].shape:raise ValueError('Optimizer parameter order differs')
    state=out['optimizer']['state'][params[names.index('output.weight')]]
    for key in ('exp_avg','exp_avg_sq','max_exp_avg_sq'):
        if key in state:state[key]=extension(state[key])
    out['binding']=dict(out['binding'],prefix_tokens=new_prefix,head_max_tokens=273+new_prefix)
    # Use the same canonical object digest as probe.py.
    import hashlib,json
    out['binding_sha256']=hashlib.sha256(json.dumps(out['binding'],sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
    out['prefix_migration']=dict(old_prefix=old_prefix,new_prefix=new_prefix,block_reset=False,optimizer_reset=False,ema_reset=False,rank_rng_reset=False,new_linear_coordinates_zero=True)
    return out

def migrate(base,new_prefix):
    import json,torch
    from pathlib import Path
    from reproduction.prefix20_online.launch import read,sha,once
    from .probe import atomic_checkpoint
    b=Path(base);before=read(b/'probe/latest.json');saved=torch.load(b/'probe/latest.pt',map_location='cpu',weights_only=False)
    old=saved['binding']['prefix_tokens']
    if old==new_prefix:return before
    migrated=expand(saved,new_prefix)
    # Dimension migration is not an Adam write and cannot rewrite a milestone.
    receipt=atomic_checkpoint(b/'probe',migrated,before['completed_outer'],dimension_migration=True)
    once(b/'preparation'/('prefix-migration-%d-to-%d-at-%d.json'%(old,new_prefix,before['completed_outer'])),dict(before=before,after=receipt,migration=migrated['prefix_migration']))
    return receipt


def complete(config_path,outer):
    from .pool import read,immutable
    from .rollout import root_dir,launch_phase,validate_selection
    from .periodic_data import actual_em_selection,project_top512
    c=read(config_path);n=continuation_questions(outer)
    if not ordinary(c,outer):raise ValueError('Only ordinary B suffix may use this expanded pool')
    root=root_dir(c,outer);selection=root/'probe-score/pool-selection.json'
    validate_selection(read(selection),read(root/'prefix/complete.json'),n)
    conf=root/'full-pool-config.json';immutable(conf,dict(c,rollout_continuation_questions=n,rollout_full_subdirectory='full-pool'))
    launch_phase(conf,outer,'continue',selection)
    final=actual_em_selection(config_path,outer,allow_b_progressive=True)
    return project_top512(config_path,outer,final,allow_b_progressive=True)
