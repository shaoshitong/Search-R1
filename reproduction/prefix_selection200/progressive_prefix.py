"""Explicit CP50 continuation: increasing causal prefixes and three-epoch P10.

Disabled for every historical run unless its exact execution mode is selected.
An actor outer consumes the probe fitted after the preceding boundary: outer100
still uses Prefix20; the post-100 fit and outer101 use Prefix40. Outer200 has
no probe update. CP50's old250 Adam writes are retained before150 new writes.
"""
import os

MODE='cp50_prefix20_40_80_three_epoch_v1'
PREFIXES=(5,20,40,80)
BOUNDARIES=(50,100,150)
QMAX=273
DIM=2048
EPOCHS=3
STEPS_PER_EPOCH=50
UPDATES=EPOCHS*STEPS_PER_EPOCH


def enabled():
    value=os.environ.get('DYNRPO_PROGRESSIVE_PREFIX','')
    if value not in ('',MODE):raise ValueError('Unknown progressive prefix execution mode')
    return value==MODE


def process_prefix():
    value=int(os.environ.get('DYNRPO_PREFIX_TOKENS','5'))
    if value not in PREFIXES or (not enabled() and value!=5):
        raise ValueError('Only the explicit progressive execution admits20/40/80')
    return value


def after_boundary(completed_outer):
    if type(completed_outer) is not int or not 0<=completed_outer<=200:
        raise ValueError('Completed student boundary must be0..200')
    return 5 if completed_outer<50 else 20 if completed_outer<100 else 40 if completed_outer<150 else 80


def actor_prefix(outer):
    if type(outer) is not int or not 1<=outer<=200:raise ValueError('Actor outer must be1..200')
    return after_boundary(outer-1)


def boundary(completed_outer):
    if type(completed_outer) is not int or not 0<=completed_outer<=200:
        raise ValueError('Completed student boundary must be0..200')
    if completed_outer<50:
        last=completed_outer//10*10;steps=last//10*50
    else:
        last=min(completed_outer//10*10,190)
        steps=250+(1+(last-50)//10)*UPDATES
    return dict(last_update_outer=last,actual_adam_per_rank=steps)


def before_fit(outer):
    if outer not in range(50,200,10):raise ValueError('Progressive probe fits50,60,...190')
    return dict(last_update_outer=50,actual_adam_per_rank=250) if outer==50 else boundary(outer-1)


def width(prefix):
    if prefix not in PREFIXES:raise ValueError('Unsupported prefix length')
    return QMAX+prefix


def expand_probe_state(saved,new_prefix):
    """Retain every block tensor and Adam/RNG clock; extend only Linear slots.

    Old question and policy positions retain their weights/moments. New positions
    receive zero weights and zero first/second moments, with the parameter's
    existing Adam step preserved. Bias, block, buffers and four RNGs are retained.
    This is an explicit parameter expansion, never a student/block reinitialization.
    """
    import copy
    import torch
    result=copy.deepcopy(saved)
    old=result['model']['output.weight'];old_width=old.shape[1]//DIM
    new_width=width(new_prefix)
    if old.shape!=(1,old_width*DIM) or old_width-QMAX not in PREFIXES or new_width<old_width:
        raise ValueError('Only an admitted nonshrinking flattened readout expansion is allowed')
    if new_width==old_width:return result
    def extend(value):
        if value.shape!=old.shape:raise ValueError('Output Adam moment must match its old parameter')
        new=torch.zeros((1,new_width*DIM),dtype=value.dtype,device=value.device)
        new[:,:old.shape[1]].copy_(value);return new
    result['model']['output.weight']=extend(old)
    names=list(result['model'])
    # The saved optimizer records model.parameters() order. Require its explicit
    # parameter-name seal, rather than guessing an integer ID from tensor shape.
    layout=saved.get('optimizer_parameter_names')
    if layout is None:raise ValueError('Seal the actual model/optimizer parameter order before expansion')
    flat=[param for group in result['optimizer']['param_groups'] for param in group['params']]
    if len(layout)!=len(flat) or set(layout)!=set(names) or len(set(flat))!=len(flat):
        raise ValueError('Saved optimizer parameter identity layout differs')
    output_id=flat[layout.index('output.weight')]
    state=result['optimizer']['state'][output_id]
    for key in ('exp_avg','exp_avg_sq'):
        state[key]=extend(state[key])
    if 'max_exp_avg_sq' in state:state['max_exp_avg_sq']=extend(state['max_exp_avg_sq'])
    result['prefix_migration']=dict(old_prefix_tokens=old_width-QMAX,new_prefix_tokens=new_prefix,
        question_slots_preserved=QMAX,block_parameters_reinitialized=False,
        output_old_coordinates_preserved=True,new_output_coordinates_zero=True,
        optimizer_steps_reset=False,rank_rng_reset=False)
    return result
