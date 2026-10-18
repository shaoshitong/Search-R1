"""Authorized suffix: 5120 native-prefix groups -> predicted1024 -> actual512.

P10 remains the independent unselected2560G5/three-epoch probe-fit contract.
Ordinary actor groups use outcomes only after all1024 original G5s complete.
"""
import json
from pathlib import Path
from .pool import read, immutable, digest

MODE='fixed5120_probe1024_actual512_v1'

def specification(first_outer):
    return dict(mode=MODE,effective_from_outer=first_outer,prefix_questions=5120,
        continuation_questions=1024,actor_questions=512,periodic_questions=2560,
        periodic_probe_budget_unchanged=True,ranking='actual_mixed_then_probe_q_descending_then_ordinal',
        shortfall='probe_score_fill',same_original_G5=True)

def validate(config):
    spec=config.get('two_stage_selection')
    if spec is None:return None
    first=spec.get('effective_from_outer')
    if type(first) is not int or not 2<=first<=200 or spec!=specification(first):
        raise ValueError('Two-stage selection must bind the exact authorized5120/1024/512 contract')
    if config.get('actual_em_selection_fallback')!='probe_score_fill':
        raise ValueError('The retained explicit shortfall policy is probe_score_fill')
    return spec

def active(config,outer):
    spec=validate(config)
    return bool(spec and outer>=spec['effective_from_outer'] and not config.get('diagnostic_only')
                and not config.get('progressive_probe_refresh'))

def ordinary(config,outer):
    from .periodic import is_probe_update
    return active(config,outer) and not is_probe_update(outer)

def historical(config,outer):
    source=Path(config['resume_joint']['source_base'])
    from .adaptive_candidates import budget
    return budget(read(source/'preparation/launch-config.json'),outer)

def derive(config,outer):
    from .adaptive_candidates import digest as obj_sha
    spec=validate(config)
    if outer<spec['effective_from_outer']:return historical(config,outer)
    from .adaptive_candidates import budget
    previous_config=dict(config);previous_config.pop('candidate_budget',None)
    previous=budget(previous_config,outer-1)
    prior=Path(config['base'])/'outers'/('outer-%06d'%(outer-1))/'complete.json'
    joint=read(prior)
    if joint['outer']!=outer-1 or joint['run_id']!=config['run_id']:
        raise ValueError('Fixed suffix requires its actual immediately previous joint')
    from .periodic import is_probe_update
    fixed=is_probe_update(outer);count=2560 if fixed else 5120
    return dict(schema_version=2,run_id=config['run_id'],outer=outer,policy_sha256=obj_sha(spec),
        mode='fixed_probe_update' if fixed else MODE,questions=count,previous_outer=outer-1,
        previous_joint_sha256=digest(prior),previous_questions=previous['questions'],
        previous_selected_mixed_questions=joint['rollout']['actual_selected_mixed_questions'],
        consumed_before=previous['consumed_after'],consumed_after=previous['consumed_after']+count,
        fixed_budget_not_mixed_feedback=True)

def checked(config,outer,value):
    from .adaptive_candidates import digest as obj_sha
    from .periodic import is_probe_update
    spec=validate(config);fixed=is_probe_update(outer);count=2560 if fixed else 5120
    if (value.get('schema_version')!=2 or value.get('run_id')!=config['run_id'] or value.get('outer')!=outer
        or value.get('policy_sha256')!=obj_sha(spec) or value.get('questions')!=count
        or value.get('mode')!=('fixed_probe_update' if fixed else MODE)
        or value.get('previous_outer')!=outer-1 or len(value.get('previous_joint_sha256',''))!=64
        or type(value.get('consumed_before')) is not int or value['consumed_before']<0
        or value.get('consumed_after')!=value['consumed_before']+count
        or value.get('fixed_budget_not_mixed_feedback') is not True):
        raise ValueError('Two-stage fixed budget/cursor/parent joint changed')
    return dict(value)

def budget(config,outer):
    spec=validate(config)
    if outer<spec['effective_from_outer']:return historical(config,outer)
    if config.get('candidate_budget') is not None:return checked(config,outer,config['candidate_budget'])
    p=Path(config['base'])/'outers'/('outer-%06d'%outer)/'candidate-budget.json'
    return checked(config,outer,read(p)) if p.exists() else derive(config,outer)

def ensure(config,outer):
    expected=derive(config,outer)
    p=Path(config['base'])/'outers'/('outer-%06d'%outer)/'candidate-budget.json'
    if p.exists():
        if checked(config,outer,read(p))!=expected:raise ValueError('Reserved fixed budget differs from actual parent/cursor')
    else:immutable(p,expected)
    return expected

def complete(config_path,outer):
    from .rollout import root_dir,launch_phase,validate_selection
    from .periodic_data import actual_em_selection,project_top512
    c=read(config_path)
    if not ordinary(c,outer):raise ValueError('Use two-stage1024 only on its ordinary suffix')
    root=root_dir(c,outer);selected=root/'probe-score/pool-selection.json'
    validate_selection(read(selected),read(root/'prefix/complete.json'),1024)
    scoped=root/'full-pool-config.json'
    immutable(scoped,dict(c,rollout_continuation_questions=1024,rollout_full_subdirectory='full-pool'))
    launch_phase(scoped,outer,'continue',selected)
    final=actual_em_selection(config_path,outer,allow_two_stage=True)
    return project_top512(config_path,outer,final)


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--outer',type=int,required=True)
    args=parser.parse_args();print(json.dumps(complete(args.config,args.outer)))
