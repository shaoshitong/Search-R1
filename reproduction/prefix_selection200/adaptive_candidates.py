"""Causal ordinary-outer budgets from the previous selected512 own-EM G5s.

Budget receipts are small immutable checkpoint state. No logits or predicted
correctness are feedback; only an actually committed prior joint is admissible.
P10 probe-update pools remain2560, and200 is an ordinary adaptive outer.
"""
import hashlib
import json
import math
import os
from pathlib import Path

from .contract import FINAL_OUTER, PERIODIC_CANDIDATES, PROBE_PERIOD, SELECTED_QUESTIONS, WORLD

MAX_QUESTIONS = 20480


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def policy(max_questions):
    return dict(kind='previous_selected512_actual_mixed_v1',initial_questions=5120,
        epsilon=1e-6,all_mixed_denominator=1.1,minimum_questions=SELECTED_QUESTIONS,
        maximum_questions=min(MAX_QUESTIONS,max_questions//WORLD*WORLD),rounding='ceil_multiple_of_world4',
        periodic_questions=PERIODIC_CANDIDATES,probe_update_period=PROBE_PERIOD,
        probe_update_at_final=False,feedback='immediately_previous_committed_outer_selected512_actual_EM')


def validate(config):
    value=config.get('adaptive_candidates')
    if value is None:return None
    if (not isinstance(value,dict) or type(value.get('maximum_questions')) is not int
            or value['maximum_questions']<5120 or value!=policy(value['maximum_questions'])
            or any(type(value[k]) is not type(v) for k,v in policy(value['maximum_questions']).items())):
        raise ValueError('Bind the exact adaptive ordinary/P10-fixed candidate policy')
    return dict(value)


def prediction(previous_questions, mixed_questions, spec):
    if (type(previous_questions) is not int or previous_questions%WORLD
            or not SELECTED_QUESTIONS<=previous_questions<=spec['maximum_questions']
            or type(mixed_questions) is not int or not 0<=mixed_questions<=SELECTED_QUESTIONS):
        raise ValueError('Feedback requires a real candidate count and own-EM mixed count out of512')
    ratio=mixed_questions/SELECTED_QUESTIONS
    denominator=(spec['all_mixed_denominator'] if mixed_questions==SELECTED_QUESTIONS else ratio)+spec['epsilon']
    raw=previous_questions/denominator
    rounded=math.ceil(raw/WORLD)*WORLD
    count=max(spec['minimum_questions'],min(spec['maximum_questions'],rounded))
    return dict(questions=count,previous_questions=previous_questions,
        previous_selected_mixed_questions=mixed_questions,previous_selected_mixed_fraction=ratio,
        denominator=denominator,raw_predicted_questions=raw,rounded_questions=rounded,
        bounded=count!=rounded)


def path(config,outer):
    return Path(config['base'])/'outers'/('outer-%06d'%outer)/'candidate-budget.json'


def read(path):
    path=Path(path)
    if path.is_symlink() or path.stat().st_size>4*1024**2:
        raise ValueError('Adaptive budget input must be regular small metadata')
    return json.loads(path.read_text())


def checked(config,outer,value):
    spec=validate(config)
    if (type(outer) is not int or not 1<=outer<=FINAL_OUTER or spec is None
            or value.get('outer')!=outer or value.get('policy_sha256')!=digest(spec)
            or value.get('run_id')!=config.get('run_id')):
        raise ValueError('Adaptive budget belongs to another run/outer/policy')
    if outer==1:
        count=spec['initial_questions'];mode='initial'
        if value.get('consumed_before')!=0:raise ValueError('Fresh adaptive candidate clock must start at zero')
    else:
        expected=prediction(value['previous_questions'],value['previous_selected_mixed_questions'],spec)
        if any(value.get(k)!=v for k,v in expected.items() if k!='questions'):
            raise ValueError('Adaptive budget formula or actual512 feedback changed')
        fixed=outer<FINAL_OUTER and outer%PROBE_PERIOD==0
        count=spec['periodic_questions'] if fixed else expected['questions'];mode='fixed_probe_update' if fixed else 'adaptive'
        if value.get('previous_outer')!=outer-1 or not isinstance(value.get('previous_joint_sha256'),str) or len(value['previous_joint_sha256'])!=64:
            raise ValueError('Adaptive budget needs its immediately previous actual joint provenance')
    if (value.get('questions')!=count or value.get('mode')!=mode
            or type(value.get('consumed_before')) is not int or value['consumed_before']<0
            or value.get('consumed_after')!=value['consumed_before']+count):
        raise ValueError('Adaptive candidate count/mode/cumulative clock differs')
    return dict(value)


def derive(config,outer):
    if config.get('two_stage_selection'):
        from .two_stage_selection import derive as fixed_derive
        return fixed_derive(config,outer)
    spec=validate(config)
    common=dict(schema_version=1,run_id=config['run_id'],outer=outer,policy_sha256=digest(spec))
    if outer==1:
        return checked(config,outer,dict(common,mode='initial',questions=spec['initial_questions'],
            consumed_before=0,consumed_after=spec['initial_questions']))
    previous=budget(config,outer-1)
    joint_path=Path(config['base'])/'outers'/('outer-%06d'%(outer-1))/'complete.json'
    raw=joint_path.read_bytes();joint=json.loads(raw);full=joint['rollout']
    if (joint.get('run_id')!=config['run_id'] or joint.get('outer')!=outer-1
            or full.get('run_id')!=config['run_id'] or full.get('outer')!=outer-1
            or full.get('questions')!=SELECTED_QUESTIONS or full.get('trajectories')!=SELECTED_QUESTIONS*5):
        raise ValueError('Candidate feedback must use an actual committed previous512-G5 rollout')
    mixed=full['actual_selected_mixed_questions']
    forecast=prediction(previous['questions'],mixed,spec)
    if (not isinstance(full.get('actual_selected_mixed_fraction'),(int,float))
            or not math.isfinite(full['actual_selected_mixed_fraction'])
            or abs(full['actual_selected_mixed_fraction']-mixed/SELECTED_QUESTIONS)>1e-12):
        raise ValueError('Actual selected512 mixed count and fraction disagree')
    fixed=outer<FINAL_OUTER and outer%PROBE_PERIOD==0
    count=spec['periodic_questions'] if fixed else forecast['questions']
    return checked(config,outer,dict(common,**{k:v for k,v in forecast.items() if k!='questions'},
        mode='fixed_probe_update' if fixed else 'adaptive',questions=count,
        previous_outer=outer-1,previous_joint_sha256=hashlib.sha256(raw).hexdigest(),
        consumed_before=previous['consumed_after'],consumed_after=previous['consumed_after']+count))


def budget(config,outer):
    if config.get('two_stage_selection'):
        from .two_stage_selection import budget as fixed_budget
        return fixed_budget(config,outer)
    if config.get('progressive_prefix') and outer<=50:
        # Historical budgets retain the old cap/hash and source journal. Only
        # the newly authorized suffix is512..5120; do not rewrite its prefix.
        source=Path(config['resume_joint']['source_base'])
        return budget(read(source/'preparation/launch-config.json'),outer)
    if type(outer) is not int or not 1<=outer<=FINAL_OUTER:
        raise ValueError('Adaptive candidate outer must be1..200')
    if config.get('candidate_budget') is not None:
        return checked(config,outer,config['candidate_budget'])
    target=path(config,outer)
    return checked(config,outer,read(target)) if target.exists() else derive(config,outer)


def ensure(config,outer):
    """Call under the candidate-pool writer lock, before advancing its cursor."""
    if config.get('two_stage_selection'):
        from .two_stage_selection import ensure as fixed_ensure
        return fixed_ensure(config,outer)
    expected=derive(config,outer);target=path(config,outer)
    if target.exists():
        if checked(config,outer,read(target))!=expected:
            raise ValueError('Existing adaptive budget/source feedback differs; never silently resize a reservation')
    else:
        with target.open('x') as f:
            json.dump(expected,f,sort_keys=True,allow_nan=False);f.flush();os.fsync(f.fileno())
    return expected
