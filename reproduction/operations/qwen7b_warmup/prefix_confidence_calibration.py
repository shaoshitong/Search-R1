"""One fixed CPU-only eight-scalar Prefix20 confidence calibrator.

Pure mathematics and complete scalar-state interfaces only. The caller owns
qualified extraction, file SHA/provenance, locks, durable save-before-holdout,
internal joint gate and any later single external score. No dataset/model file
is opened here, no original probe/Adam is loaded or changed, and no grid exists.
"""
import copy
import hashlib
import itertools
import json
import random
import time

import numpy as np
import torch
import torch.nn.functional as F

from . import platt_g10_calibration as base
from .g5_event_scores_dp import group_event_scores_dp

FEATURE_NAMES = ('mean_logp','std_logp','min_logp','first5_mean_logp','last5_mean_logp','actual_len_over20')
SEED = 9137
FIT_QUESTIONS = 6144
HOLDOUT_QUESTIONS = 2048
REGULARIZER = .01
MAX_ITER = 40
MAX_EVAL = 60
CHUNK_QUESTIONS = 128
PHASE = 'ALL_FIT_LBFGS_UPDATES_COMPLETE_BEFORE_HOLDOUT_ASSESSMENT'
COMBINATIONS = tuple(itertools.combinations(range(10),5))
PROTOCOL = dict(name='prefix-confidence-E4-eight-scalar-v2-matched-control',feature_names=FEATURE_NAMES,seed=SEED,
    probability_clip=1e-6,scale_floor=1e-6,regularizer=REGULARIZER,
    trajectory_weight=1.,three_event_weight=1.,views_per_G10=252,question_weight=1.,
    max_iterations=MAX_ITER,max_evaluations=MAX_EVAL,holdout_response_indices=(5,6,7,8,9),
    modes=('prefix_full8','matched_platt2'),matched_control_is_external_candidate=False,
    whole_two_mode_fit_deadline_seconds=1560)

probability_logits = base.probability_logits
raw_probability_summary = base.raw_probability_summary
same_state = base.same_state


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


def question_split(qids,holdout=HOLDOUT_QUESTIONS):
    """Freeze only-ID split before labels/scores; all ten trajectories stay together."""
    qids=[str(q) for q in qids]
    if len(qids)!=len(set(qids)) or type(holdout) is not int or not 0<holdout<len(qids):
        raise ValueError('Unique questions and a proper fixed holdout budget required')
    chosen=set(sorted(qids,key=lambda q:(hashlib.sha256(('prefix-confidence-qid-9137:'+q).encode()).hexdigest(),q))[:holdout])
    return np.asarray([q in chosen for q in qids],dtype=bool)


def tensor(value):
    if isinstance(value,torch.Tensor) and value.device.type!='cpu':
        raise ValueError('This scalar module accepts CPU tensors only')
    value=torch.as_tensor(value,dtype=torch.float64,device='cpu')
    if not bool(torch.isfinite(value).all()):raise ValueError('Finite CPU arrays required')
    return value


def validate_features(features,shape=None):
    x=tensor(features)
    if x.ndim!=3 or x.shape[-1]!=6 or x.shape[1] not in (5,10) or (shape is not None and tuple(x.shape[:2])!=tuple(shape)):
        raise ValueError('Six Prefix20 features per authentic G5/G10 trajectory required')
    if (bool((x[...,1]<0).any()) or bool((x[...,[0,2,3,4]]>1e-8).any())
            or bool((x[...,2]>x[...,0]+1e-8).any()) or bool((x[...,5]<=0).any())
            or bool((x[...,5]>1).any()) or bool(((x[...,5]*20-(x[...,5]*20).round()).abs()>1e-8).any())):
        raise ValueError('Chosen-logp/std/min/actual kept-length feature invariants differ')
    return x


def normalizer_fit(features,fit_qids):
    x=validate_features(features)
    qids=[str(q) for q in fit_qids]
    if x.shape[1]!=10 or len(qids)!=len(x) or len(qids)!=len(set(qids)) or not qids:
        raise ValueError('Normalizer receives fit questions only, with ten trajectories each')
    flat=x.reshape(-1,6)
    return dict(mean=flat.mean(0),population_std=flat.std(0,unbiased=False),
                scale=flat.std(0,unbiased=False).clamp_min(1e-6),feature_names=FEATURE_NAMES,
                fitting_questions=len(qids),fitting_trajectories=len(flat),fit_qids_sha256=digest(qids),
                fitted_on_fit_rows_only=True,scale_floor=1e-6)


def validate_normalizer(normalizer):
    if (tuple(normalizer['feature_names'])!=FEATURE_NAMES or normalizer['scale_floor']!=1e-6
            or normalizer.get('fitted_on_fit_rows_only') is not True):
        raise ValueError('Frozen fit-only six-feature normalizer required')
    mean,std,scale=[tensor(normalizer[k]) for k in ('mean','population_std','scale')]
    if any(tuple(v.shape)!=(6,) for v in (mean,std,scale)) or bool((std<0).any()) or not torch.equal(scale,std.clamp_min(1e-6)):
        raise ValueError('Normalizer population variance or scale floor differs')
    return mean,scale


def parameters(raw,use_prefix=True):
    if not isinstance(raw,torch.Tensor) or raw.device.type!='cpu' or raw.shape!=(8,) or not bool(torch.isfinite(raw).all()):
        raise ValueError('Eight finite CPU scalars required')
    log_scale=3*raw[0].tanh();bias=10*raw[1].tanh()
    if type(use_prefix) is not bool:raise ValueError('Explicit fixed calibration mode required')
    return log_scale.exp(),bias,log_scale,raw[2:] if use_prefix else raw[2:]*0


def calibrated(logits,features,raw,normalizer,use_prefix=True):
    z=tensor(logits);a,b,unused,w=parameters(raw,use_prefix)
    if not use_prefix:
        if normalizer is not None:raise ValueError('Matched2 control must not use prefix normalizer')
        return a*z+b
    x=validate_features(features,z.shape);mean,scale=validate_normalizer(normalizer)
    return a*z+b+((x-mean)/scale*w).sum(-1)


def regularization(raw,use_prefix=True):
    unused,b,log_a,w=parameters(raw,use_prefix)
    return REGULARIZER*(log_a.square()+b.square()+w.square().sum())


def objective_parts(logits,labels,features,raw,normalizer,use_prefix=True):
    z=tensor(logits);y=tensor(labels)
    if z.ndim!=2 or z.shape!=y.shape or z.shape[1]!=10 or not bool(((y==0)|(y==1)).all()) or not len(z):
        raise ValueError('Authentic strict own-EM/logits[Q,10] required')
    out=calibrated(z,features,raw,normalizer,use_prefix)
    trajectory=F.binary_cross_entropy_with_logits(out,y,reduction='mean')
    ix=torch.tensor(COMBINATIONS,dtype=torch.long)
    k=y[:,ix].sum(-1).reshape(-1)
    events=group_event_scores_dp(out[:,ix].reshape(-1,5))
    event=-torch.where(k==0,events['log_allwrong'],torch.where(k==5,events['log_allcorrect'],events['log_mixed'])).mean()
    # Same252 views for every question; each original question has weight1.
    return dict(trajectory=trajectory,three_event=event)


def chunk_objective(logits,labels,features,raw,normalizer,use_prefix=True):
    parts=objective_parts(logits,labels,features,raw,normalizer,use_prefix)
    return parts['trajectory']+parts['three_event']


def optimizer_for(raw,max_iter=MAX_ITER,max_eval=MAX_EVAL):
    return torch.optim.LBFGS([raw],lr=1.,max_iter=max_iter,max_eval=max_eval,
        tolerance_grad=1e-8,tolerance_change=1e-10,line_search_fn='strong_wolfe')


def capture_rng():
    return dict(torch=torch.get_rng_state().clone(),numpy=copy.deepcopy(np.random.get_state()),python=copy.deepcopy(random.getstate()))


def restore_rng(state):
    torch.set_rng_state(state['torch']);np.random.set_state(state['numpy']);random.setstate(state['python'])


def _binding(source_binding,qualification_only,mode):
    receipt=source_binding.get('availability_receipt',{})
    if (source_binding.get('availability_qualified') is not True or not isinstance(receipt,dict)
            or not {'path','bytes','sha256'}<=set(receipt) or len(receipt['sha256'])!=64):
        raise ValueError('Actual qualified Prefix20 logp extraction receipt required before fitting')
    if qualification_only != (source_binding.get('qualification_only') is True):
        raise ValueError('Toy qualification and real source binding must be explicit')
    if source_binding.get('calibration_mode')!=mode:raise ValueError('Source-bound calibration mode differs')


def fit(logits,labels,features,fit_qids,source_binding,progress=None,checkpoint_callback=None,
        *,qualification_only=False,max_iter=MAX_ITER,max_eval=MAX_EVAL,use_prefix=True,deadline_monotonic=None):
    """Fit once, checkpoint full state before any holdout; no automatic refit/resume.

    Real run must use6144fit qids and fixed40/60 budget. Smaller synthetic test
    qualification is explicit. A restore is available for verification, not a
    declaration that another optimizer step remains inside the registered budget.
    """
    if type(use_prefix) is not bool:raise ValueError('Explicit prefix mode required')
    mode='prefix_full8' if use_prefix else 'matched_platt2'
    _binding(source_binding,qualification_only,mode)
    if ((max_iter,max_eval)!=(MAX_ITER,MAX_EVAL) or len(fit_qids)!=FIT_QUESTIONS) and not qualification_only:
        raise ValueError('Only the fixed6144/40iteration/60evaluation real recipe is authorized')
    if not 1<=max_iter<=MAX_ITER or not 1<=max_eval<=MAX_EVAL:
        raise ValueError('Finite registered scalar fitting budget required')
    if deadline_monotonic is not None and not np.isfinite(deadline_monotonic):raise ValueError('Finite shared fit deadline required')
    z=tensor(logits);y=tensor(labels);x=validate_features(features,z.shape) if use_prefix else None
    if z.shape!=y.shape or z.ndim!=2 or z.shape[1]!=10 or len(z)!=len(fit_qids) or not bool(((y==0)|(y==1)).all()):
        raise ValueError('Aligned fit-only G10 logits/strict EM/features/qids required')
    normalizer=normalizer_fit(x,fit_qids) if use_prefix else None
    raw=torch.zeros(8,dtype=torch.float64,requires_grad=True)
    optimizer=optimizer_for(raw,max_iter,max_eval);curve=[]
    def measured(backward):
        totals=dict(trajectory=0.,three_event=0.)
        for start in range(0,len(z),CHUNK_QUESTIONS):
            if deadline_monotonic is not None and time.monotonic()>=deadline_monotonic:
                raise TimeoutError('Shared whole-two-mode fit deadline exhausted; no holdout/external assessment')
            stop=min(start+CHUNK_QUESTIONS,len(z));weight=(stop-start)/len(z)
            parts=objective_parts(z[start:stop],y[start:stop],x[start:stop] if use_prefix else None,raw,normalizer,use_prefix)
            if backward:((parts['trajectory']+parts['three_event'])*weight).backward()
            for key,value in parts.items():totals[key]+=float(value.detach())*weight
        penalty=regularization(raw,use_prefix)
        if backward:penalty.backward()
        totals['regularization']=float(penalty.detach());totals['total']=sum(totals.values())
        if not np.isfinite(list(totals.values())).all():raise ValueError('Nonfinite scalar objective')
        return totals
    def closure():
        if deadline_monotonic is not None and time.monotonic()>=deadline_monotonic:
            raise TimeoutError('Shared whole-two-mode fit deadline exhausted; no holdout/external assessment')
        if len(curve)>=max_eval:
            raise RuntimeError('Registered60-evaluation budget exceeded; no accepted scalar checkpoint or holdout score')
        optimizer.zero_grad();values=measured(True)
        if not bool(torch.isfinite(raw.grad).all()):raise ValueError('Nonfinite scalar gradient')
        curve.append(dict(evaluation=len(curve)+1,**values))
        if progress is not None:progress(len(curve),values['total'])
        return torch.tensor(values['total'],dtype=torch.float64)
    optimizer_accepted=False
    try:
        optimizer.step(closure);optimizer_accepted=True
        with torch.no_grad():accepted=measured(False)
    except BaseException as error:
        # Preserve the full auxiliary trial state. It is deliberately NOT an
        # accepted optimizer-boundary checkpoint and restore() rejects it.
        partial=dict(schema='PREFIX_CONFIDENCE_E4_PARTIAL_SCALAR_STATE_V1',mode=mode,use_prefix=use_prefix,
            raw=raw.detach().clone(),optimizer=copy.deepcopy(optimizer.state_dict()),rng=capture_rng(),
            normalizer=copy.deepcopy(normalizer),source_binding=copy.deepcopy(source_binding),protocol=copy.deepcopy(PROTOCOL),
            fit_qids_sha256=digest([str(q) for q in fit_qids]),
            phase='FIT_INTERRUPTED_ACCEPTED_OPTIMIZER_BOUNDARY_NO_HOLDOUT' if optimizer_accepted else 'FIT_INTERRUPTED_UNACCEPTED_LBFGS_TRIAL_NO_HOLDOUT',
            closure_loss_curve=copy.deepcopy(curve),accepted=optimizer_accepted,can_resume_without_requalification=False,
            error_type=type(error).__name__,error=str(error),holdout_assessed=False,external_opened=False)
        if checkpoint_callback is not None:checkpoint_callback(partial)
        raise
    report=dict(closure_calls=len(curve),initial_loss=curve[0]['total'],last_closure_loss=curve[-1]['total'],
                accepted_final_regularized_loss=accepted['total'],accepted_final_components=accepted,
                closure_loss_curve=curve,optimizer_iterations=int(optimizer.state[raw].get('n_iter',0)),
                original_probe_or_Adam_updated=False,scalar_optimizer_new=True,qualification_only=qualification_only,
                mode=mode,use_prefix=use_prefix,final_loss_verification_passes=1)
    if report['optimizer_iterations']>max_iter:raise ValueError('Accepted iteration budget exceeded')
    state=dict(schema='PREFIX_CONFIDENCE_E4_FULL_SCALAR_STATE_V1',raw=raw.detach().clone(),
        mode=mode,use_prefix=use_prefix,
        optimizer=copy.deepcopy(optimizer.state_dict()),rng=capture_rng(),normalizer=copy.deepcopy(normalizer),
        source_binding=copy.deepcopy(source_binding),fit_qids_sha256=digest([str(q) for q in fit_qids]),
        protocol=copy.deepcopy(PROTOCOL),phase=PHASE,fit=copy.deepcopy(report),
        source_probe_optimizer_reset=False,source_probe_model_loaded=False,holdout_assessed=False,external_opened=False)
    if checkpoint_callback is not None:report['full_checkpoint']=checkpoint_callback(copy.deepcopy(state))
    return state,report


def restore(state,*,restore_random_state=True):
    if (state.get('schema')!='PREFIX_CONFIDENCE_E4_FULL_SCALAR_STATE_V1' or state.get('phase')!=PHASE
            or state.get('protocol')!=PROTOCOL or state.get('holdout_assessed') is not False
            or state.get('external_opened') is not False or state.get('source_probe_optimizer_reset') is not False):
        raise ValueError('Frozen pre-holdout full scalar checkpoint required')
    mode=state.get('mode');use_prefix=state.get('use_prefix')
    if mode not in PROTOCOL['modes'] or type(use_prefix) is not bool or (mode=='prefix_full8')!=use_prefix:
        raise ValueError('Checkpoint calibration mode differs')
    _binding(state['source_binding'],state['fit']['qualification_only'],mode)
    raw=tensor(state['raw']).clone().requires_grad_();parameters(raw,use_prefix)
    if use_prefix:validate_normalizer(state['normalizer'])
    elif state['normalizer'] is not None or bool((raw[2:]!=0).any()):raise ValueError('Matched2 must preserve zero fixedw/no prefix normalizer')
    if use_prefix and state['fit_qids_sha256']!=state['normalizer']['fit_qids_sha256']:
        raise ValueError('Normalizer/fit identity mismatch')
    optimizer=optimizer_for(raw);optimizer.load_state_dict(copy.deepcopy(state['optimizer']))
    if optimizer.state[raw].get('n_iter',0)!=state['fit']['optimizer_iterations']:
        raise ValueError('Full LBFGS iteration cursor mismatch')
    if restore_random_state:restore_rng(state['rng'])
    return dict(raw=raw,optimizer=optimizer,normalizer=copy.deepcopy(state['normalizer']),
                source_binding=copy.deepcopy(state['source_binding']),phase=state['phase'],mode=mode,use_prefix=use_prefix)


def score(logits,features,state):
    with torch.no_grad():
        return group_event_scores_dp(calibrated(logits,features,state['raw'],state['normalizer'],state['use_prefix']))['mixed_probability'].numpy()


def summary(logits,labels,features,qids,state):
    y=tensor(labels)
    if y.ndim!=2 or y.shape[1]!=5 or len(qids)!=len(y) or not bool(((y==0)|(y==1)).all()):
        raise ValueError('Own fixed G5 labels and qids required for assessment')
    return base.metric(score(logits,features,state),y.sum(1).numpy(),qids,.5,min(512,len(qids)))
