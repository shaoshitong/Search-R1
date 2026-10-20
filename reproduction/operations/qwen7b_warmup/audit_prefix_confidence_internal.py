"""Independent CPU pre-external audit of complete prefix calibration states.

Only small auxiliary CPs, compact prefix statistics and INTERNAL predictions.
No original model/optimizer payload or external/final prediction is opened.
Actual scalars, normalizer values, optimizer/RNG and rows remain remote.
"""
import argparse
import gzip
import hashlib
import io
import json
import math
from pathlib import Path
import time

from . import e2_e4_fixed_ensemble as n

BASE=n.A/'prefix-confidence-incremental-v1'
MODES=('matched_platt2','prefix_full8')
FEATURES=('mean_logp','std_logp','min_logp','first5_mean_logp','last5_mean_logp','actual_len_over20')
PROTOCOL_COMMIT='a1ec8bbe44a24deb9ada77b59f3edf57cdc653e2'
PROTOCOL_SHA='999d173ca5cb4b68db5a0d1d80e46e46d54e11dc9f556a43e92ef34003759658'
MATH_SHA='3244d6a0fb68793fb23913650874d4b24ace4b3721df1c4d82446771ea22a761'
PHASE='ALL_FIT_LBFGS_UPDATES_COMPLETE_BEFORE_HOLDOUT_ASSESSMENT'
STATUS='PREFIX_CONFIDENCE_INTERNAL_FULLSTATE_AND_METRICS_INDEPENDENTLY_PASSED'


def close(a,b,tol=1e-11):
    return a==b if a is None or b is None else abs(a-b)<=tol


def sigmoid(value):
    if value>=0:return 1/(1+math.exp(-value))
    e=math.exp(value);return e/(1+e)


def split(qids):
    chosen=set(sorted(qids,key=lambda q:(hashlib.sha256(('prefix-confidence-qid-9137:'+str(q)).encode()).hexdigest(),str(q)))[:2048])
    return [q for q in qids if q not in chosen],[q for q in qids if q in chosen]


def compare_metrics(actual,declared):
    p,s=actual['pool'],actual['selected512']
    mapping=dict(balanced_accuracy=p['mixed_balanced_accuracy'],Mixed_recall=p['mixed_recall'],
        nonMixed_recall=p['nonmixed_recall'],allwrong_FPR=p['classwise']['allwrong']['false_positive_rate'],
        allcorrect_FPR=p['classwise']['allcorrect']['false_positive_rate'],AUC=p['mixed_roc_auc'])
    if any(not close(v,declared[k]) for k,v in mapping.items()):raise ValueError('Independent internal metric differs')
    if declared['confusion']!=p['confusion'] or declared['questions']!=p['questions'] or declared['fixed_threshold']!=.5:
        raise ValueError('Internal confusion/threshold/count differs')
    top=declared['Top512']
    if (top['Mixed']!=s['mixed_questions'] or top['K_histogram']!=s['k_histogram']
            or top['questions']!=512 or not close(top['Mixed_fraction'],s['mixed_fraction'])):
        raise ValueError('Independent Top512 own composition differs')


def gate(raw,control,full):
    rb,cb,fb=[x['pool']['mixed_balanced_accuracy'] for x in (raw,control,full)]
    rt,ct,ft=[x['selected512']['mixed_questions'] for x in (raw,control,full)]
    values=dict(raw_strict_BA=fb>rb,raw_strict_Top512=ft>rt,control_BA_nondecrease=fb>=cb,
                control_Top512_nondecrease=ft>=ct,control_at_least_one_strict=fb>cb or ft>ct)
    values['passed']=all(values.values());return values


def clipping_order_audit(raw,clipped):
    def ordered(rows):return sorted(rows,key=lambda r:(-r['score'],str(r['qid'])))
    def ties(rows):
        counts={}
        for row in rows:counts[row['score']]=counts.get(row['score'],0)+1
        boundary=ordered(rows)[511]['score']
        return dict(tied_score_groups=sum(v>1 for v in counts.values()),tied_questions=sum(v for v in counts.values() if v>1),
                    tied_pairs=sum(v*(v-1)//2 for v in counts.values()),Top512_boundary_equal_score_questions=counts[boundary])
    a,b=ordered(raw),ordered(clipped);aq=[r['qid'] for r in a];bq=[r['qid'] for r in b]
    return dict(Top512_ID_intersection=len(set(aq[:512])&set(bq[:512])),Top512_order_identical=aq[:512]==bq[:512],
        all_question_rank_positions_changed=sum(x!=y for x,y in zip(aq,bq)),raw_order_sha256=n.digest(aq),
        clipped_identity_order_sha256=n.digest(bq),raw_ties=ties(raw),clipped_identity_ties=ties(clipped))


def read_gzip(reference):
    with gzip.open(n.checked(reference),'rt') as stream:return [json.loads(line) for line in stream]


def fingerprint(value):
    """Hash only, never return actual learned state values to the operator."""
    h=hashlib.sha256()
    def add(v):
        if hasattr(v,'detach'):
            t=v.detach().cpu().contiguous()
            if t.numel()>65536:raise ValueError('Auxiliary audit refuses a model-sized tensor')
            h.update(str((str(t.dtype),tuple(t.shape))).encode());h.update(t.numpy().tobytes())
        elif type(v).__module__.startswith('numpy') and hasattr(v,'tobytes'):
            h.update(str((str(v.dtype),tuple(v.shape))).encode());h.update(v.tobytes())
        elif isinstance(v,dict):
            for key in sorted(v,key=str):h.update(str(key).encode());add(v[key])
        elif isinstance(v,(list,tuple)):
            h.update(type(v).__name__.encode())
            for item in v:add(item)
        else:h.update(json.dumps(v,sort_keys=True,allow_nan=False).encode())
    add(value);return h.hexdigest()


def load_aux(reference,mode,config_ref,config,fit_qids):
    if not 0<reference['bytes']<1024**2:raise ValueError('Only a small complete scalar auxiliary CP is allowed')
    path=n.checked(reference)
    if path.name!='aux-'+mode+'-complete.pt':raise ValueError('Unknown/partial scalar checkpoint refused')
    import torch
    state=torch.load(path,map_location='cpu',weights_only=False)
    if (state.get('schema')!='PREFIX_CONFIDENCE_E4_FULL_SCALAR_STATE_V1' or state.get('phase')!=PHASE
            or state.get('mode')!=mode or state.get('use_prefix')!=(mode=='prefix_full8')
            or state.get('holdout_assessed') is not False or state.get('external_opened') is not False
            or state.get('source_probe_optimizer_reset') is not False or state.get('source_probe_model_loaded') is not False):
        raise ValueError('Complete source-bound pre-holdout scalar state required')
    raw=state['raw']
    if raw.shape!=(8,) or raw.dtype!=torch.float64 or raw.device.type!='cpu' or not bool(torch.isfinite(raw).all()):
        raise ValueError('Eight finite CPU auxiliary scalars required')
    binding=state['source_binding']
    if binding.get('calibration_mode')!=mode or binding.get('availability_qualified') is not True or binding.get('config')!=config_ref:
        raise ValueError('Scalar source/config/mode binding differs')
    for key in ('features','groups','E4_predictions','split','source_checkpoint','source_EMA_sha256','protocol'):
        if binding[key]!=config[key]:raise ValueError('Scalar source reference differs: '+key)
    if binding['availability_receipt']!=config['extract_complete'] or state['fit_qids_sha256']!=n.digest(fit_qids):
        raise ValueError('Fit-only identity/availability binding differs')
    protocol=state['protocol']
    expected=dict(seed=9137,probability_clip=1e-6,scale_floor=1e-6,regularizer=.01,trajectory_weight=1.,
                  three_event_weight=1.,views_per_G10=252,question_weight=1.,max_iterations=40,max_evaluations=60,
                  whole_two_mode_fit_deadline_seconds=1560,matched_control_is_external_candidate=False)
    if (any(protocol.get(k)!=v for k,v in expected.items()) or tuple(protocol['feature_names'])!=FEATURES
            or tuple(protocol['modes'])!=MODES[::-1] or tuple(protocol['holdout_response_indices'])!=(5,6,7,8,9)):
        raise ValueError('Fixed math protocol differs')
    report=state['fit'];opt=state['optimizer']
    if (report['qualification_only'] is not False or report['mode']!=mode or report['use_prefix']!=(mode=='prefix_full8')
            or not 0<report['closure_calls']<=60 or not 0<report['optimizer_iterations']<=40
            or len(report['closure_loss_curve'])!=report['closure_calls']):raise ValueError('Incomplete/changed scalar fitting budget')
    for i,row in enumerate(report['closure_loss_curve'],1):
        if row['evaluation']!=i or any(not math.isfinite(row[k]) for k in ('trajectory','three_event','regularization','total')):
            raise ValueError('Invalid full scalar loss curve')
    if len(opt['param_groups'])!=1 or len(opt['param_groups'][0]['params'])!=1 or len(opt['state'])!=1:
        raise ValueError('Single auxiliary optimizer state required')
    optstate=next(iter(opt['state'].values()))
    if optstate['n_iter']!=report['optimizer_iterations'] or optstate['func_evals']!=report['closure_calls']:
        raise ValueError('LBFGS full-state counters differ')
    parameter=raw.clone().requires_grad_();restored=torch.optim.LBFGS([parameter]);restored.load_state_dict(opt)
    if fingerprint(restored.state_dict())!=fingerprint(opt):raise ValueError('Full scalar LBFGS CPU reload differs')
    # Exercise all three auxiliary RNG restore interfaces in this isolated CPU
    # process; no original probe or running worker RNG is touched.
    import numpy as np
    import random
    current=(torch.get_rng_state(),np.random.get_state(),random.getstate())
    try:
        torch.set_rng_state(state['rng']['torch']);np.random.set_state(state['rng']['numpy']);random.setstate(state['rng']['python'])
        if fingerprint(dict(torch=torch.get_rng_state(),numpy=np.random.get_state(),python=random.getstate()))!=fingerprint(state['rng']):
            raise ValueError('Auxiliary RNG full-state reload differs')
    finally:torch.set_rng_state(current[0]);np.random.set_state(current[1]);random.setstate(current[2])
    small=io.BytesIO();torch.save(state,small);small.seek(0);again=torch.load(small,map_location='cpu',weights_only=False)
    if fingerprint(state)!=fingerprint(again):raise ValueError('Full scalar state serialization differs')
    if mode=='matched_platt2' and (state['normalizer'] is not None or bool((raw[2:]!=0).any())):
        raise ValueError('Matched2 must retain zero w and no prefix normalizer')
    return state,dict(checkpoint=reference,mode=mode,full_CP_cpu_reload=True,LBFGS_cpu_reload=True,RNG_cpu_reload=True,
        optimizer_iterations=report['optimizer_iterations'],closure_calls=report['closure_calls'],
        component_sha256={k:fingerprint(state[k]) for k in ('raw','optimizer','rng','normalizer','source_binding')})


def run(config_ref,source_publication_ref):
    config_path=n.checked(config_ref);config=n.read(config_path);fit=config_path.parent/'fit'
    if config_path.parent!=BASE or config_path.name!='fit-config-v2.json':raise ValueError('Exact current v2 calibration target required')
    if not (fit/'internal-complete.json').exists():return dict(status='WAITING_FOR_COMPLETE_INTERNAL_STATES_NO_CP_OPENED',external_opening_permitted=False)
    if (fit/'failure.json').exists() or (fit/'external-open-intent.json').exists():raise ValueError('Failure or prior external intent requires explicit inspection')
    internal_ref=n.proof(fit/'internal-complete.json');done=n.read(internal_ref['path'])
    if (done.get('status') not in ('PREFIX_CONFIDENCE_V2_INTERNAL_COMPLETE_AWAITING_INDEPENDENT_AUDIT','PREFIX_CONFIDENCE_V2_SINGLE_INFORMATION_TEST_COMPLETE')
            or done['config']!=config_ref or done.get('external_payload_opened') is not False
            or done.get('source_probe_model_or_Adam_loaded') is not False or done.get('source_probe_updates')!=0
            or done.get('actor_updates')!=0 or done.get('GPU_used') is not False or done.get('final_read') is not False):
        raise ValueError('Unknown/partial internal stage or forbidden model/external operation')
    if config['protocol_git_commit']!=PROTOCOL_COMMIT or config['protocol']['sha256']!=PROTOCOL_SHA:
        raise ValueError('Preregistered a1ec v2 protocol differs')
    n.checked(config['protocol']);pub=n.read(n.checked(source_publication_ref))
    if pub['commit']!=config['source_commit'] or pub['code_root']!=config['code_root']:
        raise ValueError('Actual immutable fit source/config differs')
    math_file='reproduction/operations/qwen7b_warmup/prefix_confidence_calibration.py'
    if pub['files'][math_file]['sha256']!=MATH_SHA:raise ValueError('Qualified two-mode math source differs')
    for name,ref in pub['files'].items():
        path=Path(pub['code_root'])/name
        if n.proof(path)['sha256']!=ref['sha256']:raise ValueError('Immutable fit source changed')
    accepted=n.read(n.checked(config['extract_complete']))
    if (accepted['availability_qualified'] is not True or accepted['all_own_ID_EM_order_native_LP_and_P20_exact'] is not True
            or accepted['questions']!=8192 or accepted['trajectories']!=81920):raise ValueError('Real complete extraction gate missing')
    for key in ('features','groups','split','E4_predictions','source_checkpoint','source_EMA_sha256'):
        if accepted[key]!=config[key]:raise ValueError('Accepted extraction source mismatch')
    truth=n.read(n.checked(config['groups']));truth=truth['groups'] if isinstance(truth,dict) else truth
    qids=[g['qid'] for g in truth];seen=set()
    if len(qids)!=8192 or len(set(qids))!=8192:raise ValueError('Frozen8192 unique question cohort required')
    for g in truth:
        if len(g['trajectory_ids'])!=10 or len(set(g['trajectory_ids']))!=10 or seen.intersection(g['trajectory_ids']) or len(g['labels'])!=10 or any(type(v) is not int or v not in (0,1) for v in g['labels']):
            raise ValueError('Authentic own G10 identity/strict EM required')
        seen.update(g['trajectory_ids'])
    fit_qids,held_qids=split(qids);held=set(held_qids);division=n.read(n.checked(config['split']))
    if (division['fit_qids']!=fit_qids or division['held_qids']!=held_qids or len(fit_qids)!=6144 or len(held_qids)!=2048
            or division['seed']!=9137 or division['labels_or_scores_used'] is not False
            or division['fit_ordered_qid_sha256']!=n.digest(fit_qids) or division['held_ordered_qid_sha256']!=n.digest(held_qids)):
        raise ValueError('Independent qid-only split differs')
    feature_rows=read_gzip(config['features']);columns=[[] for _ in FEATURES];held_features={}
    if len(feature_rows)!=8192:raise ValueError('Incomplete compact prefix feature file')
    for g,row in zip(truth,feature_rows):
        if row['qid']!=g['qid'] or row['trajectory_ids']!=g['trajectory_ids'] or set(row)!=set(('qid','trajectory_ids','features')) or len(row['features'])!=10:
            raise ValueError('Compact feature own-ID/order differs')
        for f in row['features']:
            if len(f)!=6 or any(not math.isfinite(v) for v in f) or f[1]<0 or not 0<f[5]<=1:raise ValueError('Invalid compact feature row')
        if g['qid'] in held:held_features[g['qid']]=row['features'][5:]
        else:
            for f in row['features']:
                for j,v in enumerate(f):columns[j].append(v)
    means=[math.fsum(v)/len(v) for v in columns]
    stds=[math.sqrt(math.fsum((x-mean)**2 for x in values)/len(values)) for values,mean in zip(columns,means)]
    scales=[max(v,1e-6) for v in stds]
    source=read_gzip(config['E4_predictions']);raw_rows=[];old_prob={};gold={}
    if len(source)!=8192:raise ValueError('Incomplete E4 original probability file')
    for g,row in zip(truth,source):
        if row['qid']!=g['qid'] or row['trajectory_ids']!=g['trajectory_ids'] or row['strict_em']!=g['labels'] or len(row['p_j'])!=10:
            raise ValueError('Source E4 own G10 EM/IDs differ')
        ps=row['p_j'];zs=row['logits']
        if len(zs)!=10 or any(not math.isfinite(z) for z in zs) or any(not math.isfinite(p) or not 0<=p<=1 for p in ps) or max(abs(sigmoid(z)-p) for z,p in zip(zs,ps))>1e-6:
            raise ValueError('Source E4 finite probabilities/logits differ')
        if g['qid'] not in held:continue
        one=dict(qid=g['qid'],trajectory_ids=g['trajectory_ids'][5:],strict_em=g['labels'][5:],score=n.qscore(ps[5:]))
        raw_rows.append(one);old_prob[g['qid']]=ps[5:];gold[g['qid']]=one
    # Match the declared observable numeric interface exactly; all statistics
    # and rankings below are independently computed with the stdlib evaluator.
    import numpy as np
    baseline_p=np.asarray([old_prob[r['qid']] for r in raw_rows],dtype=np.float64)
    baseline_q=np.clip(1-baseline_p.prod(1)-(1-baseline_p).prod(1),0,1)
    for row,q in zip(raw_rows,baseline_q):row['score']=float(q)
    results={'rawE4':n.summary(raw_rows)};states={};cp_audits={};pred_refs={};normalizer_error=0.
    for mode in MODES:
        cp=done['full_aux_checkpoints'][mode];state,cp_audits[mode]=load_aux(cp,mode,config_ref,config,fit_qids);states[mode]=state
        norm=state['normalizer']
        if mode=='prefix_full8':
            if norm['fit_qids_sha256']!=n.digest(fit_qids) or norm['fitting_questions']!=6144 or norm['fitting_trajectories']!=61440 or tuple(norm['feature_names'])!=FEATURES or norm['fitted_on_fit_rows_only'] is not True:
                raise ValueError('Normalizer fit-only cohort differs')
            if any(tuple(norm[key].shape)!=(6,) for key in ('mean','population_std','scale')):raise ValueError('Normalizer must contain exactly six statistics')
            errors=[abs(a-b) for key,values in [('mean',means),('population_std',stds),('scale',scales)] for a,b in zip(norm[key].tolist(),values)]
            normalizer_error=max(errors)
            if normalizer_error>1e-10:raise ValueError('Independent fit-only population normalizer differs')
        rec=n.read(fit/('internal-'+mode+'.json'));pred_ref=rec['predictions'];pred_refs[mode]=pred_ref
        if rec['checkpoint']!=cp or rec['phase']!='CP_SAVED_AND_RELOADED_BEFORE_THIS_INTERNAL_ASSESSMENT' or rec['external_payload_opened'] is not False:
            raise ValueError('Internal assessment/save phase differs')
        rows=read_gzip(pred_ref);checked=[];raw=state['raw'].tolist();a=math.exp(3*math.tanh(raw[0]));b=10*math.tanh(raw[1]);max_p_error=max_q_error=0.
        if len(rows)!=2048:raise ValueError('Internal heldout2048 complete predictions required')
        for qid,row in zip(held_qids,rows):
            g=gold[qid]
            if any(row[k]!=g[k] for k in ('qid','trajectory_ids','strict_em')):raise ValueError('Internal own newseed5 EM/IDs differ')
            expected=[]
            for p,f in zip(old_prob[qid],held_features[qid]):
                p=min(1-1e-6,max(1e-6,p));z=a*(math.log(p)-math.log1p(-p))+b
                if mode=='prefix_full8':z+=math.fsum(w*(v-mean)/scale for w,v,mean,scale in zip(raw[2:],f,means,scales))
                expected.append(sigmoid(z))
            if len(row['p_j'])!=5 or any(not math.isfinite(p) or not 0<=p<=1 for p in row['p_j']) or not math.isfinite(row['q']) or not 0<=row['q']<=1:raise ValueError('Finite internal G5 probabilities required')
            max_p_error=max(max_p_error,max(abs(x-y) for x,y in zip(expected,row['p_j'])))
            max_q_error=max(max_q_error,abs(n.qscore(row['p_j'])-row['q']))
            checked.append(dict(g,score=row['q']))
        if max_p_error>1e-9 or max_q_error>1e-11:raise ValueError('Independent scalar inference/probability/q differs')
        result=n.summary(checked);compare_metrics(result,rec['metrics']);compare_metrics(result,done['internal_metrics'][mode]);results[mode]=result
        reload=n.read(fit/('checkpoint-'+mode+'-reload-audit.json'))
        if reload['checkpoint']!=cp or reload['full_raw_optimizer_RNG_normalizer_sources_phase_reload'] is not True or not Path(cp['path']).stat().st_mtime<=reload['time']<=Path(pred_ref['path']).stat().st_mtime:
            raise ValueError('Durable complete CP/reload must precede internal prediction file')
        cp_audits[mode].update(predictions=pred_ref,p_probability_max_absolute_error=max_p_error,q_formula_max_absolute_error=max_q_error,
                              complete_CP_and_reload_before_internal_prediction_file=True)
    compare_metrics(results['rawE4'],done['internal_raw_and_clipping']['original_raw'])
    cp=np.clip(baseline_p,1e-6,1-1e-6);cz=np.log(cp)-np.log1p(-cp);identity_p=1/(1+np.exp(-cz))
    cq=np.clip(1-identity_p.prod(1)-(1-identity_p).prod(1),0,1)
    clipped=[dict(row,score=float(q)) for row,q in zip(raw_rows,cq)]
    results['clipped_identity']=n.summary(clipped);compare_metrics(results['clipped_identity'],done['internal_raw_and_clipping']['clipped_identity'])
    clipping_flips=sum((a['score']>=.5)!=(b['score']>=.5) for a,b in zip(raw_rows,clipped))
    if clipping_flips!=done['internal_raw_and_clipping']['threshold_flips']:raise ValueError('Clipped identity threshold flip count differs')
    decision=gate(results['rawE4'],results['matched_platt2'],results['prefix_full8'])
    if decision!=done['internal_gate']:raise ValueError('Independent pre-external joint gate differs')
    freeze=n.read(n.checked(done['frozen_candidate']))
    if freeze['checkpoint']!=done['full_aux_checkpoints']['prefix_full8'] or freeze['control_checkpoint']!=done['full_aux_checkpoints']['matched_platt2'] or freeze['internal_gate']!=decision or freeze['external_payload_opened'] is not False:
        raise ValueError('Frozen exact auxiliary checkpoints/gate differs')
    original=Path(config['source_checkpoint']['path']).stat()
    if [original.st_dev,original.st_ino,original.st_size,original.st_mtime_ns,original.st_ctime_ns]!=freeze['original_probe_file_stamp']:
        raise ValueError('Original probe metadata changed; no model payload opened')
    output=dict(status=STATUS,config=config_ref,internal_complete=internal_ref,source_publication=source_publication_ref,
        full_aux_checkpoints=done['full_aux_checkpoints'],auxiliary_audits=cp_audits,internal_predictions=pred_refs,
        internal_metrics=results,internal_gate=decision,external_opening_permitted=decision['passed'],
        split=dict(seed=9137,fit_questions=6144,heldout_questions=2048,overlap=0,fit_qids_sha256=n.digest(fit_qids),held_qids_sha256=n.digest(held_qids),qid_only=True),
        normalizer_fit_only_population_std_recomputed=True,normalizer_max_absolute_difference=normalizer_error,
        raw_to_clipped_identity_threshold_flips=clipping_flips,original_probe_file_stat_verified=True,
        raw_to_clipped_identity_order=clipping_order_audit(raw_rows,clipped),
        all_own_newseed5_IDs_EM_verified=True,threshold=.5,Top512_rule='(-stored_q,str(qid))',
        original_model_or_optimizer_payload_opened=False,real_scalar_values_returned=False,
        external_predictions_or_final_opened=False,CUDA_used=False,epoch=time.time())
    target=fit/'independent-internal-audit.json'
    if target.exists():raise ValueError('Existing independent audit preserved')
    n.atomic(target,output);return output


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',required=True);parser.add_argument('--config-sha256',required=True)
    parser.add_argument('--source-publication',required=True);parser.add_argument('--source-publication-sha256',required=True)
    parser.add_argument('--internal-complete-sha256',required=True);args=parser.parse_args()
    config=n.proof(args.config);source=n.proof(args.source_publication)
    if config['sha256']!=args.config_sha256 or source['sha256']!=args.source_publication_sha256:raise ValueError('Actual v2 source/config binding differs')
    if n.proof(Path(args.config).parent/'fit/internal-complete.json')['sha256']!=args.internal_complete_sha256:raise ValueError('Actual known complete internal stage changed')
    print(json.dumps(run(config,source)),flush=True)
