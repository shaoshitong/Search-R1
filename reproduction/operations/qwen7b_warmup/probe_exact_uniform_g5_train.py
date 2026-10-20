"""Explicit all-combination G10 entry; no scheduler, default-runner hotpatch.

The loss also supports authentic G20. This initial production-capable *entry*
accepts only the existing G10 matched6528 B48x136/E1/fullstate protocol; G20
loader/runner integration requires a separately explicit prepared deployment.
Never claim GPU qualification from the CPU mathematical/toy tests.
"""
import argparse,copy,fcntl,time,traceback
from pathlib import Path
import torch
from . import probe_g10_train as inherited
from . import probe_ablation_train as u
from .exact_uniform_g5_loss import exact_uniform_g5_loss,validate_source,weights as component_weights

MODE='exact-uniform-all-real-G5'
RECIPE='g10-matched6528-exact-uniform-all-G5'
REQUIRED=('exact_uniform_g5_loss.py','probe_exact_uniform_g5_train.py','g20_grouping.py')


def bounded_exact_update(model,ema,optimizer,reader,groups,device,*,loss_kwargs,
        combination_chunk=64,micro_trajectories=5,ema_decay=.99):
    """Unchanged two-pass native forward count, one bounded exact logit VJP."""
    size=validate_source(groups);component_weights(**loss_kwargs)
    if type(micro_trajectories) is not int or micro_trajectories<1 or not 0<=float(ema_decay)<1:raise ValueError('Explicit positive micro/valid EMA required')
    if any(g['lr']!=5e-6 or g['weight_decay']!=0 for g in optimizer.param_groups):raise ValueError('Inherited Adam LR/weight decay must remain unchanged')
    if {'logits','source_groups','combination_chunk'}&set(loss_kwargs):raise ValueError('Recipe may not inject identities or chunk controls')
    before=u.steps(optimizer);started=time.monotonic();model.train();optimizer.zero_grad(set_to_none=True)
    tids=[t for g in groups for t in g['trajectory_ids']];micros=[];parts=[];native_bytes=0
    for start in range(0,len(tids),micro_trajectories):
        ids=tids[start:start+micro_trajectories];data=reader.get(ids,device);native=u.buffers(model);native_bytes+=sum(v.numel()*v.element_size() for v in native.values())
        if native_bytes>256*1024**2:raise ValueError('Native replay snapshots exceed256MiB budget')
        micros.append((start,ids,u.rng_state(device),native));out=inherited._forward(model,data,device)
        if out.numel()!=len(ids) or not bool(torch.isfinite(out).all()):raise ValueError('Incomplete/nonfinite actual trajectory logits')
        parts.append(out.detach().clone());del out,data
    after_rng=u.rng_state(device);after_native=u.buffers(model);logits=torch.cat(parts).reshape(len(groups),size).requires_grad_(True);del parts
    result=exact_uniform_g5_loss(logits,groups,combination_chunk=combination_chunk,**loss_kwargs);derivative=torch.autograd.grad(result['loss'],logits)[0].detach().reshape(-1);expected=logits.detach().reshape(-1)
    if not bool(torch.isfinite(derivative).all()):raise ValueError('Nonfinite exact uniform dL/dlogits')
    for start,ids,rng,native in micros:
        data=reader.get(ids,device);u.restore_rng(rng,device);u.restore_buffers(model,native);out=inherited._forward(model,data,device)
        if not torch.equal(out.detach(),expected[start:start+len(ids)]):raise ValueError('Two-pass actual micro logits differ; no Adam update accepted')
        torch.autograd.backward(out,derivative[start:start+len(ids)].to(out.dtype));del out,data
    u.restore_rng(after_rng,device);u.restore_buffers(model,after_native)
    params=list(model.parameters())
    if not params or not all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in params):raise ValueError('Missing/nonfinite exact uniform parameter gradient')
    optimizer.step()
    if u.steps(optimizer)!=before+1:raise ValueError('One complete original-question Adam update required')
    with torch.no_grad():
        for target,value in zip(ema.parameters(),params):target.mul_(float(ema_decay)).add_(value,alpha=1-float(ema_decay))
    confusion=result['question_weighted_confusion'];tp,fn,fp,tn=[confusion[k] for k in ('tp','fn','fp','tn')];tpr=tp/(tp+fn) if tp+fn else None;tnr=tn/(tn+fp) if tn+fp else None
    return dict(loss=float(result['loss'].detach()),components={k:float(v) for k,v in result['components'].items()},
        grouping=dict(mode=MODE,source_group_size=size,combinations_per_question=result['combinations_per_question'],views_per_question=result['combinations_per_question'],view_weight=result['view_weight'],per_question_total_weight_exactly_one=True,all_combinations_exactly_once=True,weighted_mixed_fraction=result['weighted_Mixed_fraction'],source_groups_sha256=u.digest(groups)),
        original_questions=len(groups),unique_real_trajectories=len(tids),virtual_views=result['virtual_views'],view_trajectory_exposures=5*result['virtual_views'],
        actual_forward_trajectory_exposures=2*len(tids),forward_passes=2,backward_passes=1,micro_trajectories=micro_trajectories,
        maximum_retained_transformer_graph_microbatches=1,native_replay_snapshot_bytes=native_bytes,
        maximum_resident_G5_rows=result['maximum_resident_G5_rows'],maximum_resident_pattern_rows=result['maximum_resident_pattern_rows'],
        retained_loss_gradient_elements=result['retained_loss_gradient_elements'],combination_chunk=combination_chunk,
        gradient_strategy='two_pass_same_micro_RNG_native_buffer_plus_exact_all_G5_chunked_first_order_logit_VJP',
        recomputed_logits_bit_identical=True,monolithic_bitexact_claim=False,first_order_only=True,
        inter_view_pairs_allocated=0,per_view_python_identity_metadata_allocated=0,
        denominators=result['denominators'],question_weighted_confusion=confusion,
        mixed_recall=tpr,nonmixed_recall=tnr,mixed_balanced_accuracy=(tpr+tnr)/2 if tpr is not None and tnr is not None else None,
        seconds=time.monotonic()-started,actor_updates=0,supports_G20=size==20)


def update_callback(model,ema,optimizer,reader,lookup,qids,config,device,epoch):
    grouping=config['grouping']
    if grouping!={'mode':MODE,'source_group_size':10,'combination_chunk':64} or epoch!=0:raise ValueError('Explicit exact-uniform G10 matched epoch0 recipe required')
    return bounded_exact_update(model,ema,optimizer,reader,[lookup[q] for q in qids],device,
        loss_kwargs=config['recipe']['loss_kwargs'],combination_chunk=grouping['combination_chunk'],micro_trajectories=config['micro_trajectories'])


def make_config(reference,plan,*,hashes,commit,launch_authorized=False):
    """Copy authentic reference assets/order. No new bank, trajectory or label."""
    if (reference['source_checkpoint']['sha256']!=u.SOURCE_SHA or reference['global_questions']!=48 or reference['update_budget']!=136 or reference['fit_questions']!=6528 or reference['fit_real_trajectories']!=65280 or reference.get('grouping',{}).get('mode')!='random' or len(commit)!=40):raise ValueError('Existing sealed E1/G10/B48x136 reference configuration required')
    used=[q for batch in plan['batches'] for q in batch]
    if len(plan['batches'])!=136 or plan['batch_epochs']!=[0]*136 or any(len(b)!=48 for b in plan['batches']) or len(set(used))!=6528 or len(used)!=6528 or set(used)!=set(plan['cohort_qids']):raise ValueError('Preserve all actual136 G10 batches exactly')
    if not {'reproduction/operations/qwen7b_warmup/'+n for n in REQUIRED}<=set(hashes):raise ValueError('New immutable exact loss/entry/G20-validation bindings required')
    c=copy.deepcopy(reference);c.update(grouping=dict(mode=MODE,source_group_size=10,combination_chunk=64),recipe=dict(id=RECIPE,loss_kwargs=dict(trajectory_weight=1.,three_event_weight=1.)),code_hashes=copy.deepcopy(hashes),source_commit=commit,
        exact_uniform_all_G5=True,source_group_size=10,combinations_per_question=252,original_question_total_weight=1.,view_weight_numerator=1,view_weight_denominator=252,
        production_replacement_authorized=False,launch_requires_explicit_parent_go_after_G20_final=True,
        no_extra_native_model_forwards_relative_to_random4=True,first_order_loss_VJP_only=True,
        explicit_launch_authorized_after_G20_final=bool(launch_authorized),
        maximum_resident_G5_rows=48*64,maximum_resident_pattern_rows=48*64*32,inter_view_pairs_allocated=0,
        inherited_reference_plan_unchanged=True,inherited_reference_plan=copy.deepcopy(reference['plan']))
    c.pop('max_virtual_views',None);c.pop('max_inter_view_pairs',None)
    return c


def run(base):
    c=u.read(base/'config.json')
    if c['recipe']!={'id':RECIPE,'loss_kwargs':dict(trajectory_weight=1.,three_event_weight=1.)} or c['grouping']!={'mode':MODE,'source_group_size':10,'combination_chunk':64} or c['global_questions']!=48 or c['update_budget']!=136 or c['fit_questions']!=6528 or c['fit_real_trajectories']!=65280 or c['plan']!=c['inherited_reference_plan']:raise ValueError('Explicit same-source/order/budget exact-uniform experiment required')
    if not c.get('explicit_launch_authorized_after_G20_final',False):raise ValueError('No new experiment launch until root explicitly authorizes after G20 evidence')
    if not {'reproduction/operations/qwen7b_warmup/'+n for n in REQUIRED}<=set(c['code_hashes']):raise ValueError('Immutable exact uniform sources required')
    inherited.run(base,update_callback=update_callback)


def main():
    p=argparse.ArgumentParser();p.add_argument('--action',choices=('prepare','train'),required=True);p.add_argument('--base',type=Path,required=True);p.add_argument('--reference-config',type=Path);p.add_argument('--code-root',type=Path);p.add_argument('--explicit-go-after-G20-final',action='store_true');a=p.parse_args();base=a.base
    if not base.is_absolute() or not str(base).startswith('/data/user/jxu957/experiments/') or base.resolve()!=base:raise ValueError('Independent remote shared output required')
    base.mkdir(parents=True,exist_ok=True)
    with (base/'exact-g5-train.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if a.action=='prepare':
            if not a.reference_config or not a.code_root or a.reference_config.parent.resolve()==base.resolve():raise ValueError('Separate authentic reference and published immutable code root required')
            reference=u.read(a.reference_config);plan=u.read(u.checked(reference['plan']));pub=u.read(a.code_root.parent/'source-publication.json')
            if Path(pub['code_root'])!=a.code_root:raise ValueError('Publication root differs')
            hashes={n:v['sha256'] for n,v in pub['files'].items()}
            for n,h in hashes.items():
                if u.sha(a.code_root/n)!=h:raise ValueError('Unpublished or changed runtime source')
            c=make_config(reference,plan,hashes=hashes,commit=pub.get('commit',pub.get('source_commit')),launch_authorized=a.explicit_go_after_G20_final);c['reference_config']=dict(path=str(a.reference_config),bytes=a.reference_config.stat().st_size,sha256=u.sha(a.reference_config))
            path=base/'config.json'
            if path.exists():
                if u.read(path)!=c:raise ValueError('Existing prepared experiment preserved')
            else:u.atomic(path,c)
            print(__import__('json').dumps(dict(status='EXACT_UNIFORM_G5_SAME_E1_G10_ORDER_BUDGET_PREPARED_NOT_LAUNCHED',config_sha256=u.sha(path),GPU_requested=0)))
        else:
            try:run(base)
            except BaseException as e:u.atomic(base/'failure.json',dict(type=type(e).__name__,message=str(e),traceback=traceback.format_exc(),time=time.time()));raise
if __name__=='__main__':main()
