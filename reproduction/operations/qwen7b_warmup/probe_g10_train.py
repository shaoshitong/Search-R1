"""Isolated real-G10 question-weighted probe runner; never submits jobs.

Reuse sealed E1 raw/EMA/block/readout/Adam10464 without resetting parameters.
Two-pass logit VJP retains only ONE microforward graph, not all Q*10 graphs.
One extra forward pass is an explicit compute cost; the loss/gradient objective
is unchanged. Microaccumulation may change floating summation order, so no
four-rank or monolithic bit-exact claim is made. Real GPU qualification is still
required. Existing G5/active runners do not import this module.
"""
import argparse
import copy
import fcntl
import os
from pathlib import Path
import signal
import time
import traceback
from contextlib import nullcontext
import torch
from . import probe_ablation_train as u
from .g10_grouping import _validate, build_training_views, fixed_g5
from .g10_weighted_balanced import build_training_views as build_global_training_views
from .g10_question_loss import g10_question_loss
# CPU reference for tests/qualification only; production uses bounded VJP below.
from .g10_question_update import question_weighted_update as joined_graph_reference

STOP=False
MERGED='PILOT_ACTUAL_ORIGINAL5_PLUS_EXTRA5_G10_SHA_OWN_EM_VERIFIED'


def _forward(model,data,device):
    context=torch.autocast('cuda',dtype=torch.bfloat16) if device.type=='cuda' else nullcontext()
    with context:return model(data).reshape(-1)


def input_pair_mask(reader,groups,device):
    """Only opposite-EM pairs whose actual packed model inputs differ qualify.

    Retained BF16 feature hash plus lengths/positions binds the complete reader
    input; this does not claim that a different input is necessarily learnable.
    """
    if not hasattr(reader,'records'):raise ValueError('Authentic input index required for ranking')
    result=[]
    for g in groups:
        signatures=[]
        for tid in g['trajectory_ids']:
            _,row=reader.records[tid]
            signatures.append(u.digest({k:row[k] for k in
                ('retained_feature_sha256','question_length','policy_length','policy_positions')}))
        result.append([[bool(g['labels'][i]==1 and g['labels'][j]==0 and
                              signatures[i]!=signatures[j]) for j in range(10)] for i in range(10)])
    return torch.tensor(result,dtype=torch.bool,device=device)


def bounded_question_update(model,ema,optimizer,reader,groups,device,*,
        grouping_kwargs,loss_kwargs,micro_trajectories=5,
        distinguishable_pair_mask=None,ema_decay=.99,max_virtual_views=2048,
        max_inter_view_pairs=1000000,grouping_builder=build_training_views):
    """Same complete-question loss, exact chain rule, bounded activation graphs.

    Pass1 uses the SAME grad-enabled training forward as pass2, but detaches each
    result immediately. Per-micro RNG and nonpersistent native buffers are
    replayed exactly; recomputed logits must be bit-identical before any Adam
    write. Failure never accepts a partial optimizer update. Parameter-gradient
    accumulation roundoff is not claimed bit-identical to joined-graph backward.
    """
    _validate(groups,require_labels=True)
    if type(micro_trajectories) is not int or micro_trajectories<1:
        raise ValueError('Positive explicit micro trajectory budget required')
    if not 0<=float(ema_decay)<1:raise ValueError('EMA decay must be in [0,1)')
    if {'logits','source_groups','views','distinguishable_pair_mask'}&set(loss_kwargs):
        raise ValueError('Recipe cannot inject identities, labels or masks')
    grouping=grouping_builder(groups,role='train',**grouping_kwargs)
    views=grouping['groups'];positive=sum(v['true_mixed'] for v in views)
    if len(views)>max_virtual_views or positive*(len(views)-positive)>max_inter_view_pairs:
        raise ValueError('Explicit virtual-view/loss-pair memory budget exceeded')
    if any(group['lr']!=5e-6 or group['weight_decay']!=0 for group in optimizer.param_groups):
        raise ValueError('Inherited Adam LR/weight decay must remain unchanged before update')
    before=u.steps(optimizer);started=time.monotonic();model.train()
    optimizer.zero_grad(set_to_none=True)
    tids=[tid for g in groups for tid in g['trajectory_ids']]
    micros=[];parts=[];native_replay_bytes=0
    for start in range(0,len(tids),micro_trajectories):
        ids=tids[start:start+micro_trajectories]
        # FeatureReader performs no random sampling. Capture state immediately
        # before actual forward, after any reader work.
        data=reader.get(ids,device)
        native=u.buffers(model)
        native_replay_bytes+=sum(v.numel()*v.element_size() for v in native.values())
        if native_replay_bytes>256*1024**2:
            raise ValueError('Native replay snapshots exceed explicit 256MiB CPU-state budget')
        micros.append((start,ids,u.rng_state(device),native))
        out=_forward(model,data,device)
        if out.numel()!=len(ids) or not bool(torch.isfinite(out).all()):
            raise ValueError('Nonfinite or incomplete actual trajectory logits')
        parts.append(out.detach().clone())
        del out,data  # releases this micro's entire transformer activation graph
    after_forward_rng=u.rng_state(device);after_forward_buffers=u.buffers(model)
    logits=torch.cat(parts).reshape(len(groups),10).requires_grad_(True)
    del parts
    result=g10_question_loss(logits,groups,views,
        distinguishable_pair_mask=distinguishable_pair_mask,**loss_kwargs)
    if not bool(torch.isfinite(result['loss'])):raise ValueError('Nonfinite complete question loss')
    derivative=torch.autograd.grad(result['loss'],logits)[0].detach().reshape(-1)
    if not bool(torch.isfinite(derivative).all()):raise ValueError('Nonfinite dL/dlogits')
    expected=logits.detach().reshape(-1)
    for start,ids,rng,native in micros:
        data=reader.get(ids,device);u.restore_rng(rng,device);u.restore_buffers(model,native)
        out=_forward(model,data,device)
        if not torch.equal(out.detach(),expected[start:start+len(ids)]):
            raise ValueError('Two-pass actual micro logits differ; no Adam update accepted')
        torch.autograd.backward(out,derivative[start:start+len(ids)].to(out.dtype))
        del out,data
    # The second forward is recomputation, not a second RNG/native-buffer clock.
    u.restore_rng(after_forward_rng,device);u.restore_buffers(model,after_forward_buffers)
    parameters=list(model.parameters())
    if not parameters or not all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in parameters):
        raise ValueError('Missing/nonfinite complete-question parameter gradient')
    optimizer.step()
    if u.steps(optimizer)!=before+1:raise ValueError('Adam must advance exactly once')
    with torch.no_grad():
        for target,value in zip(ema.parameters(),parameters):
            target.mul_(float(ema_decay)).add_(value,alpha=1-float(ema_decay))
    pred=result['mixed_probability'].detach()>=.5;truth=result['mixed_label'].detach().bool()
    w=result['view_weights'].detach()
    tp=float(w[pred&truth].sum());fn=float(w[~pred&truth].sum())
    fp=float(w[pred&~truth].sum());tn=float(w[~pred&~truth].sum())
    tpr=tp/(tp+fn) if tp+fn else None;tnr=tn/(tn+fp) if tn+fp else None
    return dict(loss=float(result['loss'].detach()),
        components={k:float(v.detach()) for k,v in result['components'].items()},
        grouping=grouping['summary'],original_questions=len(groups),
        unique_real_trajectories=len(tids),virtual_views=len(views),
        view_trajectory_exposures=5*len(views),actual_forward_trajectory_exposures=2*len(tids),
        forward_passes=2,backward_passes=1,micro_trajectories=micro_trajectories,
        maximum_retained_transformer_graph_microbatches=1,
        native_replay_snapshot_bytes=native_replay_bytes,
        gradient_strategy='two_pass_same_micro_RNG_native_buffer_logit_VJP',
        recomputed_logits_bit_identical=True,monolithic_bitexact_claim=False,
        denominators=result['denominators'],
        inter_eligible_weighted_pair_mass_exact=result['inter_eligible_weighted_pair_mass_exact'],
        question_weighted_confusion=dict(tp=tp,fp=fp,fn=fn,tn=tn),
        mixed_recall=tpr,nonmixed_recall=tnr,
        mixed_balanced_accuracy=(tpr+tnr)/2 if tpr is not None and tnr is not None else None,
        seconds=time.monotonic()-started,actor_updates=0,supports_G20=False)


def accepted_g10(reference):
    """Load SHA-bound accepted original5+extra5 and rejoin own EM/IDs.

    The merge acceptance already audited actual full/prefix/seed journals. Here
    its immutable shard receipts and both label banks are rechecked. Native
    feature bytes are losslessly checked by FeatureReader on every fetched row.
    """
    complete=u.read(u.checked(reference))
    if (complete['status']!=MERGED or complete['actor_updates']!=0 or complete['synthetic_EM']
            or complete['model_revision']!='a09a35458c702b33eeacc393d103063234e8bc28'
            or not complete['original_fixed5_preserved'] or complete['validation_or_final_questions_generated']):
        raise ValueError('Accepted training-only actual G10 merge required')
    groups=u.read(u.checked(complete['groups']))['train'];_validate(groups,True)
    if (complete['questions']!=len(groups) or complete['trajectories']!=10*len(groups)
            or complete['extra_real_trajectories']!=5*len(groups)):
        raise ValueError('Accepted G10 counts differ')
    old_rows=u.read(u.checked(complete['original_G5_bank']))['train']
    old={g['qid']:g for g in old_rows}
    if len(old)!=len(old_rows):raise ValueError('Repeated original-bank question identity')
    extra={}
    for ref in complete['extra_shards']:
        shard=u.read(u.checked(ref))
        if shard['status']!='EXTRA_REAL_G5_SHARD_NATIVE_FULL_OWN_EM_SHA_VERIFIED' or shard['actor_updates']!=0 or shard['synthetic_EM']:
            raise ValueError('Invalid accepted real extra-G5 shard')
        for g in shard['groups']:
            if g['qid'] in extra:raise ValueError('Extra question repeated between accepted shards')
            extra[g['qid']]=g
    for g in groups:
        a,b=old[g['qid']],extra[g['qid']]
        if (g['trajectory_ids']!=a['trajectory_ids']+b['trajectory_ids'] or
                g['labels']!=a['labels']+b['labels'] or
                g['original_fixed5_trajectory_ids']!=a['trajectory_ids'] or
                g['question_hash']!=a['question_hash'] or g['question_hash']!=b['question_hash']):
            raise ValueError('G10 IDs/own EM no longer match authentic old5+new5')
    if set(extra)!={g['qid'] for g in groups}:raise ValueError('Extra/G10 question coverage differs')
    if u.digest([g['qid'] for g in groups])!=complete['ordered_qid_sha256']:
        raise ValueError('Accepted ordered question hash differs')
    original=u.read(u.checked(complete['original_feature_manifest']))
    refs=u.read(u.checked(complete['extra_G5_feature_sources']))
    if not isinstance(original,list) or any(not isinstance(path,str) for path in original) or not isinstance(refs,list):
        raise ValueError('Independent original/extra G5 feature manifest lists required')
    return groups,list(original)+[str(u.checked(ref)) for ref in refs],complete


def native_reader(manifests):
    """Verify native L26/Prefix20 index metadata, defer bulk per-row bytes to get."""
    from .packed_features import FeatureReader
    for manifest in manifests:
        p=Path(manifest);m=u.read(p)
        if (m['status']!='LOSSLESS_NATIVE_FEATURES_COMPLETE' or m['feature_layer']!=26 or
                m['prefix_tokens']!=20 or m['dimension']!=3584 or m['question_max_tokens']!=273):
            raise ValueError('Require authentic native L26 question+Prefix20 features')
        # Bind identities and prototype, then get() checks reconstructed feature
        # hashes. Do not silently assume cached per-question inputs are identical.
        for key in ('index.jsonl','prototype.bin'):
            if u.checked(m['files'][key]).resolve()!=(p.parent/key).resolve():
                raise ValueError('Packed reader identity index/prototype path binding differs')
    return FeatureReader(manifests)


def validate_plan(plan,groups,config,forbidden=()):
    n=config['global_questions'];budget=config['update_budget']
    if type(n) is not int or not 1<=n<=51 or type(budget) is not int or budget<1:
        raise ValueError('Explicit original-question B1..51 and positive update budget required')
    if n!=51 and not config.get('smaller_question_budget_acknowledged',False):
        raise ValueError('Smaller original-Q batch must acknowledge it differs from matched B51')
    lookup={g['qid']:g for g in groups};cohort=plan['cohort_qids']
    if len(lookup)!=len(groups) or len(cohort)!=len(set(cohort)) or set(cohort)&set(forbidden):
        raise ValueError('Duplicate or external question in train cohort')
    if len(plan['batches'])!=budget or len(plan['batch_epochs'])!=budget:
        raise ValueError('Explicit update and grouping-epoch plan required')
    used=set()
    for batch,epoch in zip(plan['batches'],plan['batch_epochs']):
        if len(batch)!=n or len(set(batch))!=n or not set(batch)<=set(cohort):
            raise ValueError('Each global update must contain exactly declared distinct original questions')
        if type(epoch) is not int or epoch<0:raise ValueError('Nonnegative predeclared grouping epoch required')
        used.update(batch)
        if config.get('grouping',{}).get('mode')=='global-weighted-balanced':
            if (config['grouping'].get('weighting')!='equal-view-quota' or
                    config['grouping'].get('on_unbalance')!='raise'):
                raise ValueError('Global balanced experiment requires exact equal-view quotas, never silent fallback')
            if any(q not in lookup for q in batch):raise ValueError('Unknown training question')
            if 2*sum(0<sum(lookup[q]['labels'])<10 for q in batch)<len(batch):
                raise ValueError('Declared batch cannot attain1:1 with original-question weight1')
    if used!=set(cohort) or not used<=set(lookup):raise ValueError('Cohort/batch/G10 bank mismatch')
    return lookup


def frozen_validation(groups,plan):
    """Validate precommitted label-blind G5 IDs, then join their own EM.

    G10 validation never calls label-aware build_training_views. Existing G5
    diagnostics use their designated original IDs through this same interface.
    """
    plan=plan.get('fixed_G5',plan)
    if (plan['summary'].get('labels_consulted') is not False or
            plan['summary'].get('role')!='validation'):
        raise ValueError('Explicit label-blind frozen validation selection required')
    selected=plan['groups'];by={g['qid']:g for g in groups}
    if len(by)!=len(groups) or {g['qid'] for g in selected}!=set(by) or len(selected)!=len(by):
        raise ValueError('Frozen validation coverage/duplicate qid differs')
    out=[];seen=set()
    for v in selected:
        g=by[v['qid']];tids=v['trajectory_ids']
        if len(tids)!=5 or len(set(tids))!=5 or not set(tids)<=set(g['trajectory_ids']):
            raise ValueError('Frozen validation IDs must be five distinct own actual trajectories')
        if any(t in seen for t in tids):raise ValueError('Validation trajectory repeated across questions')
        seen.update(tids)
        labels=dict(zip(g['trajectory_ids'],g['labels']))
        if (len(g['trajectory_ids']) not in (5,10) or len(labels)!=len(g['trajectory_ids']) or
                len(g['labels'])!=len(g['trajectory_ids']) or
                any(type(y) is not int or y not in (0,1) for y in g['labels'])):
            raise ValueError('Actual G5/G10 validation bank required')
        out.append(dict(qid=g['qid'],trajectory_ids=list(tids),labels=[labels[t] for t in tids]))
    if u.digest(selected)!=plan['summary']['selected_g5_sha256']:
        raise ValueError('Frozen label-blind G5 identity hash differs')
    return out


def run_update(model,ema,optimizer,reader,lookup,qids,config,device,epoch):
    selected=[lookup[q] for q in qids];loss=config['recipe']['loss_kwargs']
    mask=input_pair_mask(reader,selected,device) if float(loss.get('within_rank_weight',0.)) else None
    grouping=dict(config['grouping']);grouping['epoch']=epoch
    builder=build_global_training_views if grouping['mode']=='global-weighted-balanced' else build_training_views
    return bounded_question_update(model,ema,optimizer,reader,selected,device,
        grouping_kwargs=grouping,loss_kwargs=loss,
        micro_trajectories=config['micro_trajectories'],distinguishable_pair_mask=mask,
        max_virtual_views=config['max_virtual_views'],max_inter_view_pairs=config['max_inter_view_pairs'],
        grouping_builder=builder)


def blob(model,ema,optimizer,device,cursor,binding,source_ranks):
    b=u.checkpoint_blob(model,ema,optimizer,device,cursor,binding,source_ranks)
    b.update(role='ISOLATED_REAL_G10_QUESTION_WEIGHTED_PROBE',phase='ACCEPTED_UPDATE_SAVED_BEFORE_EXTERNAL_EVALUATION',
             actor_updates=0,optimizer_reset=False,readout_reset=False,block_reset=False)
    return b


def run(base):
    c=u.read(base/'config.json');config_sha=u.sha(base/'config.json')
    if type(c['save_every']) is not int or c['save_every']<1:raise ValueError('Positive save cadence required')
    if c['source_checkpoint']['sha256']!=u.SOURCE_SHA:raise ValueError('Pinned sealed E1 required')
    required=('source_checkpoint','source_config','source_bank','model_config','g10_complete','plan')
    paths={k:u.checked(c[k]) for k in required}
    repo=Path(__file__).resolve().parents[3]
    code_required={'reproduction/operations/qwen7b_warmup/'+n for n in
        ('probe_g10_train.py','g10_question_loss.py','g10_question_update.py','g10_grouping.py','g10_weighted_balanced.py',
         'probe_ablation_train.py','probe_ablation_loss.py','model.py','packed_features.py','native_capture.py')}
    if not code_required<=set(c['code_hashes']):raise ValueError('Missing immutable runtime code bindings')
    for name,want in c['code_hashes'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts or u.sha(repo/name)!=want:
            raise ValueError('Immutable runner source mismatch: '+name)
    if int(os.environ.get('WORLD_SIZE','1'))!=1 or torch.cuda.device_count()!=1:
        raise ValueError('Isolated branch requires exactly one visible GPU')
    if os.environ.get('FLASH_ATTENTION_DETERMINISTIC')!='1' or os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':
        raise ValueError('Set deterministic FA/CUBLAS startup flags')
    os.environ['QWEN7B_QUESTION_MAX_TOKENS']='273';os.environ['SEARCH_G1_B_PREFIX_TOKENS']='20'
    device=torch.device('cuda',0);torch.cuda.set_device(device)
    props=torch.cuda.get_device_properties(device)
    if 'H100' not in props.name or props.total_memory<75*1024**3:raise ValueError('Dedicated >=75GiB H100 required')
    torch.use_deterministic_algorithms(True);torch.backends.cudnn.benchmark=False
    torch.backends.cuda.matmul.allow_tf32=False
    if (Path(c['model_dir'])/'config.json').resolve()!=paths['model_config'].resolve():
        raise ValueError('Hash-bound official model configuration required')
    groups,manifests,acceptance=accepted_g10(c['g10_complete']);plan=u.read(paths['plan'])
    evaluations=[];external=set();external_hashes=set();evaluation_names=set()
    for e in c['evaluations']:
        if (e['role']!='validation' or not e['name'].replace('_','').isalnum() or
                e['name'] in evaluation_names):
            raise ValueError('Only named recipe-selection validation is allowed in this fitting runner')
        bank=u.read(u.checked(e['bank']))[e['split']]
        evaluation_names.add(e['name'])
        frozen=frozen_validation(bank,u.read(u.checked(e['fixed_g5'])))
        ids={g['qid'] for g in frozen}
        if external&ids:raise ValueError('Evaluation pools overlap each other')
        hashes={g['question_hash'] for g in bank}
        if len(hashes)!=len(bank) or hashes&external_hashes:
            raise ValueError('Evaluation question hashes repeat within/across pools')
        external|=ids;external_hashes|=hashes;evaluations.append((e,frozen))
    if not evaluations:raise ValueError('Frozen label-blind G5 validation required')
    if ({g['qid'] for g in groups}&external or {g['question_hash'] for g in groups}&external_hashes):
        raise ValueError('Evaluation qid/question hash appears in accepted train G10')
    lookup=validate_plan(plan,groups,c,external)
    source=torch.load(paths['source_checkpoint'],map_location='cpu',mmap=True,weights_only=False)
    if (source['steps']!=10464 or source['epoch']!=1 or not source['completed_epoch'] or source['cursor']!=0 or
            source['actor_updates']!=0 or source['config_sha256']!=c['source_config']['sha256'] or
            source['bank_sha256']!=c['source_bank']['sha256']):raise ValueError('Sealed source E1 lineage/counters differ')
    if len(source['rank_states'])!=4 or {s['rank'] for s in source['rank_states']}!={0,1,2,3}:
        raise ValueError('Full four-rank source RNG archive required')
    source_ranks=copy.deepcopy(source['rank_states'])
    binding=dict(config_sha256=config_sha,matched_binding_sha256=u.digest(c),
        matched_binding=c,source_checkpoint=c['source_checkpoint'],g10_complete=c['g10_complete'],
        recipe=c['recipe'],plan_sha256=c['plan']['sha256'])
    from .model import Probe
    model=Probe(c['model_dir'],273,20).to(device);ema=copy.deepcopy(model).eval().requires_grad_(False)
    optimizer=torch.optim.Adam(model.parameters(),lr=5e-6,weight_decay=0)
    reader=native_reader(manifests)
    needed={tid for q in plan['cohort_qids'] for tid in lookup[q]['trajectory_ids']}
    if not needed<=set(reader.records):raise ValueError('Missing real original/extra native L26 trajectory features')
    u.atomic(base/'controller.json',dict(pid=os.getpid(),job_id=os.environ.get('SLURM_JOB_ID'),
        status='REAL_G10_TRAINING_INPUTS_ACCEPTED_NO_ACTOR_UPDATES',device=props.name,
        global_original_questions=c['global_questions'],source_steps=10464,
        current_world_size=1,source_world_size=4,four_rank_bitexact_resume_claim=False,**binding))
    for sig in (signal.SIGTERM,signal.SIGUSR1,signal.SIGINT):signal.signal(sig,request_stop)
    cursor=0
    try:
        if (base/'latest.json').exists():
            saved=torch.load(u.checked(u.read(base/'latest.json')),map_location='cpu',mmap=True,weights_only=False)
            cursor=saved['cursor']
            if saved['config_sha256']!=config_sha or saved['steps']!=10464+cursor or not 0<cursor<=c['update_budget']:
                raise ValueError('Exact G10 accepted checkpoint/config/cursor required')
            for filename,key in [('progress.json','cursor'),('update-intent.json','next_cursor')]:
                if (base/filename).exists() and u.read(base/filename)[key]>cursor:
                    raise ValueError('Unresolved later update; refuse silent rollback')
            u.restore_blob(saved,model,ema,optimizer,device)
            if u.state_hashes(model,ema,optimizer,device)!=saved['accepted_hashes']:
                raise ValueError('Restored fullstate hashes differ')
            source_ranks=saved['source_all_rank_states'];del saved
        else:
            if any((base/n).exists() for n in ('progress.json','update-intent.json')):
                raise ValueError('Unresolved progress/intent without accepted checkpoint; refuse reset')
            u.restore_blob(source,model,ema,optimizer,device)
            for k,current in [('model_sha256',u.model_hash(model)),('ema_sha256',u.model_hash(ema)),('optimizer_sha256',u.optimizer_hash(optimizer))]:
                if any(s[k]!=current for s in source_ranks):raise ValueError('Source fullstate '+k+' mismatch')
            u.atomic(base/'update-intent.json',dict(next_cursor=1,state='FIRST_UPDATE_REPLAY_IN_PROGRESS'))
            run_update(model,ema,optimizer,reader,lookup,plan['batches'][0],c,device,plan['batch_epochs'][0])
            expected=u.state_hashes(model,ema,optimizer,device)
            u.restore_blob(source,model,ema,optimizer,device)
            initial=run_update(model,ema,optimizer,reader,lookup,plan['batches'][0],c,device,plan['batch_epochs'][0])
            actual=u.state_hashes(model,ema,optimizer,device)
            u.atomic(base/'next-update-replay.json',dict(status='PASSED' if actual==expected else 'FAILED',
                expected=expected,actual=actual,source_steps=10464,candidate_steps=10465,
                source_world_size=4,current_world_size=1,four_rank_bitexact_resume_claim=False))
            if actual!=expected:raise ValueError('Actual first G10 next-update replay differs')
            cursor=1
            u.save_checkpoint(base,blob(model,ema,optimizer,device,cursor,binding,source_ranks))
            row=dict(cursor=cursor,steps=u.steps(optimizer),time=time.time(),**initial)
            u.atomic(base/'progress.json',row)
            with (base/'curve.jsonl').open('a') as f:f.write(__import__('json').dumps(row)+'\n');f.flush()
            print(__import__('json').dumps(row),flush=True)
        if any(g['lr']!=5e-6 or g['weight_decay']!=0 for g in optimizer.param_groups):
            raise ValueError('Inherited Adam hyperparameters changed')
        del source
        while cursor<c['update_budget'] and not STOP:
            u.atomic(base/'update-intent.json',dict(next_cursor=cursor+1,state='OPTIMIZER_TRANSACTION_IN_PROGRESS'))
            metrics=run_update(model,ema,optimizer,reader,lookup,plan['batches'][cursor],c,device,plan['batch_epochs'][cursor])
            cursor+=1;row=dict(cursor=cursor,steps=u.steps(optimizer),time=time.time(),**metrics)
            u.atomic(base/'progress.json',row)
            with (base/'curve.jsonl').open('a') as f:f.write(__import__('json').dumps(row)+'\n');f.flush()
            print(__import__('json').dumps(row),flush=True)
            if cursor%c['save_every']==0 or STOP or cursor==c['update_budget']:
                u.save_checkpoint(base,blob(model,ema,optimizer,device,cursor,binding,source_ranks))
        if STOP:
            if u.read(base/'latest.json')['cursor']!=cursor:
                u.save_checkpoint(base,blob(model,ema,optimizer,device,cursor,binding,source_ranks))
            u.atomic(base/'interrupted.json',dict(status='EXACT_ACCEPTED_CURSOR_SAVED',cursor=cursor));return
        # A checkpoint is accepted before every external evaluation call.
        if u.read(base/'latest.json')['cursor']!=cursor:raise ValueError('Final fullstate must be durable before evaluation')
        metrics={};(base/'predictions').mkdir(exist_ok=True)
        for e,frozen in evaluations:
            er=native_reader(u.read(u.checked(e['features'])))
            rng=u.rng_state(device);native=u.buffers(ema)
            try:metrics[e['name']]=u.evaluate(ema,er,frozen,device,base/'predictions'/(e['name']+'.jsonl.gz'))
            finally:u.restore_rng(rng,device);u.restore_buffers(ema,native);er.close()
        u.atomic(base/'complete.json',dict(status='REAL_G10_QUESTION_WEIGHTED_PROBE_BUDGET_AND_VALIDATION_COMPLETE',
            checkpoint=u.read(base/'latest.json'),cursor=cursor,steps=u.steps(optimizer),metrics=metrics,
            actor_updates=0,optimizer_reset=False,**binding))
    finally:reader.close()


def request_stop(_sig,_frame):
    global STOP
    STOP=True


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);a=p.parse_args()
    base=a.base.resolve()
    if not str(base).startswith('/data/user/jxu957/experiments/'):
        raise ValueError('All feature/checkpoint/prediction outputs must remain remote shared storage')
    base.mkdir(exist_ok=True,parents=True)
    with (base/'g10-train.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:run(base)
        except BaseException as error:
            u.atomic(base/'failure.json',dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc(),time=time.time()));raise


if __name__=='__main__':main()
