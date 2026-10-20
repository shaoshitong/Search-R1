"""Matched, one-H100, actor-frozen Prefix20 loss ablation; never submits jobs.

Run ``python -m ...probe_ablation_train --base /remote/variant`` after preparing
config.json and its explicit immutable qid batch plan. This is a NEW one-GPU
branch from sealed E1, not a bit-exact continuation of the four-rank algorithm.
All source rank RNG states are archived; canonical source rank0 seeds the branch.
Complete B51 logits retain all microforward graphs until ONE global backward.
"""
import argparse
import copy
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import time
import traceback
from contextlib import nullcontext
import numpy as np
import torch
from .probe_ablation_loss import probe_ablation_loss, group_event_scores

SOURCE_SHA = '02884a3c76e88d1014c4876e1ee2ff4b4405e7f94de7b9112881fed183862724'
STOP = False


def read(path):
    return json.loads(Path(path).read_text())


def atomic(path, value):
    path = Path(path);temp = path.with_name(path.name+'.tmp')
    with temp.open('w') as f:
        json.dump(value, f, indent=2, allow_nan=False);f.flush();os.fsync(f.fileno())
    temp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024**2), b''):h.update(block)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',',':'),
                                     allow_nan=False).encode()).hexdigest()


def checked(reference):
    p = Path(reference['path'])
    if not p.is_absolute() or sha(p) != reference['sha256']:
        raise ValueError('Absolute immutable asset path/SHA mismatch: '+str(p))
    if 'bytes' in reference and p.stat().st_size != reference['bytes']:
        raise ValueError('Asset byte count mismatch: '+str(p))
    return p


def tensor_hash(mapping):
    h = hashlib.sha256()
    for name, value in mapping.items():
        h.update(name.encode());h.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def model_hash(model):
    # Match the legacy source receipt's raw FP32 hash convention.
    return model_mapping_hash(model.state_dict())


def model_mapping_hash(mapping):
    h = hashlib.sha256()
    for name, value in mapping.items():
        h.update(name.encode());h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def optimizer_hash(optimizer):
    return optimizer_blob_hash(optimizer.state_dict())


def optimizer_blob_hash(saved):
    h = hashlib.sha256()
    for group in saved['param_groups']:
        for parameter in group['params']:
            for key, value in sorted(saved['state'].get(parameter,{}).items()):
                h.update(key.encode())
                h.update(value.detach().cpu().contiguous().numpy().tobytes()
                         if torch.is_tensor(value) else repr(value).encode())
    return h.hexdigest()


def steps(optimizer):
    counts = {int(v['step'].item()) for v in optimizer.state.values() if v}
    if len(counts) != 1:raise ValueError('Adam update counts are absent or inconsistent')
    return counts.pop()


def buffers(model):
    state = model.state_dict()
    return {k:v.detach().cpu().clone() for k,v in model.named_buffers() if k not in state}


def restore_buffers(model, saved):
    targets = {k:v for k,v in model.named_buffers() if k not in model.state_dict()}
    if set(targets) != set(saved):raise ValueError('Native buffer names mismatch')
    with torch.no_grad():
        for k, v in targets.items():
            if v.shape != saved[k].shape:raise ValueError('Native buffer geometry mismatch')
            v.copy_(saved[k].to(v.device))


def rng_state(device):
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state(device) if device.type=='cuda' else None)


def restore_rng(saved, device):
    random.setstate(saved['python']);np.random.set_state(saved['numpy'])
    torch.set_rng_state(saved['torch'].clone())
    if device.type=='cuda':torch.cuda.set_rng_state(saved['cuda'].clone(),device)


def rng_hash(value):
    h = hashlib.sha256(repr(value['python']).encode())
    n = value['numpy'];h.update(n[0].encode());h.update(n[1].tobytes());h.update(repr(n[2:]).encode())
    for name in ('torch','cuda'):
        v = value[name];h.update(name.encode());h.update(v.numpy().tobytes() if v is not None else b'None')
    return h.hexdigest()


def state_hashes(model, ema, optimizer, device):
    return dict(raw=model_hash(model),ema=model_hash(ema),adam=optimizer_hash(optimizer),
                raw_buffers=tensor_hash(buffers(model)),ema_buffers=tensor_hash(buffers(ema)),
                rng=rng_hash(rng_state(device)))


def restore_blob(blob, model, ema, optimizer, device):
    model.load_state_dict(blob['raw'],strict=True);ema.load_state_dict(blob['ema'],strict=True)
    restore_buffers(model,blob['raw_buffers']);restore_buffers(ema,blob['ema_buffers'])
    # Adam can otherwise retain source CPU step tensors by reference.
    optimizer.load_state_dict(copy.deepcopy(blob['optimizer']))
    for state in optimizer.state.values():
        for name, value in state.items():
            if torch.is_tensor(value) and name!='step':state[name]=value.to(device)
    if 'rng' in blob:
        rng = blob['rng']
    else:
        states = [s for s in blob['rank_states'] if s['rank']==0]
        if len(states)!=1:raise ValueError('Exactly one canonical source rank0 required')
        rng = states[0]['rng']
    restore_rng(rng,device)
    if steps(optimizer)!=blob['steps']:raise ValueError('Restored Adam count mismatch')


def validate_plan(plan, train, update_budget, global_groups=51, forbidden=()):
    if global_groups!=51 or not isinstance(update_budget,int) or update_budget<1:
        raise ValueError('Matched ablation requires positive update budget and global B51')
    lookup = {g['qid']:g for g in train}
    if len(lookup)!=len(train):raise ValueError('Duplicate training qid')
    cohort = plan['cohort_qids'];batches = plan['batches']
    if len(cohort)!=len(set(cohort)) or len(batches)!=update_budget:
        raise ValueError('Duplicate cohort or mismatched update budget')
    if set(cohort)&set(forbidden):raise ValueError('External evaluation leaked into fit cohort')
    used = set()
    for batch in batches:
        if len(batch)!=51 or len(set(batch))!=51 or not set(batch)<=set(cohort):
            raise ValueError('Each explicit global batch needs 51 distinct cohort questions')
        used.update(batch)
    if used!=set(cohort) or not used<=set(lookup):raise ValueError('Explicit cohort/plan/train disagreement')
    for qid in cohort:
        g = lookup[qid]
        if len(g['trajectory_ids'])!=5 or len(set(g['trajectory_ids']))!=5 or len(g['labels'])!=5 or any(y not in (0,1) for y in g['labels']):
            raise ValueError('Complete own-EM G5 required')
    return lookup


def run_update(model, ema, optimizer, reader, groups, qids, config, device):
    before = steps(optimizer);optimizer.zero_grad(set_to_none=True);model.train()
    logits = [];micro = config['micro_groups'];selected = [groups[q] for q in qids]
    if not isinstance(micro,int) or not 1<=micro<=51:raise ValueError('Micro groups must be 1..51')
    start = time.monotonic()
    for offset in range(0,len(selected),micro):
        part = selected[offset:offset+micro]
        data = reader.get([tid for g in part for tid in g['trajectory_ids']],device)
        context = torch.autocast('cuda',dtype=torch.bfloat16) if device.type=='cuda' else nullcontext()
        with context:logits.append(model(data).reshape(len(part),5))
    # Retain every graph; group ranking sees the actual SAME complete global B.
    z = torch.cat(logits,0)
    y = torch.tensor([g['labels'] for g in selected],device=device,dtype=z.dtype)
    kwargs = dict(config['recipe']['loss_kwargs'])
    if float(kwargs.get('within_rank_weight',0.)):
        kwargs['within_pair_mask'] = torch.tensor([g['pair_mask'] for g in selected],device=device,dtype=torch.bool)
    loss = probe_ablation_loss(z,y,**kwargs)
    if not bool(torch.isfinite(loss['loss'])):raise ValueError('Nonfinite global loss')
    loss['loss'].backward()
    if not all(p.grad is None or bool(torch.isfinite(p.grad).all()) for p in model.parameters()):
        raise ValueError('Nonfinite global probe gradient')
    optimizer.step()
    if steps(optimizer)!=before+1:raise ValueError('Adam did not advance exactly once')
    with torch.no_grad():
        for target, value in zip(ema.parameters(),model.parameters()):target.mul_(.99).add_(value,alpha=.01)
    names = ('loss','trajectory_bce','group_bce','group_event_nll','within_rank_loss',
             'inter_rank_loss','stat_mean_mse','stat_variance_mse')
    predicted=loss['mixed_probability'].detach()>=.5
    target=loss['mixed_label'].detach().bool()
    tp=int((predicted&target).sum());fn=int((~predicted&target).sum())
    fp=int((predicted&~target).sum());tn=int((~predicted&~target).sum())
    tpr=tp/(tp+fn) if tp+fn else None
    tnr=tn/(tn+fp) if tn+fp else None
    return dict({k:float(loss[k].detach()) for k in names},
                seconds=time.monotonic()-start,global_questions=len(qids),
                within_eligible_groups=loss['within_eligible_groups'],
                inter_actual_pairs=loss['inter_actual_pairs'],actor_updates=0,
                mixed_confusion=dict(tp=tp,fp=fp,fn=fn,tn=tn),
                mixed_recall=tpr,nonmixed_recall=tnr,
                mixed_balanced_accuracy=(tpr+tnr)/2 if tpr is not None and tnr is not None else None)


def roc_auc(scores, labels):
    positives = sum(labels);negatives = len(labels)-positives
    if not positives or not negatives:return None
    ordered = sorted(zip(scores,labels));rank_sum=0.;i=0
    while i<len(ordered):
        j=i+1
        while j<len(ordered) and ordered[j][0]==ordered[i][0]:j+=1
        rank_sum += ((i+1+j)/2)*sum(y for _,y in ordered[i:j]);i=j
    return (rank_sum-positives*(positives+1)/2)/(positives*negatives)


def prediction_metrics(rows):
    if not rows:raise ValueError('No evaluation questions')
    counts = [sum(r['strict_em']) for r in rows]
    labels = [0<k<5 for k in counts];scores=[r['predicted_mixed'] for r in rows]
    predicted = [q>=.5 for q in scores]
    tp=sum(p and y for p,y in zip(predicted,labels));fn=sum(not p and y for p,y in zip(predicted,labels))
    fp=sum(p and not y for p,y in zip(predicted,labels));tn=sum(not p and not y for p,y in zip(predicted,labels))
    recall = tp/(tp+fn) if tp+fn else None
    specificity = tn/(tn+fp) if tn+fp else None
    per_class={}
    for name, valid in [('allwrong',[k==0 for k in counts]),('mixed',labels),('allcorrect',[k==5 for k in counts])]:
        indices=[i for i,b in enumerate(valid) if b]
        called_mixed=sum(predicted[i] for i in indices)
        per_class[name]=dict(questions=len(indices),predicted_mixed=sum(predicted[i] for i in indices),
                             predicted_mixed_fraction=called_mixed/len(indices) if indices else None,
                             mean_q=sum(scores[i] for i in indices)/len(indices) if indices else None,
                             false_positive_rate=called_mixed/len(indices) if indices and name!='mixed' else None,
                             false_negative_rate=1-called_mixed/len(indices) if indices and name=='mixed' else None,
                             correct_group_recall=(called_mixed if name=='mixed' else len(indices)-called_mixed)/len(indices) if indices else None)
    return dict(questions=len(rows),mixed_questions=sum(labels),
                mixed_fraction=sum(labels)/len(rows),
                confusion=dict(tp=tp,fp=fp,fn=fn,tn=tn),mixed_recall=recall,
                nonmixed_recall=specificity,
                mixed_false_negative_rate=1-recall if recall is not None else None,
                nonmixed_false_positive_rate=1-specificity if specificity is not None else None,
                mixed_balanced_accuracy=(recall+specificity)/2 if recall is not None and specificity is not None else None,
                mixed_roc_auc=roc_auc(scores,labels),classwise=per_class,
                k_histogram=[counts.count(k) for k in range(6)],mean_q=sum(scores)/len(rows))


@torch.no_grad()
def evaluate(model, reader, groups, device, destination, micro=8):
    model.eval();rows=[]
    for offset in range(0,len(groups),micro):
        part=groups[offset:offset+micro]
        data=reader.get([tid for g in part for tid in g['trajectory_ids']],device)
        context=torch.autocast('cuda',dtype=torch.bfloat16) if device.type=='cuda' else nullcontext()
        with context:z=model(data).reshape(len(part),5)
        # No labels enter this score computation or its question ordering.
        probabilities=z.sigmoid().cpu().tolist()
        q=group_event_scores(z)['mixed_probability'].cpu().tolist()
        for g,p,value in zip(part,probabilities,q):
            rows.append(dict(qid=g['qid'],trajectory_ids=g['trajectory_ids'],p_j=p,
                             predicted_mixed=value,strict_em=g['labels']))
    selected=sorted(rows,key=lambda r:(-r['predicted_mixed'],str(r['qid'])))[:512]
    temporary=destination.with_suffix('.tmp')
    with gzip.open(temporary,'wt') as f:
        for row in rows:f.write(json.dumps(row)+'\n')
    temporary.replace(destination)
    return dict(pool=prediction_metrics(rows),selected512=prediction_metrics(selected),
                predictions=dict(path=str(destination),sha256=sha(destination)),
                evaluation_only=True,calibration_fitted=False)


def save_checkpoint(base, blob):
    root=base/'checkpoints';root.mkdir(exist_ok=True)
    destination=root/('cursor-%06d.pt'%blob['cursor']);temporary=destination.with_suffix('.tmp')
    if destination.exists():raise ValueError('Refusing to overwrite immutable accepted checkpoint')
    with temporary.open('wb') as f:torch.save(blob,f);f.flush();os.fsync(f.fileno())
    temporary.replace(destination)
    check=torch.load(destination,map_location='cpu',mmap=True,weights_only=False)
    if check['cursor']!=blob['cursor'] or check['steps']!=blob['steps']:
        raise ValueError('Saved cursor/Adam mismatch')
    for field in ('raw','ema'):
        if not all(bool(torch.isfinite(v).all()) for v in check[field].values()):
            raise ValueError('Nonfinite saved parameter')
    for state in check['optimizer']['state'].values():
        if int(state['step'])!=blob['steps'] or not all(bool(torch.isfinite(state[k]).all()) for k in ('exp_avg','exp_avg_sq')):
            raise ValueError('Invalid saved Adam state')
    expected=check['accepted_hashes']
    actual=dict(raw=model_mapping_hash(check['raw']),ema=model_mapping_hash(check['ema']),
                adam=optimizer_blob_hash(check['optimizer']),
                raw_buffers=tensor_hash(check['raw_buffers']),
                ema_buffers=tensor_hash(check['ema_buffers']),rng=rng_hash(check['rng']))
    if expected!=actual:raise ValueError('Saved raw/EMA/Adam/native-buffer/RNG component hashes mismatch')
    if len(check['source_all_rank_states'])!=4:
        raise ValueError('Missing original four-rank RNG archive')
    del check
    receipt=dict(path=str(destination),sha256=sha(destination),bytes=destination.stat().st_size,
                 cursor=blob['cursor'],steps=blob['steps'],config_sha256=blob['config_sha256'],
                 matched_binding_sha256=blob['matched_binding_sha256'],epoch_time=time.time())
    receipt['accepted_hashes']=actual
    atomic(base/'latest.json',receipt)
    return receipt


def checkpoint_blob(model,ema,optimizer,device,cursor,binding,source_rank_states):
    return dict(role='NEW_SINGLE_GPU_MATCHED_PROBE_LOSS_ABLATION',cursor=cursor,
                steps=steps(optimizer),raw=model.state_dict(),ema=ema.state_dict(),
                optimizer=optimizer.state_dict(),raw_buffers=buffers(model),
                ema_buffers=buffers(ema),rng=rng_state(device),
                accepted_hashes=state_hashes(model,ema,optimizer,device),
                source_all_rank_states=source_rank_states,
                hardware_topology_changed=True,source_world_size=4,current_world_size=1,
                canonical_source_rng_rank=0,four_rank_bitexact_resume_claim=False,
                actor_updates=0,**binding)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--base',type=Path,required=True)
    args=parser.parse_args();base=args.base.resolve()
    if not str(base).startswith('/data/user/jxu957/experiments/'):
        raise ValueError('Outputs/checkpoints must use the remote shared experiment storage')
    base.mkdir(exist_ok=True,parents=True)
    lock=(base/'ablation.lock').open('a+')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    try:run(base)
    except BaseException as error:
        atomic(base/'failure.json',dict(type=type(error).__name__,message=str(error),
                                        traceback=traceback.format_exc(),epoch_time=time.time()))
        raise
    finally:lock.close()


def run(base):
    global STOP
    c=read(base/'config.json');config_sha=sha(base/'config.json')
    if (not isinstance(c['save_every'],int) or c['save_every']<1 or
            not isinstance(c['micro_groups'],int) or not 1<=c['micro_groups']<=51):
        raise ValueError('Explicit positive save cadence and micro groups 1..51 required')
    if set(c['recipe']['loss_kwargs']) & {'logits','labels','within_pair_mask'}:
        raise ValueError('Recipe may not inject labels, logits, or an external mask')
    required=('source_checkpoint','source_config','source_bank','model_config','bank','features',
              'selection_bank','selection_features','plan')
    paths={name:checked(c[name]) for name in required}
    if (Path(c['model_dir'])/'config.json').resolve()!=paths['model_config'].resolve():
        raise ValueError('model_dir must use the exact hash-bound model configuration')
    if c['source_checkpoint']['sha256']!=SOURCE_SHA:
        raise ValueError('Ablations must share the pinned sealed E1 checkpoint')
    repo=Path(__file__).resolve().parents[3]
    code_required={'reproduction/operations/qwen7b_warmup/'+name for name in
                   ('probe_ablation_train.py','probe_ablation_loss.py','model.py',
                    'packed_features.py','native_capture.py','download_model.py')}
    if not code_required<=set(c['code_hashes']):raise ValueError('Missing actual runtime code bindings')
    for name,expected in c['code_hashes'].items():
        if Path(name).is_absolute() or '..' in Path(name).parts or sha(repo/name)!=expected:
            raise ValueError('Runtime source SHA mismatch: '+name)
    if int(os.environ.get('WORLD_SIZE','1'))!=1 or torch.cuda.device_count()!=1:
        raise ValueError('This matched branch intentionally requires exactly ONE visible GPU')
    device=torch.device('cuda',0);properties=torch.cuda.get_device_properties(device)
    if 'H100' not in properties.name or properties.total_memory<75*1024**3:
        raise ValueError('Complete global graph requires a dedicated >=75GiB H100')
    if os.environ.get('FLASH_ATTENTION_DETERMINISTIC')!='1' or os.environ.get('CUBLAS_WORKSPACE_CONFIG')!=':4096:8':
        raise ValueError('Set deterministic FA and CUBLAS flags before process startup')
    torch.cuda.set_device(device);torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False;torch.backends.cuda.matmul.allow_tf32=False
    os.environ['QWEN7B_QUESTION_MAX_TOKENS']='273';os.environ['SEARCH_G1_B_PREFIX_TOKENS']='20'
    from .model import Probe
    from .packed_features import FeatureReader
    bank=read(paths['bank']);held=read(paths['selection_bank'])['selection_validation']
    diagnostic=bank['diagnostic'];train=bank['train'];plan=read(paths['plan'])
    if len(diagnostic)!=512 or len(held)!=2048:raise ValueError('Fixed512 and2048 evaluation banks required')
    external={g['qid'] for g in diagnostic+held}
    if len(external)!=2560:raise ValueError('Diagnostic and selection question overlap')
    lookup=validate_plan(plan,train,c['update_budget'],c['global_groups'],external)
    if {g['qid'] for g in train}&external:raise ValueError('External questions appear in source train pool')
    source=torch.load(paths['source_checkpoint'],map_location='cpu',mmap=True,weights_only=False)
    if (source['steps']!=10464 or source['epoch']!=1 or not source['completed_epoch'] or
            source['cursor']!=0 or source['actor_updates']!=0 or
            source['config_sha256']!=c['source_config']['sha256'] or
            source['bank_sha256']!=c['source_bank']['sha256']):
        raise ValueError('Sealed E1 lineage/counters do not match explicit source bindings')
    if len(source['rank_states'])!=4 or {s['rank'] for s in source['rank_states']}!={0,1,2,3}:
        raise ValueError('Full original four-rank RNG archive required')
    source_ranks=copy.deepcopy(source['rank_states'])
    matched=dict(source_assets={k:c[k]['sha256'] for k in required},
                 update_budget=c['update_budget'],global_groups=c['global_groups'],
                 micro_groups=c['micro_groups'],prefix_tokens=20,question_max_tokens=273,
                 lr=5e-6,ema=.99,code_hashes=c['code_hashes'],
                 model_dir=str(Path(c['model_dir']).resolve()),model_config_sha256=c['model_config']['sha256'])
    binding=dict(config_sha256=config_sha,matched_binding_sha256=digest(matched),
                 matched_binding=matched,source_checkpoint=c['source_checkpoint'],
                 recipe=c['recipe'],plan_sha256=c['plan']['sha256'])
    atomic(base/'controller.json',dict(pid=os.getpid(),job_id=os.environ.get('SLURM_JOB_ID'),
           device=properties.name,total_gpu_bytes=properties.total_memory,
           status='NEW_SINGLE_GPU_ABLATION_LOCKED',hardware_topology_changed=True,
           canonical_source_rng_rank=0,four_rank_bitexact_resume_claim=False,
           source_all_ranks_archived=True,actor_updates=0,epoch_time=time.time(),**binding))
    model=Probe(c['model_dir'],273,20).to(device);ema=copy.deepcopy(model).eval().requires_grad_(False)
    optimizer=torch.optim.Adam(model.parameters(),lr=5e-6,weight_decay=0)
    reader=FeatureReader(read(paths['features']));held_reader=None
    for sig in (signal.SIGTERM,signal.SIGUSR1,signal.SIGINT):signal.signal(sig,request_stop)
    resumed=(base/'latest.json').exists()
    cursor=0
    if resumed:
        receipt=read(base/'latest.json');checkpoint=checked(receipt)
        saved=torch.load(checkpoint,map_location='cpu',mmap=True,weights_only=False)
        if saved['config_sha256']!=config_sha or saved['matched_binding_sha256']!=binding['matched_binding_sha256']:
            raise ValueError('Resume configuration or matched comparison bindings changed')
        cursor=saved['cursor']
        if saved['steps']!=10464+cursor or not 0<=cursor<=c['update_budget']:
            raise ValueError('Accepted cursor/update budget mismatch')
        if (base/'progress.json').exists() and read(base/'progress.json')['cursor']>cursor:
            raise ValueError('Unsaved accepted progress exists: refusing silent rollback')
        if (base/'update-intent.json').exists() and read(base/'update-intent.json')['next_cursor']>cursor:
            raise ValueError('Unresolved later optimizer transaction: refusing silent rollback')
        restore_blob(saved,model,ema,optimizer,device)
        if state_hashes(model,ema,optimizer,device)!=saved['accepted_hashes']:
            raise ValueError('Restored raw/EMA/Adam/buffer/RNG hashes differ from accepted checkpoint')
        if any(g['lr']!=5e-6 or g['weight_decay']!=0 for g in optimizer.param_groups):
            raise ValueError('Inherited Adam LR/weight decay changed')
        source_ranks=saved['source_all_rank_states'];del saved
    else:
        if (base/'progress.json').exists():raise ValueError('Progress without checkpoint: refuse reset')
        restore_blob(source,model,ema,optimizer,device)
        if any(g['lr']!=5e-6 or g['weight_decay']!=0 for g in optimizer.param_groups):
            raise ValueError('Inherited Adam LR/weight decay changed')
        for key,current in [('model_sha256',model_hash(model)),('ema_sha256',model_hash(ema)),('optimizer_sha256',optimizer_hash(optimizer))]:
            if any(s[key]!=current for s in source_ranks):raise ValueError('Source '+key+' mismatch')
        # Two actual complete-global-batch optimizer updates from the SAME source.
        run_update(model,ema,optimizer,reader,lookup,plan['batches'][0],c,device)
        expected=state_hashes(model,ema,optimizer,device)
        restore_blob(source,model,ema,optimizer,device)
        initial=run_update(model,ema,optimizer,reader,lookup,plan['batches'][0],c,device)
        actual=state_hashes(model,ema,optimizer,device)
        proof=dict(status='SINGLE_GPU_NEXT_UPDATE_REPLAY_PASSED' if actual==expected else 'FAILED',
                   expected=expected,actual=actual,source_steps=10464,candidate_steps=10465,
                   no_four_rank_equivalence_claim=True,epoch_time=time.time())
        atomic(base/'next-update-replay.json',proof)
        if actual!=expected:raise ValueError('Actual first next-update replay mismatch')
        cursor=1
        save_checkpoint(base,checkpoint_blob(model,ema,optimizer,device,cursor,binding,source_ranks))
        initial_row=dict(cursor=cursor,steps=steps(optimizer),budget=c['update_budget'],
                         max_gpu_allocated=torch.cuda.max_memory_allocated(device),
                         **initial,epoch_time=time.time())
        atomic(base/'progress.json',initial_row)
        with (base/'curve.jsonl').open('a') as f:f.write(json.dumps(initial_row)+'\n');f.flush()
        print(json.dumps(initial_row),flush=True)
    del source
    try:
        while cursor<c['update_budget'] and not STOP:
            atomic(base/'update-intent.json',dict(next_cursor=cursor+1,state='OPTIMIZER_TRANSACTION_IN_PROGRESS',
                                                  epoch_time=time.time()))
            metrics=run_update(model,ema,optimizer,reader,lookup,plan['batches'][cursor],c,device)
            cursor+=1
            row=dict(cursor=cursor,steps=steps(optimizer),budget=c['update_budget'],
                     max_gpu_allocated=torch.cuda.max_memory_allocated(device),
                     epoch_time=time.time(),**metrics)
            atomic(base/'progress.json',row)
            with (base/'curve.jsonl').open('a') as f:f.write(json.dumps(row)+'\n');f.flush()
            print(json.dumps(row),flush=True)
            if cursor%c['save_every']==0 or STOP or cursor==c['update_budget']:
                save_checkpoint(base,checkpoint_blob(model,ema,optimizer,device,cursor,binding,source_ranks))
        if STOP:
            latest=read(base/'latest.json')
            if latest['cursor']!=cursor:
                save_checkpoint(base,checkpoint_blob(model,ema,optimizer,device,cursor,binding,source_ranks))
            atomic(base/'interrupted.json',dict(status='SAVED_EXACT_ACCEPTED_CURSOR',cursor=cursor,
                                               steps=steps(optimizer),epoch_time=time.time()))
            return
        # Final optimizer state is durable BEFORE any external evaluation.
        rng_before=rng_state(device);ema_buffers=buffers(ema)
        held_reader=FeatureReader(read(paths['selection_features']))
        (base/'predictions').mkdir(exist_ok=True)
        try:
            metrics=dict(diagnostic512=evaluate(ema,reader,diagnostic,device,base/'predictions/diagnostic512.jsonl.gz'),
                         selection2048=evaluate(ema,held_reader,held,device,base/'predictions/selection2048.jsonl.gz'))
        finally:restore_rng(rng_before,device);restore_buffers(ema,ema_buffers)
        atomic(base/'metrics.json',dict(metrics=metrics,fit_updates=cursor,
                                       config_sha256=config_sha,epoch_time=time.time()))
        atomic(base/'complete.json',dict(status='MATCHED_PROBE_LOSS_ABLATION_COMPLETE',
               cursor=cursor,steps=steps(optimizer),checkpoint=read(base/'latest.json'),
               metrics=metrics,actor_updates=0,epoch_time=time.time(),**binding))
    finally:
        reader.close()
        if held_reader is not None:held_reader.close()


def request_stop(_sig,_frame):
    global STOP
    STOP=True


if __name__=='__main__':main()
