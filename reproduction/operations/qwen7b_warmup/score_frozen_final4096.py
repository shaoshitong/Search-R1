"""One frozen candidate versus E1 EMA on the two untouched real-G5 pools.

No preparation action reads final labels. The scoring entry requires an explicit
candidate-freeze receipt, exact checkpoint SHA and viewed-validation evidence.
Only then does the owned two-GPU allocation open sealed real journals/own EM.
Both probes see precisely the original precommitted own five trajectories.
Online causal prefill and the direct Mixed head are the existing pilot code.
"""
import argparse, copy, fcntl, gc, gzip, hashlib, inspect, json, math, os, time, traceback
from pathlib import Path

B = Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019')
A = B/'algorithm-improvement-20261021'
BASE = A/'frozen-final4096-scoring-20261022-v1'
MATERIALIZATION = B/'confirmation-materialization-deploy-20261021/materialized/materialization.json'
MATERIALIZATION_SHA = 'ff68f243b1daa7d8a617a2c7a825e1126d0771eee2e92b96a85dc5f9b4f2bfe3'
E1 = dict(path=str(A/'joint-mixed-direct-official-169028-b128-v1/checkpoints/cursor-002577.pt'),
          bytes=3781073882, sha256='c0a1c7dc3663d4216966a3440c2103348bcc52db3cd04c2f5c59a784a2792917')
REVISION = 'a09a35458c702b33eeacc393d103063234e8bc28'


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def remote(path):
    path = Path(path)
    if not path.is_absolute() or path.resolve() != path or not str(path).startswith(str(B)+'/'):
        raise ValueError('All models, journals, tensors and predictions remain remote')
    return path


def proof(path):
    path = remote(path); h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024**2), b''): h.update(chunk)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=h.hexdigest())


def checked(ref):
    if proof(ref['path']) != {key:ref[key] for key in ('path', 'bytes', 'sha256')}:
        raise ValueError('Frozen file SHA/size differs: '+ref['path'])
    return remote(ref['path'])


def atomic(path, value):
    path = remote(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.%d.tmp' % os.getpid())
    with temp.open('wb') as stream:
        stream.write(canonical(value)+b'\n'); stream.flush(); os.fsync(stream.fileno())
    temp.replace(path)


def freeze_contract(value):
    """Pure metadata gate; a checkpoint may be pilot or full E1 continuation."""
    if (value.get('status') != 'CANDIDATE_FROZEN_BEFORE_FINAL_SCORES'
            or value.get('final_labels_or_scores_used_for_selection') is not False
            or value.get('further_candidate_selection_on_final_allowed') is not False
            or value.get('primary_validation_questions') != 2048
            or value.get('baseline') != dict(checkpoint=E1, state_key='ema', layer=27, prefix=20)
            or not isinstance(value.get('freeze_epoch'), (float, int))):
        raise ValueError('Explicit viewed-validation-only candidate freeze required')
    candidate = value['candidate']
    if (candidate.get('layer'), candidate.get('prefix')) not in ((27,20),(21,20),(27,40),(27,80)):
        raise ValueError('Predeclared native layer/prefix geometry required')
    if candidate.get('state_key') not in ('raw', 'ema', 'EMA'):
        raise ValueError('Freeze actual raw/EMA checkpoint field, never implicit alias')
    for ref in (candidate['checkpoint'], value['viewed_validation_evaluation'], value['scorer_publication']):
        if set(('path','bytes','sha256'))-set(ref) or len(ref['sha256']) != 64:
            raise ValueError('Exact checkpoint, viewed-validation and source proofs required')
    return True


def plan_groups(planned):
    """Join no labels here; validate the original pre-generation labelblind G5."""
    fixed = planned['fixed_G5']; own = planned['planned_G10']
    if (planned['labels_observed'] is not False or fixed['summary']['role'] != 'final'
            or fixed['summary']['labels_consulted'] is not False
            or len(own) != 2048 or len(fixed['groups']) != 2048
            or digest(fixed['groups']) != fixed['summary']['selected_g5_sha256']):
        raise ValueError('Original untouched labelblind2048 ownG5 plan required')
    by = {v['qid']:v for v in own}; out = []
    if len(by) != 2048: raise ValueError('Repeated final question')
    for row in fixed['groups']:
        ids = row['trajectory_ids']; original = by[row['qid']]
        if (set(row) != {'qid','trajectory_ids'} or len(ids) != 5 or len(set(ids)) != 5
                or len(original['trajectory_ids']) != 10 or not set(ids)<=set(original['trajectory_ids'])):
            raise ValueError('Frozen own G5 identities changed')
        out.append(dict(qid=row['qid'], trajectory_ids=list(ids)))
    if {g['qid'] for g in out} != set(by): raise ValueError('Fixed G5 coverage changed')
    return out, by


def load_pool(pool, rank, world):
    """Called only AFTER candidate freeze. Fresh-verify all16 accepted batches."""
    from .peer_long_prefix_diagnostic import reconstruct
    selected, original = plan_groups(read(checked(pool['planned_G10_and_fixed_G5'])))
    mine = {g['qid']:g for i,g in enumerate(sorted(selected,key=lambda g:g['qid'])) if i%world==rank}
    wanted = {tid:q for q,g in mine.items() for tid in g['trajectory_ids']}
    records = {}; seen_all = set(); source_proofs = []; counts = dict(questions=0, trajectories=0)
    for event_index,ref in enumerate(pool['event_configs']):
        c = read(checked(ref))
        if (c['model_revision'] != REVISION or c['actor_optimizer_updates'] != 0
                or c['fitting_authorized'] is not False or c['event_index'] != event_index):
            raise ValueError('Frozen official event lineage changed')
        for shard in range(4):
            root = remote(c['base'])/'sampling'/('shard-%02d'%shard)
            finished = read(root/'complete.json')
            if (finished['status'] != 'ALL_ASSIGNED_CANONICAL_G5_COMPLETE'
                    or finished['questions'] != 512 or finished['trajectories'] != 2560
                    or finished['run_id'] != c['run_id'] or len(finished['batches']) != 4):
                raise ValueError('Every final event shard must genuinely finish before scoring')
            source_proofs.append(proof(root/'complete.json'))
            for batch_path in finished['batches']:
                batch_path = remote(batch_path)
                if batch_path.parent.parent != root/'batches': raise ValueError('Foreign batch receipt')
                batch = read(batch_path); full = batch['full']
                if batch['run_id'] != c['run_id'] or batch['actor_optimizer_updates'] != 0:
                    raise ValueError('Batch official actor provenance differs')
                em_path = checked(full['own_em']); own_em = {}
                with em_path.open() as stream:
                    for line in stream:
                        v=json.loads(line); tid=v['trajectory_id']
                        if tid in own_em or type(v['strict_em']) is not int or v['strict_em'] not in (0,1):
                            raise ValueError('Duplicate/nonbinary genuine own EM')
                        own_em[tid]=v['strict_em']
                found = set()
                with gzip.open(checked(full['full']), 'rt') as stream:
                    for line in stream:
                        v=json.loads(line); tid=v['trajectory_id']; q=v['qid']
                        if (tid in seen_all or q not in original or tid not in original[q]['event_trajectory_ids'][event_index]
                                or v['run_id'] != c['run_id'] or v['actor_version'] != 'official7b-'+REVISION
                                or v['strict_em'] != own_em.get(tid)):
                            raise ValueError('Wrong/duplicate actual final question, event, trajectory or own EM')
                        seen_all.add(tid); found.add(tid)
                        if tid in wanted:
                            if wanted[tid] != q: raise ValueError('Foreign own trajectory')
                            r=reconstruct(v,80)
                            records[tid]={k:r[k] for k in ('segments','positions','policy_tokens')}
                            records[tid].update(qid=q, trajectory_id=tid, prompt=v['initial_prompt_token_ids'], label=v['strict_em'])
                            if len(records[tid]['prompt'])>273: raise ValueError('Frozen prompt exceeds original geometry')
                if found != set(own_em) or len(found) != batch['trajectories']:
                    raise ValueError('Full actual journal/strict EM coverage differs')
                counts['questions']+=batch['questions']; counts['trajectories']+=batch['trajectories']
                source_proofs.extend([proof(batch_path),full['full'],full['own_em']])
    expected={tid for g in original.values() for tid in g['trajectory_ids']}
    if seen_all != expected or set(records) != set(wanted) or counts != dict(questions=4096,trajectories=20480):
        raise ValueError('Incomplete original real G10 or fixed own G5')
    for g in mine.values():
        g['labels']=[records[t]['label'] for t in g['trajectory_ids']]
        g['true_mixed']=0<sum(g['labels'])<5
    return list(mine.values()), records, dict(accepted_actual_G10=counts, proofs=source_proofs,
        original_fixed_G5_sha256=digest(selected), own_selected_G5_questions=len(mine))


def capture(actor, records, views, prefix, device):
    """Same actual causal context and padding as online_mixed_prefix_explore."""
    import torch
    from .online_mixed_prefix_explore import cut
    own=[cut(records[t],prefix) for v in views for t in v['trajectory_ids']]
    width=273+prefix; features=torch.zeros((len(own),width,3584),dtype=torch.bfloat16,device=device)
    mask=torch.zeros((len(own),width),dtype=torch.bool,device=device); positions=torch.zeros((len(own),width),dtype=torch.long,device=device); segments=[]
    for index,r in enumerate(own):
        qlen=len(r['prompt']); plen=r['policy_tokens']; mask[index,:qlen]=True; mask[index,273:273+plen]=True
        positions[index,:qlen]=torch.arange(qlen,device=device); positions[index,273:273+plen]=torch.tensor(r['positions'],device=device)
        for j,s in enumerate(r['segments']): segments.append((index,j==0,qlen,s))
    for offset in range(0,len(segments),32):
        part=segments[offset:offset+32]; length=max(len(s['context_tokens']) for _,_,_,s in part)
        ids=torch.full((len(part),length),151643,dtype=torch.long,device=device); am=torch.zeros_like(ids)
        for i,(_,_,_,s) in enumerate(part):
            tok=s['context_tokens']; ids[i,:len(tok)]=torch.tensor(tok,device=device); am[i,:len(tok)]=1
        hidden=actor(input_ids=ids,attention_mask=am,use_cache=False,return_dict=True).last_hidden_state
        for i,(index,first,qlen,s) in enumerate(part):
            if first: features[index,:qlen].copy_(hidden[i,:qlen])
            n=s['policy_length']; start=s['policy_start']; request=s['request_length']
            features[index,273+start:273+start+n].copy_(hidden[i,request:request+n])
        del hidden
    return dict(features=features,attention_mask=mask,position_ids=positions)


def paired_summary(rows, repeats=2000, seed=20261022):
    """Paired question bootstrap; same resampled questions for both methods.

    Re-select Top512 inside each resample, preserving its fixed budget. These
    intervals quantify this pool, not training-seed or dataset uncertainty.
    """
    import numpy as np
    from .online_mixed_prefix_explore import metrics
    if type(repeats) is not int or repeats<20: raise ValueError('At least20 predeclared paired bootstrap replicates required')
    rows=sorted(rows,key=lambda r:r['qid']); n=len(rows)
    if n != 2048 or len({r['qid'] for r in rows}) != n: raise ValueError('One intact final2048 pool required')
    y=np.asarray([r['true_mixed'] for r in rows],dtype=bool)
    scores=np.asarray([[r['baseline_logit'],r['candidate_logit']] for r in rows],dtype=float)
    if not np.isfinite(scores).all() or not y.any() or y.all(): raise ValueError('Finite scores and both Mixed classes required')
    result={name:metrics([dict(r,logit=r[name+'_logit']) for r in rows]) for name in ('baseline','candidate')}
    observed=[result['candidate']['balanced_accuracy']-result['baseline']['balanced_accuracy'],
              result['candidate']['selected512']['Mixed_fraction']-result['baseline']['selected512']['Mixed_fraction']]
    rng=np.random.default_rng(seed); samples=[]
    for _ in range(repeats):
        idx=rng.integers(0,n,size=n); yy=y[idx]; z=scores[idx]
        if not yy.any() or yy.all(): continue
        pred=z>=0; ba=.5*(pred[yy].mean(0)+(~pred[~yy]).mean(0))
        tops=[yy[np.argsort(-z[:,j],kind='stable')[:512]].mean() for j in (0,1)]
        samples.append([ba[1]-ba[0],tops[1]-tops[0]])
    ci=np.quantile(np.asarray(samples),[.025,.975],axis=0)
    top_sets={name:{r['qid'] for r in sorted(rows,key=lambda r:(-r[name+'_logit'],r['qid']))[:512]} for name in ('baseline','candidate')}
    return dict(metrics=result, paired_difference_candidate_minus_E1_EMA=dict(
        Mixed_balanced_accuracy=observed[0], Top512_true_Mixed_fraction=observed[1],
        percentile95_question_bootstrap=dict(Mixed_balanced_accuracy=ci[:,0].tolist(),Top512_true_Mixed_fraction=ci[:,1].tolist()),
        paired_discordant_correctness=dict(candidate_only=int(((scores[:,1]>=0)==y).sum())-int((((scores[:,1]>=0)==y)&((scores[:,0]>=0)==y)).sum()),
            baseline_only=int(((scores[:,0]>=0)==y).sum())-int((((scores[:,1]>=0)==y)&((scores[:,0]>=0)==y)).sum())),
        selected512_overlap=len(top_sets['baseline']&top_sets['candidate'])),
        bootstrap=dict(seed=seed,repeats=len(samples),unit='question',paired=True,reselect_Top512_in_each_resample=True,
            limitation='Within-pool question uncertainty only; no training-seed or cross-dataset guarantee'),
        threshold=0.5,threshold_fitted=False,labels_used_for_selection=False)


def score_variant(descriptor, views, records, device, progress, rank):
    import torch
    from torch import nn
    from transformers import AutoModel
    from .native_mixed_group_classifier import MixedTokenHead,NativeMixedGroupProbe
    layer,prefix=descriptor['layer'],descriptor['prefix']
    actor=AutoModel.from_pretrained(str(B/'assets/qwen25-7b-instruct-a09a35458c70'),local_files_only=True,
        torch_dtype=torch.bfloat16,attn_implementation='flash_attention_2').to(device).eval().requires_grad_(False)
    block=copy.deepcopy(actor.layers[layer]).float(); rotary=copy.deepcopy(actor.rotary_emb).float()
    actor.layers=nn.ModuleList(list(actor.layers[:layer])); actor.norm=nn.Identity()
    head=MixedTokenHead()
    if prefix!=20: head.policy_slots=nn.Embedding(prefix,256); head.prefix=prefix
    probe=NativeMixedGroupProbe(block,rotary,'cache_position' in inspect.signature(block.forward).parameters,head)
    blob=torch.load(checked(descriptor['checkpoint']),map_location='cpu',mmap=True,weights_only=False)
    if blob.get('actor_updates') != 0: raise ValueError('Frozen unchanged official-actor probe lineage required')
    key=descriptor['state_key']; probe.load_state_dict(blob[key],strict=True)
    if key+'_buffers' in blob:
        state=probe.state_dict(); targets={n:v for n,v in probe.named_buffers() if n not in state}
        if set(targets)!=set(blob[key+'_buffers']): raise ValueError('Exact native rotary buffers required')
        for name,value in targets.items(): value.copy_(blob[key+'_buffers'][name])
    if 'layer' in blob and (blob['layer']!=layer or blob['prefix']!=prefix): raise ValueError('Checkpoint actual geometry differs')
    if descriptor['checkpoint']==E1 and (key!='ema' or blob['cursor']!=2577 or blob['actor_updates']!=0):
        raise ValueError('Exact accepted original E1 EMA required')
    del blob; gc.collect(); probe.to(device).eval().requires_grad_(False); rows=[]; started=time.time()
    with torch.inference_mode():
        for offset in range(0,len(views),8):
            part=views[offset:offset+8]; data=capture(actor,records,part,prefix,device)
            with torch.autocast('cuda',dtype=torch.bfloat16): logits=probe(data).cpu().tolist()
            rows.extend(dict(v,logit=z) for v,z in zip(part,logits))
            atomic(progress,dict(stage='FROZEN_FINAL_REAL_CAUSAL_PREFILL',method=descriptor['name'],rank=rank,
                questions_scored=len(rows),expected_questions=len(views),layer=layer,prefix=prefix,
                seconds=time.time()-started,actor_updates=0,probe_updates=0,time=time.time()))
            del data
    del actor,probe; gc.collect(); torch.cuda.empty_cache(); return rows


def run(freeze_path, freeze_sha):
    # Gate tensor imports and final-label access behind a frozen candidate.
    if os.uname().sysname!='Linux' or not os.environ.get('SLURM_JOB_ID') or not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise RuntimeError('Owned remote two-GPU Slurm allocation required')
    freeze_path=remote(freeze_path)
    if proof(freeze_path)['sha256']!=freeze_sha: raise ValueError('Candidate-freeze SHA changed')
    frozen=read(freeze_path); freeze_contract(frozen)
    if frozen['freeze_epoch']>time.time(): raise ValueError('Candidate freeze is in the future')
    checked(frozen['viewed_validation_evaluation']); checked(frozen['scorer_publication'])
    pub=read(frozen['scorer_publication']['path']); code=Path(__file__).resolve().parents[3]
    if pub['code_root']!=str(code): raise ValueError('Wrong independently staged scorer code')
    for name,ref in pub['files'].items(): checked(dict(ref,path=str(code/name)))
    official_root=B/'assets/qwen25-7b-instruct-a09a35458c70'; official=read(official_root/'complete.json')
    if official['revision']!=REVISION or official['status']!='MIGRATION_ALL_SELECTED_ASSETS_VERIFIED':
        raise ValueError('Exact verified frozen official actor snapshot required')
    for name,ref in official['files'].items():
        if Path(name).name!=name: raise ValueError('Invalid official actor filename')
        checked(dict(ref,path=str(official_root/name)))
    import torch
    import torch.distributed as dist
    rank=int(os.environ['RANK']); world=int(os.environ['WORLD_SIZE'])
    if world!=2: raise ValueError('Two GPU ranks required')
    device=torch.device('cuda',int(os.environ['LOCAL_RANK'])); torch.cuda.set_device(device); dist.init_process_group('nccl')
    BASE.mkdir(exist_ok=True); guard=(BASE/('scoring-rank-%d.lock'%rank)).open('a+'); fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (BASE/'final-use-started.json').exists():
        if read(BASE/'final-use-started.json')['freeze']!=proof(freeze_path): raise ValueError('Final pool already unblinded to a different candidate')
    elif rank==0: atomic(BASE/'final-use-started.json',dict(freeze=proof(freeze_path),time=time.time(),further_tuning_allowed=False))
    dist.barrier()
    if proof(MATERIALIZATION)['sha256']!=MATERIALIZATION_SHA: raise ValueError('Original untouched final identities changed')
    materialized=read(MATERIALIZATION); all_qids=set(); results={}; source_proofs={}
    for pool_name in ('confirmation2048','stability2048'):
        views,records,sources=load_pool(materialized['pools'][pool_name],rank,world)
        own_qids={r['qid'] for r in views}
        if all_qids&own_qids: raise ValueError('Two final question pools overlap')
        all_qids|=own_qids
        descriptors=[dict(frozen['baseline'],name='baseline'),dict(frozen['candidate'],name='candidate')]
        scores={}
        for descriptor in descriptors:
            rows=score_variant(descriptor,views,records,device,BASE/('progress-rank-%d.json'%rank),rank)
            scores[descriptor['name']]={r['qid']:r['logit'] for r in rows}
        paired=[dict(v,baseline_logit=scores['baseline'][v['qid']],candidate_logit=scores['candidate'][v['qid']]) for v in views]
        gathered=[None]*world; dist.all_gather_object(gathered,paired)
        if rank==0:
            rows=sorted([r for part in gathered for r in part],key=lambda r:r['qid'])
            path=BASE/(pool_name+'-paired-predictions.jsonl.gz'); temp=path.with_suffix('.tmp')
            with gzip.open(temp,'wt') as stream:
                for row in rows: stream.write(json.dumps(row)+'\n')
            temp.replace(path); results[pool_name]=paired_summary(rows)
            results[pool_name]['predictions']=proof(path); source_proofs[pool_name]=sources
            atomic(BASE/(pool_name+'-complete.json'),dict(metrics=results[pool_name],freeze=proof(freeze_path),sources=sources))
        del records; dist.barrier()
    if rank==0:
        complete=dict(status='FROZEN_FINAL4096_ONE_CANDIDATE_PAIRED_E1_EMA_COMPLETE',freeze=proof(freeze_path),
            pools=results,sources=source_proofs,questions=4096,fixed_G5_trajectories=20480,
            actor_updates=0,probe_updates=0,hidden_features_persisted=False,further_tuning_on_final_allowed=False,
            threshold_fitted=False,final_metrics_used_for_candidate_selection=False,time=time.time())
        atomic(BASE/'complete.json',complete)
    dist.destroy_process_group()


def main():
    p=argparse.ArgumentParser(); p.add_argument('--freeze-json',required=True); p.add_argument('--freeze-sha',required=True)
    args=p.parse_args()
    try: run(args.freeze_json,args.freeze_sha)
    except BaseException as error:
        if os.uname().sysname=='Linux' and BASE.exists():
            atomic(BASE/('failure-rank-%s.json'%os.environ.get('RANK','unknown')),dict(error=repr(error),traceback=traceback.format_exc(),time=time.time()))
        raise


if __name__=='__main__': main()
