"""Prepared-only large-cohort frozen-E1 count-head capacity experiment.

The module has no submission action. It accepts an already frozen, SHA-bound
train-only input packet. All model/cache/checkpoint operations are GPU-allocation
only; metadata preparation neither loads tensors nor opens external/final data.
"""
import argparse
import copy
import fcntl
import gzip
import json
import os
from pathlib import Path
import shutil
import signal
import time
import traceback

from . import direct_count_capacity as old

B = old.B
PRODUCTION = old.PRODUCTION
TARGET = B/'algorithm-improvement-20261021/direct-count-frozenE1-fit30016-hold5120-v2'
NODES = (0, 1, 2, 4, 8, 20)
FIT = 30016
HELD = 5120
Q_BATCH = 32
EPOCHS = 20
STOP = False
canonical, digest, proof = old.canonical, old.digest, old.proof
read, checked, atomic, immutable = old.read, old.checked, old.atomic, old.immutable


def checked_model_config(ref):
    """Legacy descriptor is path+SHA; byte count is optional, never inferred."""
    p=Path(ref['path'])
    if p.resolve()!=p or not p.is_relative_to(B):raise ValueError('Owned model config path required')
    actual=proof(p)
    if actual['sha256']!=ref['sha256']or('bytes'in ref and actual['bytes']!=ref['bytes']):
        raise ValueError('Model config actual SHA/optional byte count differs')
    return actual


def startup_metadata_guards(c):
    """CPU-only exact descriptor schema checks, no pretrained tensor load."""
    model_config=checked_model_config(c['model_config'])
    small=[c[k]for k in ('implementation','capacity_helper','cohort','orders','fit_view_epochs','source_plan','input_packet','accepted_input_complete','original30016_groups','original938_byte_copy','original_production_config','original_production_publication')]
    small.extend(c['inputs'].values())
    for ref in small:checked(ref)
    source=c['source_checkpoint'];p=Path(source['path'])
    if not {'path','bytes','sha256'}<=set(source)or p.stat().st_size!=source['bytes']:
        raise ValueError('Actual source checkpoint descriptor/stat differs')
    return dict(status='ACTUAL_SMALL_STARTUP_DESCRIPTORS_AND_MODEL_SHA_GUARDS_PASSED',
        descriptor_files=len(small)+1,model_config=model_config,
        source_checkpoint_actual_bytes=p.stat().st_size,
        source_checkpoint_full_SHA_deferred_to_owned_GPU_allocation=True,
        model_tensors_loaded=False,CUDA_used=False)


def memory_plan(fit_questions=FIT, holdout_questions=HELD):
    if (fit_questions, holdout_questions) != (FIT, HELD):
        raise ValueError('Exactly original30016 fit and new5120 internal holdout required')
    trajectories = (fit_questions+holdout_questions)*10
    hidden = trajectories*20*3584*2
    mask = trajectories*20
    logits = trajectories*4
    return dict(hidden_shape=[trajectories, 20, 3584], hidden_dtype='bfloat16',
        hidden_bytes=hidden, mask_bytes=mask, frozen_logit_bytes=logits,
        GPU_cache_bytes=hidden+mask+logits,
        GPU_reserve_after_cache_required_bytes=16*1024**3,
        CPU_memory_request_bytes=128*1024**3,
        disk_free_before_cache_required_bytes=hidden+8*1024**3,
        native_cache_micro=8, cache_chunk_trajectories=256,
        fit_updates_per_epoch=938, fit_updates_total=18760,
        prior_actual_cache=dict(questions=1536, trajectories=15360,
            native_cache_seconds=173.73449110984802),
        linear_cache_planning_seconds=173.73449110984802*(FIT+HELD)/1536,
        timing_is_extrapolation_not_actual_qualification=True)


def domain(group):
    value=group.get('domain',group.get('data_source'))
    if value is None:
        try:value=json.loads(group['qid'])[0]
        except (ValueError,TypeError,KeyError,IndexError):pass
    if value in ('nq','NQ','natural_questions'):return 'NQ'
    if value in ('hotpotqa','HotpotQA','hotpot'):return 'HotpotQA'
    raise ValueError('Explicit NQ or HotpotQA metadata required')


def validate_cohorts(plan, groups, selection, held_groups):
    """Identity first; holdout scores always use its committed new-seed five."""
    fit=[q for batch in plan['batches'] for q in batch]
    if (len(plan['batches'])!=938 or any(len(b)!=32 for b in plan['batches'])
            or len(fit)!=FIT or len(set(fit))!=FIT):
        raise ValueError('Unchanged original938×32 unique-question plan required')
    lookup={g['qid']:g for g in groups}
    if (len(lookup)!=len(groups) or any(not isinstance(q,str)for q in lookup)
            or not set(fit)<=set(lookup)):
        raise ValueError('Distinct original fit identities missing')
    rows=selection['groups'];held=[r['qid']for r in rows]
    if len(held)!=HELD or len(set(held))!=HELD or set(fit)&set(held):
        raise ValueError('5120 distinct holdout questions disjoint from original30016 required')
    fixed={r['qid']:r for r in held_groups}
    if set(fixed)!=set(held):raise ValueError('Committed holdout question coverage differs')
    counts={'NQ':0,'HotpotQA':0}
    for q in fit+held:
        g=lookup[q]
        if (len(g['trajectory_ids'])!=10 or len(set(g['trajectory_ids']))!=10
                or len(g['labels'])!=10 or any(type(y)is not int or y not in (0,1)for y in g['labels'])):
            raise ValueError('Ten genuine unique own-EM trajectories required')
    for row in rows:
        q=row['qid'];g=lookup[q];f=fixed[q]
        if (row.get('source_indices')!=[5,6,7,8,9]
                or row['trajectory_ids']!=g['trajectory_ids'][5:]
                or f['trajectory_ids']!=row['trajectory_ids']
                or f['labels']!=g['labels'][5:]
                or row['question_hash']!=g['question_hash']
                or f['question_hash']!=g['question_hash']):
            raise ValueError('Holdout must retain exact committed new5 own IDs/EM')
        counts[domain(row)]+=1
    if counts!={'NQ':2560,'HotpotQA':2560}:
        raise ValueError('Exactly2560 NQ and2560 HotpotQA holdout required')
    fit_hashes={lookup[q]['question_hash']for q in fit}
    if any(lookup[q]['question_hash']in fit_hashes for q in held):
        raise ValueError('Train/holdout question-hash overlap')
    return lookup,fit,held,counts


def epoch_orders(fit, original_batches):
    """All20 epochs retain the original938 question batches/order unchanged."""
    if fit!=[q for batch in original_batches for q in batch]:
        raise ValueError('Original question order must remain exact')
    return [copy.deepcopy(original_batches)for _ in range(EPOCHS)]


def prepare(packet_path,target=TARGET):
    target=Path(target)
    if target!=TARGET:raise ValueError('Separate preregistered owned target required')
    packet=read(packet_path)
    required=('merged_groups','merged_features_layer26','internal_holdout_selection',
              'internal_holdout_groups','identity_exclusion_acceptance')
    if any(k not in packet for k in required):raise ValueError('Actual complete holdout/source descriptors required; no placeholders')
    for k in required:checked(packet[k])
    exclusion=read(checked(packet['identity_exclusion_acceptance']))
    if (exclusion.get('holdout_controlled6528_qid_overlap')!=0
            or exclusion.get('holdout_controlled6528_hash_overlap')!=0
            or exclusion.get('holdout_original30016_qid_overlap')!=0
            or exclusion.get('holdout_original30016_hash_overlap')!=0
            or any(v!=0 for v in exclusion.get('holdout_fixed_evaluation_qid_and_hash_overlap',{'missing':1}).values())
            or any(v!=0 for v in exclusion.get('merged_fixed_evaluation_qid_and_hash_overlap',{'missing':1}).values())
            or exclusion.get('labels_consulted_for_selection')is not False
            or exclusion.get('scores_consulted_for_selection')is not False):
        raise ValueError('Identity-only exclusion and label-blind holdout acceptance required')
    c0=read(PRODUCTION/'config.json');publication=read(PRODUCTION/'source-publication.json')
    model_config_guard=checked_model_config(c0['model_config'])
    if (proof(PRODUCTION/'config.json')['sha256']!=old.CONFIG_SHA
            or publication['commit']!='501cf80478b77f68c3e13a96d4659e3da0971b2e'):
        raise ValueError('Exact original control metadata required')
    plan=read(checked(c0['plan']));groups=read(checked(packet['merged_groups']))['train']
    selection=read(checked(packet['internal_holdout_selection']))
    held_groups=read(checked(packet['internal_holdout_groups']))['internal_holdout']
    lookup,fit,held,counts=validate_cohorts(plan,groups,selection,held_groups)
    original=read(checked(packet['original30016_fit']))['train']
    if ({g['qid']:g for g in original}!={q:lookup[q]for q in fit}
            or proof(checked(packet['original938_plan']))['sha256']!=c0['plan']['sha256']):
        raise ValueError('Original30016 records and full938 byte-identical plan must be retained')
    checked(packet['complete'])
    target.mkdir(parents=True,exist_ok=True)
    immutable(target/'cohort.json',dict(fit_qids=fit,holdout_qids=held,
        fit_qids_sha256=digest(fit),holdout_qids_sha256=digest(held),
        holdout_domain_counts=counts,fit_questions=FIT,holdout_questions=HELD,
        both_question_and_question_hash_overlap=0,
        all_question_texts_seen_by_original_warmup=True,
        heldout_primary_source_indices=[5,6,7,8,9],
        role='TRAINING_POOL_INTERNAL_CAPACITY_DIAGNOSTIC_NOT_EXTERNAL'))
    immutable(target/'orders.json',dict(epochs=epoch_orders(fit,plan['batches']),seed=1729))
    fit_epoch={q:plan['batch_epochs'][i] for i,b in enumerate(plan['batches'])for q in b}
    immutable(target/'fit-view-epochs.json',fit_epoch)
    c=dict(kind='FROZEN_E1_LARGE_COHORT_HEAD_ONLY_SINGLE_LR_CAPACITY',
        original_production_config=proof(PRODUCTION/'config.json'),
        original_production_publication=proof(PRODUCTION/'source-publication.json'),
        original_runtime=publication['code_root'],source_checkpoint=c0['source_checkpoint'],
        implementation=proof(__file__),capacity_helper=proof(old.__file__),
        model_dir=c0['model_dir'],model_config=model_config_guard,source_plan=c0['plan'],
        input_packet=proof(packet_path),inputs={k:packet[k]for k in required},
        accepted_input_complete=packet['complete'],original30016_groups=packet['original30016_fit'],
        original938_byte_copy=packet['original938_plan'],
        cohort=proof(target/'cohort.json'),orders=proof(target/'orders.json'),
        fit_view_epochs=proof(target/'fit-view-epochs.json'),grouping=c0['grouping'],
        head_seed=1729,head_Adam0=True,head_LR=1e-4,head_weight_decay=0.,
        legacy_frozen=True,legacy_optimizer_created=False,legacy_updates=0,actor_updates=0,
        fit_questions=30016,internal_holdout_questions=5120,
        epochs=20,questions_per_update=32,updates_per_epoch=938,update_budget=18760,
        EMA_decay=.99,evaluation_epochs=list(NODES),prefix_tokens=20,question_max_tokens=273,
        fit_views_per_question=4,fit_view_weight='1/4',
        primary_fit_view='lexicographically smallest original source_indices, label-blind',
        primary_holdout_view='precommitted own newseed5 source_indices5..9',
        raw_and_EMA_separate=True,domain_metrics=True,ranking_coverage=.25,
        encode_arithmetic='CUDA_BF16_autocast_then_float_embedding',
        classifier_arithmetic='FP32_outside_autocast',memory=memory_plan(),
        external_pool_access=False,final_pool_access=False,
        diagnostic_only_not_production_candidate_or_external_selection=True)
    immutable(target/'config.json',c)
    qualification=startup_metadata_guards(c)
    immutable(target/'actual-startup-metadata-qualification.json',qualification)
    result=dict(status='LARGE_FROZEN_E1_CAPACITY_METADATA_PREPARED_NO_SUBMISSION',
        config=proof(target/'config.json'),memory=c['memory'],fit_questions=FIT,
        holdout_questions=HELD,holdout_domain_counts=counts,
        actual_GPU_or_model_tensor_loaded=False,GPU_submitted=False,
        actual_startup_metadata_qualification=proof(target/'actual-startup-metadata-qualification.json'))
    immutable(target/'preparation.json',result)
    return result


def train(config_sha):
    """Implemented entry, deliberately absent from any submit helper."""
    import torch
    import torch.nn.functional as F
    from . import direct_hidden_count as d,probe_ablation_train as u,g10_grouping as g
    from .model import Probe
    from .replay_feature_reader import ReplayFeatureReader
    global STOP
    def stop(*_):
        global STOP
        STOP=True
    for sig in (signal.SIGTERM,signal.SIGINT,signal.SIGUSR1):signal.signal(sig,stop)
    c=read(TARGET/'config.json')
    if proof(TARGET/'config.json')['sha256']!=config_sha:raise ValueError('Actual immutable config SHA required')
    if proof(__file__)!=c['implementation']or proof(old.__file__)!=c['capacity_helper']:
        raise ValueError('Prepared implementation/helper differs')
    startup_metadata_guards(c)
    if not torch.cuda.is_available()or torch.cuda.device_count()!=1 or 'H100'not in torch.cuda.get_device_properties(0).name:
        raise ValueError('Explicit owned one-H100 allocation required')
    if read(TARGET/'preparation.json')['config']!=proof(TARGET/'config.json'):raise ValueError('Prepared config changed')
    if (TARGET/'complete.json').exists():return read(TARGET/'complete.json')
    device=torch.device('cuda',0);torch.cuda.set_device(device)
    torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False
    os.environ['QWEN7B_QUESTION_MAX_TOKENS']='273';os.environ['SEARCH_G1_B_PREFIX_TOKENS']='20'
    cohort=read(checked(c['cohort']));fit=cohort['fit_qids'];held=cohort['holdout_qids'];qids=fit+held
    groups=read(checked(c['inputs']['merged_groups']))['train'];lookup={x['qid']:x for x in groups}
    fixed={x['qid']:x for x in read(checked(c['inputs']['internal_holdout_groups']))['internal_holdout']}
    orders=read(checked(c['orders']))['epochs'];view_epochs=read(checked(c['fit_view_epochs']))
    views={q:g.build_training_views([lookup[q]],role='train',epoch=view_epochs[q],**c['grouping'])['groups']for q in fit}
    primary={q:old.fixed_view(views[q])for q in fit}
    if any(len(v)!=4 for v in views.values()):raise ValueError('Four original fit views required')
    memory=c['memory'];n=len(qids)*10;cache_root=TARGET/'cache';cache_root.mkdir(exist_ok=True)
    if shutil.disk_usage(TARGET).free<memory['disk_free_before_cache_required_bytes']and not(cache_root/'complete.json').exists():
        raise ValueError('Enough owned remote disk required before native cache')
    head,ema,opt=d.initialize_head(seed=1729,device=device);opt.param_groups[0]['lr']=1e-4
    initial_head_hashes=d.new_head_hashes(head,ema,opt)
    source=torch.load(checked(c['source_checkpoint']),map_location='cpu',mmap=True,weights_only=False)
    if source['steps']!=10464 or source['epoch']!=1 or not source['completed_epoch']:raise ValueError('Original E1 required')
    archive=source['rank_states']
    legacy=Probe(c['model_dir'],273,20).to(device)
    legacy.load_state_dict(source['raw'],strict=True);u.restore_buffers(legacy,source['raw_buffers']);legacy.eval().requires_grad_(False)
    u.restore_rng(next(x['rng']for x in archive if x['rank']==0),device)
    frozen=dict(raw=u.model_hash(legacy),buffers=u.tensor_hash(u.buffers(legacy)))
    if len(archive)!=4 or any(item['model_sha256']!=frozen['raw']for item in archive):
        raise ValueError('Exact E1 raw/four-rank source archive differs')
    del source
    files=dict(hidden=cache_root/'hidden.bf16',mask=cache_root/'mask.bool',logits=cache_root/'logits.f32')
    tids=[t for q in qids for t in lookup[q]['trajectory_ids']]
    def mappings(create=False):
        sizes=dict(hidden=memory['hidden_bytes'],mask=memory['mask_bytes'],logits=memory['frozen_logit_bytes'])
        for k,path in files.items():
            if create and not path.exists():
                with path.open('wb')as f:f.truncate(sizes[k])
            if path.stat().st_size!=sizes[k]:raise ValueError('Exact cache size required')
        return (torch.from_file(str(files['hidden']),shared=True,size=n*20*3584,dtype=torch.bfloat16).reshape(n,20,3584),
                torch.from_file(str(files['mask']),shared=True,size=n*20,dtype=torch.bool).reshape(n,20),
                torch.from_file(str(files['logits']),shared=True,size=n,dtype=torch.float32))
    H,M,Z=mappings(create=True)
    if not(cache_root/'complete.json').exists():
        refs=read(checked(c['inputs']['merged_features_layer26']));need=set(tids);selected=[]
        # Small index metadata chooses required native feature chunks; the strict
        # reader still verifies packed SHA once and retained own-row SHA on get.
        for ref in refs:
            manifest=read(checked(ref));index=Path(manifest['files']['index.jsonl']['path'])
            with index.open()as f:hit=any(json.loads(line)['trajectory_id']in need for line in f)
            if hit:selected.append(ref)
        reader=ReplayFeatureReader(selected,26)
        if not need<=set(reader.records):raise ValueError('Selected true IDs missing from accepted features')
        class Recorder:
            def encode(self,h,m):self.hidden=h.detach();self.mask=m.detach();return head.encode(h,m)
        recorder=Recorder();cursor=0
        if(cache_root/'progress.json').exists():
            prior=read(cache_root/'progress.json')
            if prior['config_sha256']!=config_sha or prior['trajectory_ids_sha256']!=digest(tids):raise ValueError('Partial cache source differs')
            cursor=prior['completed_trajectories']
            for ref in prior['completed_chunks']:checked(ref)
        else:prior={'completed_chunks':[]}
        import hashlib
        chunks=list(prior['completed_chunks'])
        for ref in chunks:
            part=read(checked(ref))
            for key,span in part['spans'].items():
                with files[key].open('rb')as f:
                    f.seek(span['offset']);raw=f.read(span['bytes'])
                if hashlib.sha256(raw).hexdigest()!=span['sha256']:
                    raise ValueError('Partial cache accepted span changed; no silent prefix rollback')
        started=time.time();golden_done=(cache_root/'same-api-golden.json').exists()
        for start in range(cursor,n,256):
            end=min(n,start+256);context=[]
            for offset in range(start,end,8):
                ids=tids[offset:min(end,offset+8)];batch=reader.get(ids,device)
                with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):z,embedding=d.native_hidden_forward(legacy,batch,recorder)
                if recorder.hidden.dtype!=torch.bfloat16:raise ValueError('Native hidden dtype changed')
                H[offset:offset+len(ids)].copy_(recorder.hidden.cpu());M[offset:offset+len(ids)].copy_(recorder.mask.cpu());Z[offset:offset+len(ids)].copy_(z.cpu())
                if not golden_done:
                    h0=recorder.hidden.clone();m0=recorder.mask.clone();z0=z.clone();e0=embedding.clone()
                    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):z1,e1=d.native_hidden_forward(legacy,batch,recorder)
                    if not all((torch.equal(h0,recorder.hidden),torch.equal(m0,recorder.mask),torch.equal(z0,z1),torch.equal(e0,e1),torch.equal(H[offset:offset+len(ids)].to(device),h0),torch.equal(M[offset:offset+len(ids)].to(device),m0))):
                        raise ValueError('Same-shape cached/native actual hidden replay differs')
                    immutable(cache_root/'same-api-golden.json',dict(status='PASSED',IDs_sha256=digest(ids),mask_positions_sha256=u.tensor_hash({k:batch[k]for k in ('attention_mask','position_ids')}),dtype='bfloat16',micro=8,extra_quantization=False))
                    golden_done=True
                context.append(dict(ids_sha256=digest(ids),mask_positions_sha256=u.tensor_hash({k:batch[k]for k in ('attention_mask','position_ids')})))
                del batch,z,embedding
            # Payload is the memory-mapped remote raw files; each accepted span
            # has an independent exact bytes SHA, so resume verifies prefixes.
            spans={}
            for k,itemsize,width in [('hidden',2,20*3584),('mask',1,20),('logits',4,1)]:
                with files[k].open('rb')as f:
                    os.fsync(f.fileno())
                    f.seek(start*width*itemsize);raw=f.read((end-start)*width*itemsize)
                spans[k]=dict(offset=start*width*itemsize,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
            item=dict(start=start,end=end,trajectory_ids_sha256=digest(tids[start:end]),spans=spans,contexts_sha256=digest(context))
            p=cache_root/('chunk-%06d.json'%len(chunks));immutable(p,item);chunks.append(proof(p))
            atomic(cache_root/'progress.json',dict(config_sha256=config_sha,trajectory_ids_sha256=digest(tids),completed_trajectories=end,completed_chunks=chunks,epoch=time.time()))
            atomic(TARGET/'progress.json',dict(stage='NATIVE_FROZEN_E1_CACHE_ADVANCING',trajectories=end,total=n,elapsed=time.time()-started,epoch=time.time()))
            if STOP:return dict(status='CACHE_INTERRUPTED_DURABLE_CHUNKS_PRESERVED',completed_trajectories=end)
        reader.close();del reader,recorder
        if frozen!=dict(raw=u.model_hash(legacy),buffers=u.tensor_hash(u.buffers(legacy))):raise ValueError('Frozen E1 legacy changed')
        immutable(cache_root/'complete.json',dict(status='FROZEN_E1_LARGE_NATIVE_CACHE_COMPLETE',files={k:proof(p)for k,p in files.items()},chunks=chunks,config_sha256=config_sha,trajectory_ids_sha256=digest(tids),frozen_legacy_before=frozen,frozen_legacy_after=frozen,actual_native_hidden_dtype='bfloat16',same_api_golden=proof(cache_root/'same-api-golden.json'),actor_updates=0,legacy_updates=0))
    cache=read(cache_root/'complete.json')
    if cache['config_sha256']!=config_sha or cache['frozen_legacy_before']!=frozen or cache['trajectory_ids_sha256']!=digest(tids):raise ValueError('Actual cache lineage differs')
    for ref in cache['files'].values():checked(ref)
    del legacy
    torch.cuda.empty_cache()
    free,total=torch.cuda.mem_get_info(device)
    if free<memory['GPU_cache_bytes']+memory['GPU_reserve_after_cache_required_bytes']:
        raise ValueError('Measured GPU free memory cannot safely hold actual cache plus16GiB reserve; do not shrink/truncate silently')
    H=H.to(device);M=M.to(device);Z=Z.to(device);byq={q:i for i,q in enumerate(qids)}
    def encoded(model,qs):
        indices=torch.tensor([byq[q]*10+j for q in qs for j in range(10)],device=device)
        with torch.autocast('cuda',dtype=torch.bfloat16):emb=model.encode(H[indices],M[indices])
        return emb.reshape(len(qs),10,128),Z[indices].reshape(len(qs),10)
    def hashes():return dict(raw=u.model_hash(head),ema=u.model_hash(ema),adam=u.optimizer_hash(opt),rng=u.rng_hash(u.rng_state(device)))
    def save(cursor,epoch):
        state=dict(kind='LARGE_FROZEN_E1_HEAD_ONLY_FULL_STATE',head=d.new_head_state(head,ema,opt),rng=u.rng_state(device),cursor=cursor,epoch=epoch,
            config=proof(TARGET/'config.json'),cache=proof(cache_root/'complete.json'),source_all_four_rank_states=archive,frozen_legacy_hashes=frozen,
            initial_head_hashes=initial_head_hashes,accepted_hashes=hashes(),data_cursor=dict(cursor=cursor,orders_sha256=c['orders']['sha256']))
        p=TARGET/'checkpoints'/('cursor-%06d.pt'%cursor);p.parent.mkdir(exist_ok=True);tmp=p.with_suffix('.tmp');torch.save(state,tmp)
        with tmp.open('rb')as f:os.fsync(f.fileno())
        tmp.replace(p);ref=proof(p);loaded=torch.load(p,map_location='cpu',mmap=True,weights_only=False);s=loaded['head']
        values=dict(raw=u.model_mapping_hash(s['raw']),ema=u.model_mapping_hash(s['ema']),adam=u.optimizer_blob_hash(s['optimizer']),rng=u.rng_hash(loaded['rng']))
        if values!=state['accepted_hashes']or s['steps']!=cursor:raise ValueError('Full head CPU reload differs')
        receipt=dict(**ref,cursor=cursor,epoch=epoch,accepted_hashes=values,full_CPU_reload_passed=True,new_head_Adam_steps=cursor)
        atomic(TARGET/'latest.json',receipt);return receipt
    def update(qs):
        head.train();opt.zero_grad(set_to_none=True);emb,z=encoded(head,qs)
        loss=d.count_loss(z,emb,[lookup[q]for q in qs],[v for q in qs for v in views[q]],head)
        loss['count_NLL'].backward()
        if not all(p.grad is not None and bool(torch.isfinite(p.grad).all())for p in head.parameters()):raise ValueError('Nonfinite/missing count-head gradient')
        opt.step()
        with torch.no_grad():
            for dst,src in zip(ema.parameters(),head.parameters()):dst.mul_(.99).add_(src,alpha=.01)
        return dict(count_NLL=float(loss['count_NLL'].detach()),frozen_trajectory_BCE=float(loss['trajectory_loss'].detach()))
    def summary(rows):
        scores=[r['q']for r in rows];ys=[0<r['K']<5 for r in rows];pred=[s>=.5 for s in scores];pos=sum(ys);neg=len(ys)-pos
        tpr=sum(y and p for y,p in zip(ys,pred))/pos if pos else None;tnr=sum(not y and not p for y,p in zip(ys,pred))/neg if neg else None
        top=sorted(rows,key=lambda r:(-r['q'],r['qid']))[:len(rows)//4]
        return dict(questions=len(rows),count_NLL=sum(r['count_NLL']for r in rows)/len(rows),count_accuracy=sum(r['predicted_K']==r['K']for r in rows)/len(rows),K_histogram=[sum(r['K']==k for r in rows)for k in range(6)],Mixed_BA=(tpr+tnr)/2 if pos and neg else None,Mixed_recall=tpr,nonMixed_recall=tnr,Mixed_AUC=u.roc_auc(scores,ys),predicted_Mixed_fraction=sum(pred)/len(rows),Top25percent=dict(questions=len(top),Mixed_count=sum(0<r['K']<5 for r in top),Mixed_fraction=sum(0<r['K']<5 for r in top)/len(top)))
    def score(model,qs,all4=False):
        rows=[];nll=0.;correct=0;total=0
        with torch.no_grad():
            for start in range(0,len(qs),32):
                part=qs[start:start+32];emb,_=encoded(model,part);items=[];indices=[]
                for i,q in enumerate(part):
                    selected=views[q]if all4 else [primary[q]if q in primary else fixed[q]]
                    for v in selected:
                        own=[lookup[q]['trajectory_ids'].index(t)for t in v['trajectory_ids']]
                        items.append((q,v));indices.append([i*10+j for j in own])
                idx=torch.tensor(indices,device=device);logits=model.count_logits(emb.reshape(-1,128)[idx])
                target=torch.tensor([sum(v['labels'])for q,v in items],device=device)
                losses=F.cross_entropy(logits,target,reduction='none');prob=logits.softmax(-1)
                nll+=float(losses.sum());correct+=int((logits.argmax(-1)==target).sum());total+=len(items)
                for(q,v),s,loss_value,predicted_k in zip(items,prob[:,1:5].sum(-1).cpu().tolist(),losses.cpu().tolist(),logits.argmax(-1).cpu().tolist()):
                    rows.append(dict(qid=q,q=s,K=sum(v['labels']),domain=domain(lookup[q]),count_NLL=loss_value,predicted_K=predicted_k))
        result=dict(count_NLL=nll/total,count_accuracy=correct/total,questions=len(qs),G5_views=total,related_views_not_independent_questions=all4)
        if not all4:
            result.update(summary(rows));result['by_domain']={name:summary([r for r in rows if r['domain']==name])for name in ('NQ','HotpotQA')if any(r['domain']==name for r in rows)}
        return result,rows
    def evaluate(epoch,cp):
        before=hashes();result={};outputs={};folder=TARGET/'evaluations'/('epoch-%02d'%epoch);folder.mkdir(parents=True,exist_ok=True)
        for name,model in [('raw',head),('EMA',ema)]:
            result[name]={}
            for label,qs,all4 in [('fit_all4',fit,True),('fit_primaryG5',fit,False),('internal_holdout_new5',held,False)]:
                value,rows=score(model,qs,all4);result[name][label]=value;p=folder/(name+'-'+label+'.jsonl.gz')
                with gzip.open(p,'wt')as f:
                    for row in rows:f.write(json.dumps(row)+'\n')
                outputs[name+'-'+label]=proof(p)
        if before!=hashes():raise ValueError('Read-only evaluation changed full head state')
        done=dict(epoch=epoch,cursor=epoch*938,checkpoint=cp,metrics=result,predictions=outputs,training_components_unchanged=True,external_pool_access=False,final_pool_access=False)
        immutable(folder/'complete.json',done);return done
    cursor=0;results=[]
    if(TARGET/'latest.json').exists():
        ref=read(TARGET/'latest.json');loaded=torch.load(checked(ref),map_location='cpu',mmap=True,weights_only=False)
        if loaded['config']!=proof(TARGET/'config.json')or loaded['cache']!=proof(cache_root/'complete.json'):raise ValueError('Resume full head source mismatch')
        d.restore_new_head(loaded['head'],head,ema,opt,device);u.restore_rng(loaded['rng'],device);cursor=loaded['cursor']
        if hashes()!=loaded['accepted_hashes']or d.optimizer_steps(opt)!=cursor:raise ValueError('Resume state/cursor mismatch')
    else:
        cp=save(0,0);evaluate(0,cp)
    if cursor==0:
        state=copy.deepcopy(d.new_head_state(head,ema,opt));rng=u.rng_state(device)
        update(orders[0][0]);expected=hashes();d.restore_new_head(state,head,ema,opt,device);u.restore_rng(rng,device);update(orders[0][0])
        if hashes()!=expected:raise ValueError('First complete count-head update replay differs')
        immutable(TARGET/'first-update-replay.json',dict(status='PASSED',expected=expected,actual=hashes(),Adam_before=0,Adam_after=1))
        cursor=1;save(1,0)
    for ep,order in enumerate(orders,1):
        if cursor>=ep*938:
            if ep in NODES and not(TARGET/'evaluations'/('epoch-%02d'%ep)/'complete.json').exists():
                if cursor!=ep*938:raise ValueError('Past fixed evaluation missing; do not label future weights as a historical epoch')
                evaluate(ep,save(cursor,ep))
            continue
        for bi,qs in enumerate(order):
            step=(ep-1)*938+bi+1
            if step<=cursor:continue
            if STOP:
                cp=save(cursor,(cursor//938));atomic(TARGET/'interrupted.json',dict(cursor=cursor,checkpoint=cp));return dict(status='INTERRUPTED_FULL_HEAD_STATE_SAVED')
            value=update(qs);cursor=step
            atomic(TARGET/'progress.json',dict(stage='FROZEN_E1_LARGE_HEAD_FIT_ADVANCING',epoch=ep,cursor=cursor,head_Adam=d.optimizer_steps(opt),**value,time=time.time()))
            if cursor<=6 or cursor%128==0:save(cursor,cursor//938)
        cp=save(cursor,ep)
        if ep in NODES and not(TARGET/'evaluations'/('epoch-%02d'%ep)/'complete.json').exists():evaluate(ep,cp)
    if cursor!=18760 or d.optimizer_steps(opt)!=18760:raise ValueError('Actual20epoch head budget incomplete')
    for ep in NODES:results.append(read(TARGET/'evaluations'/('epoch-%02d'%ep)/'complete.json'))
    done=dict(status='LARGE_FROZEN_E1_COUNT_HEAD20EPOCH_COMPLETE',config=proof(TARGET/'config.json'),cache=proof(cache_root/'complete.json'),checkpoint=cp,evaluations=results,
        frozen_legacy_hashes=frozen,legacy_updates=0,actor_updates=0,head_Adam=18760,max_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),
        max_GPU_reserved_bytes=torch.cuda.max_memory_reserved(),external_pool_access=False,final_pool_access=False,capacity_diagnostic_not_external_candidate=True)
    immutable(TARGET/'complete.json',done);return done


def main():
    p=argparse.ArgumentParser();p.add_argument('--action',choices=('memory-plan','prepare','train'),required=True)
    p.add_argument('--input-packet');p.add_argument('--config-sha256');args=p.parse_args()
    if args.action=='memory-plan':print(json.dumps(memory_plan()));return
    TARGET.mkdir(exist_ok=True,parents=True)
    with(TARGET/('prepare.lock'if args.action=='prepare'else'head-only-training.lock')).open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            value=prepare(args.input_packet)if args.action=='prepare'else train(args.config_sha256)
            print(json.dumps(value),flush=True)
        except BaseException as exc:
            if args.action=='train':atomic(TARGET/'failure.json',dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc(),epoch=time.time()))
            raise


if __name__=='__main__':main()
