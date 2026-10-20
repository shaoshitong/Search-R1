"""Independent Prefix20 feature-only replay of accepted peer raw G10.

Explicit budget8192, hash-random seed1729; exclude controlled6528 and evaluation
identities before choosing. Both old5/new5 use the SAME qualified HF causal
backend. No full response/EM files, retriever, new rollout or training is used.
"""
import argparse
import fcntl
import gzip
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from . import replay_probe_layers as r
from . import peer_newseed_ingest as raw
from .replay_bank_prepare import q1_model_binding
from .replay_bank_capture import verify_completed_chunk

QUESTIONS=8192
SEED=1729
SHARDS=4
B=raw.B
A=raw.A
TARGET=A/'peer-newseed8192-g10-replay-20261021'
STATUS='PEER_RAW_G10_FROZEN_HF_MULTILAYER_CAUSAL_REPLAY_SHARD_COMPLETE'


def read(path):return raw.read(path)
def ref(path):return raw.ref(path)
def checked(item):return raw.checked(item)
def immutable(path,value):return raw.immutable(path,value)
def digest(value):return raw.digest(value)


def choose(groups,controlled,exclusions,*,budget=QUESTIONS,seed=SEED):
    """Only qid/hash/ID projections are read; labels cannot affect selection."""
    if type(budget) is not int or budget<1 or type(seed) is not int:raise ValueError('Explicit positive identity-only budget required')
    rows=[dict(qid=g['qid'],question_hash=g['question_hash'],trajectory_ids=list(g['trajectory_ids'])) for g in groups]
    ids=[g['qid'] for g in rows]
    if len(ids)!=len(set(ids)):raise ValueError('Accepted raw training identity repeats')
    excluded_ids=set();excluded_hashes=set()
    for own in [controlled]+list(exclusions):
        mapping,hashes=raw.identity_rows(own);excluded_ids.update(mapping);excluded_hashes.update(hashes)
    eligible=[g for g in rows if g['qid'] not in excluded_ids and g['question_hash'] not in excluded_hashes]
    if len(eligible)<budget:raise ValueError('Insufficient eligible peer training questions: available=%d required=%d'%(len(eligible),budget))
    result=sorted(eligible,key=lambda g:(digest(['peer-newseed-independent8192',seed,g['qid']]),g['qid']))[:budget]
    seen=set()
    for group in result:
        tids=group['trajectory_ids']
        if len(tids)!=10 or len(set(tids))!=10 or seen.intersection(tids):raise ValueError('Ten complete authentic old5/new5 IDs required')
        seen.update(tids)
    return result,dict(eligible_questions=len(eligible),selected_questions=budget,selected_trajectories=budget*10,
        ordered_qid_sha256=digest([g['qid'] for g in result]),trajectory_ids_sha256=digest([t for g in result for t in g['trajectory_ids']]),
        question_selection='SHA256(seed1729,qid) random, no labels/probe score',labels_consulted=False,
        controlled6528_and_eval_qids_and_hashes_excluded=True,final_labels_or_rollouts_read=False)


def projected_rows(prefix_paths,wanted,*,verify_sha=True):
    """Only sealed Prefix20 records. No full or own-EM refs are opened here."""
    seen=set();proofs=[]
    for item in prefix_paths:
        path=checked(item) if verify_sha else Path(item['path']);proofs.append(ref(path))
        with gzip.open(path,'rt') as stream:
            for line in stream:
                row=json.loads(line);tid=row['trajectory_id']
                if tid not in wanted:continue
                qid=wanted[tid]
                if tid in seen or row['qid']!=qid:raise ValueError('Repeated/wrong own selected prefix identity')
                r.normalize_prefix(row,tid,qid,True)  # exact qualified causal normalization
                if any(key in row for key in ('strict_em','labels','reward_model','ground_truth','response_token_ids')):
                    raise ValueError('A prefix-only feature input may not carry full/EM labels')
                seen.add(tid);yield row
    if seen!=set(wanted):raise ValueError('Accepted raw G10 is missing actual selected prefix rows')


def dense_upper_bytes(questions):
    # Conservative denseBF16 retained slots; lossless question/XOR packing is
    # usually much smaller, but disk safety does not assume a compression ratio.
    return questions*10*3*r.WIDTH*r.DIM*2


def prepare(base,raw_complete,controlled_plan,code_root,source_commit,*,budget=QUESTIONS,seed=SEED,minimum_free_bytes=600*1024**3):
    base,code_root=map(r.remote,(base,code_root));complete_path=checked(raw_complete);accepted=read(complete_path)
    if (base!=TARGET or budget!=QUESTIONS or seed!=SEED or accepted.get('status')!=raw.STATUS or
            accepted.get('feature_bank_created') is not False or accepted.get('actor_updates')!=0 or accepted.get('probe_updates')!=0):
        raise ValueError('Independent declared8192/seed1729 raw-complete target required')
    if len(source_commit)!=40:raise ValueError('Exact published source commit required')
    if base.exists() and (base/'preparation.json').exists():
        result=read(base/'preparation.json')
        if result['raw_complete']!=raw_complete or result['source_commit']!=source_commit:raise ValueError('Prepared source binding differs')
        for item in result['selections']:checked(item)
        return result
    groups=read(checked(accepted['outputs']['groups']))['train']
    plan=read(checked(controlled_plan));cohort=plan['cohort_qids']
    if len(cohort)!=6528 or len(set(cohort))!=6528:raise ValueError('Actual controlled6528 exclusion plan required')
    conf=read(checked(accepted['config']));original=read(checked(conf['original_bank']))['train'];lookup={g['qid']:g for g in original}
    controlled=[dict(qid=q,question_hash=lookup[q]['question_hash']) for q in cohort]
    exclusions=[v['groups'] for v in conf['exclusions'].values()]
    selected,identity=choose(groups,controlled,exclusions,budget=budget,seed=seed)
    q1,model_binding=q1_model_binding();source_events=read(checked(accepted['outputs']['source-events']))
    paths=[];seenpaths=set()
    for event in source_events['events']:
        for batch in event['batches']:
            item=batch['prefix']
            if item['path'] not in seenpaths:seenpaths.add(item['path']);paths.append(item)
    # Check real shared filesystem before writing any feature inputs. The user
    # requested ~500GB allowance;600GiB minimum leaves margin. Record dense bound.
    existing=base.parent
    while not existing.exists():existing=existing.parent
    free=shutil.disk_usage(existing).free
    if free<minimum_free_bytes:raise ValueError('Insufficient shared space: free=%d required=%d'%(free,minimum_free_bytes))
    base.mkdir(parents=True,exist_ok=True)
    with (base/'prepare.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        immutable(base/'identity-plan.json',dict(role='train',groups=selected,summary=identity,
            raw_complete=raw_complete,controlled6528=controlled_plan,labels_consulted=False))
        parts=[selected[i::SHARDS] for i in range(SHARDS)];tidshard={t:i for i,part in enumerate(parts) for g in part for t in g['trajectory_ids']}
        wanted={t:g['qid'] for g in selected for t in g['trajectory_ids']};counts=[0]*SHARDS;streams=[];temporary=[];prefix_refs=[]
        try:
            for shard in range(SHARDS):
                root=base/('shard-%02d'%shard);(root/'inputs').mkdir(parents=True,exist_ok=True);(root/'logs').mkdir(exist_ok=True)
                path=root/'inputs/prefix.jsonl.gz';temp=path.with_name('prefix.jsonl.gz.partial')
                if path.exists() or temp.exists():raise ValueError('Unknown prepared prefix partial preserved; do not overwrite')
                temporary.append((temp,path));streams.append(gzip.open(temp,'xb'))
            for row in projected_rows(paths,wanted):
                shard=tidshard[row['trajectory_id']];streams[shard].write(r.canonical(row)+b'\n');counts[shard]+=1
                if sum(counts)%256==0:raw.atomic(base/'prepare-progress.json',dict(stage='PREFIX_ONLY_SHA_PROJECTION',trajectories=sum(counts),expected=10*budget,time=time.time()))
        finally:
            for stream in streams:stream.close()
        selections=[]
        for shard,(temp,path) in enumerate(temporary):
            if counts[shard]!=len(parts[shard])*10:raise ValueError('Prepared shard incomplete actual G10')
            os.replace(str(temp),str(path));prefix_refs.append(ref(path))
            own=parts[shard];selection=dict(role='peer_newseed_train_G10',model_revision=r.REVISION,
                questions=len(own),trajectories=[dict(qid=g['qid'],trajectory_id=t) for g in own for t in g['trajectory_ids']],
                real_trajectories_per_question=10,label_blind_selection=True,prefix_tokens=20,capture_layers=list(r.LAYERS),
                prefix_files=[ref(path)],model_files=q1['model_files'],source_model_binding=model_binding,
                source_ingest=raw_complete,controlled_plan=controlled_plan,question_plan=ref(base/'identity-plan.json'),
                source_commit=source_commit,expected_code_root=str(code_root),shard=shard,shards=SHARDS,
                eval_backend='R2_existing_original_G5_diagnostic512_and_viewed_selection2048_only',native_bitexact_equivalence_claim=False)
            p=base/('shard-%02d'%shard)/'inputs/selection.json';immutable(p,selection);selections.append(ref(p))
        result=dict(status='PEER_RAW_G10_8192_PREFIX_ONLY_FOUR_SHARDS_PREPARED_NOT_CAPTURED',raw_complete=raw_complete,
            controlled_plan=controlled_plan,source_commit=source_commit,code_root=str(code_root),identity=identity,
            selections=selections,prefix_inputs=prefix_refs,source_prefix_files=paths,free_space_bytes=free,
            minimum_free_bytes=minimum_free_bytes,dense_uncompressed_upper_bytes=dense_upper_bytes(budget),
            questions=budget,trajectories=10*budget,layers=list(r.LAYERS),feature_bank_created=False,
            actor_updates=0,probe_updates=0,full_or_own_EM_files_opened=False,new_rollouts_generated=0,
            GPU_jobs_submitted=False,source_model_binding=model_binding,time=time.time())
        immutable(base/'preparation.json',result);return result


def load_selection(selection):
    declarations=selection['trajectories'];wanted={v['trajectory_id']:v['qid'] for v in declarations}
    if (selection.get('role')!='peer_newseed_train_G10' or selection.get('model_revision')!=r.REVISION or
            selection.get('capture_layers')!=list(r.LAYERS) or selection.get('real_trajectories_per_question')!=10 or
            selection.get('label_blind_selection') is not True or len(wanted)!=len(declarations)):
        raise ValueError('Sealed explicit actual G10 feature-only selection required')
    found={row['trajectory_id']:r.normalize_prefix(row,row['trajectory_id'],wanted[row['trajectory_id']],True)
        for row in projected_rows(selection['prefix_files'],wanted)}
    selected=[found[v['trajectory_id']] for v in declarations];counts={}
    for row in selected:counts[row['qid']]=counts.get(row['qid'],0)+1
    if len(counts)!=selection['questions'] or any(v!=10 for v in counts.values()):raise ValueError('Every feature shard must retain actual old5+new5')
    return selected


def capture(selection_path,selection_sha,output,device_name='cuda:0'):
    import torch
    if (not torch.cuda.is_available() or not os.environ.get('SLURM_JOB_ID') or not os.environ.get('CUDA_VISIBLE_DEVICES')):
        raise ValueError('Actual owned CUDA Slurm feature allocation required')
    source=r.remote(selection_path);base=r.remote(output);source_ref=ref(source)
    if source_ref['sha256']!=selection_sha:raise ValueError('Prepared prefix selection SHA differs')
    selection=read(source)
    if (str(Path(__file__).resolve().parents[3])!=selection['expected_code_root'] or
            ref(r.__file__)['sha256']!=selection['source_model_binding']['legacy_replay_module_sha256']):
        raise ValueError('Immutable source or qualified R2 replay primitive differs')
    base.mkdir(parents=True,exist_ok=True)
    with (base/'capture.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (base/'complete.json').exists():
            old=read(base/'complete.json')
            if old['source_selection']!=source_ref:raise ValueError('Existing feature bank belongs to different source')
            return old
        selected=load_selection(selection);model_dir=selection['source_model_binding']['official_view']
        proofs,config=r.verify_model(model_dir,selection)
        if any(token>=config['vocab_size'] for row in selected for seg in row['segments'] for token in seg['context_tokens']):raise ValueError('Actual token outside official vocabulary')
        from transformers import AutoModel
        device=torch.device(device_name);torch.cuda.set_device(device)
        model=AutoModel.from_pretrained(model_dir,torch_dtype=torch.bfloat16,attn_implementation='flash_attention_2',
            local_files_only=True,trust_remote_code=False,low_cpu_mem_usage=True).eval().requires_grad_(False).to(device)
        if len(model.layers)!=28 or model.config.hidden_size!=r.DIM or any(p.requires_grad for p in model.parameters()):raise ValueError('Frozen official model geometry changed')
        passes=0;started=time.monotonic();chunks=[]
        def causal(context,positions):
            nonlocal passes
            if positions!=list(range(len(context))):raise ValueError('Actual causal positions changed')
            with torch.inference_mode():
                ids=torch.tensor([context],dtype=torch.long,device=device)
                result=model(input_ids=ids,attention_mask=torch.ones_like(ids),position_ids=torch.tensor([positions],dtype=torch.long,device=device),
                    use_cache=False,output_hidden_states=True,return_dict=True)
                if len(result.hidden_states)!=29:raise ValueError('HF hidden state boundary differs')
                values={layer:result.hidden_states[layer+1][0].detach().to('cpu',dtype=torch.bfloat16,copy=True) for layer in r.LAYERS}
            passes+=1;return values
        for start in range(0,len(selected),256):
            own=selected[start:start+256];root=base/('chunk-%05d'%(start//256));tids=[row['trajectory_id'] for row in own]
            if (root/'complete.json').is_file():
                done=verify_completed_chunk(root,tids)
                if done['source_selection']!=source_ref:raise ValueError('Existing completed chunk source differs')
                chunks.append(ref(root/'complete.json'));continue
            if root.exists():raise ValueError('Unknown partial feature chunk preserved; needs explicit isolated retry')
            root.mkdir();writers={layer:r.ReplayLayerWriter(root/('layer-%02d'%layer),layer,own) for layer in r.LAYERS}
            for index,row in enumerate(own):
                values=r.replay_features(row,causal)
                for layer in r.LAYERS:writers[layer].write(row,values[layer])
                raw.atomic(base/'progress.json',dict(status='ACTUAL_PEER_G10_MULTILAYER_FEATURE_REPLAY_ADVANCING',
                    trajectories=start+index+1,expected_trajectories=len(selected),causal_prefill_passes_this_process=passes,
                    source_selection=source_ref,elapsed_seconds=time.monotonic()-started,actor_updates=0,
                    max_cuda_memory_allocated=torch.cuda.max_memory_allocated(device),time=time.time()))
            manifests={str(layer):writers[layer].finish() for layer in r.LAYERS}
            raw.atomic(root/'complete.json',dict(source_selection=source_ref,trajectory_ids=tids,feature_manifests=manifests,
                source_runtime='huggingface_frozen_causal_prefill',actor_updates=0,native_bitexact_claim=False))
            chunks.append(ref(root/'complete.json'))
        layers={str(layer):[] for layer in r.LAYERS};allids=[]
        for item in chunks:
            done=verify_completed_chunk(Path(item['path']).parent,read(item['path'])['trajectory_ids']);allids+=done['trajectory_ids']
            for layer in layers:layers[layer].append(done['feature_manifests'][layer])
        if allids!=[row['trajectory_id'] for row in selected]:raise ValueError('Whole actual peer feature shard incomplete')
        from .replay_feature_reader import ReplayFeatureReader
        for layer,refs in layers.items():
            reader=ReplayFeatureReader(refs,int(layer))
            if set(reader.records)!=set(allids):raise ValueError('Actual reader row coverage differs')
            reader.get([allids[0],allids[-1]],'cpu');reader.close()
        result=dict(status=STATUS,source_selection=source_ref,questions=selection['questions'],trajectories=len(selected),
            feature_manifests_by_capture_layer=layers,chunks=chunks,capture_layers=list(r.LAYERS),human_block_layers=[21,24,27],
            source_runtime='huggingface_frozen_causal_prefill',attention_implementation='flash_attention_2',model_revision=r.REVISION,model_files=proofs,
            actor_frozen=True,actor_updates=0,probe_updates=0,new_responses_generated=0,native_bitexact_equivalence_claim=False,
            eval_backend='R2_existing_original_G5_diagnostic512_and_viewed_selection2048_only',max_cuda_memory_allocated=torch.cuda.max_memory_allocated(device),
            job_id=os.environ['SLURM_JOB_ID'],elapsed_seconds=time.monotonic()-started,runtime_code=ref(__file__),qualified_primitive=ref(r.__file__),time=time.time())
        immutable(base/'complete.json',result);return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--action',choices=('prepare','capture'),required=True)
    p.add_argument('--base',default=str(TARGET));p.add_argument('--raw-complete');p.add_argument('--controlled-plan',default=str(A/'matched-loss128-v2/plan.json'))
    p.add_argument('--code-root');p.add_argument('--source-commit');p.add_argument('--selection');p.add_argument('--selection-sha256');p.add_argument('--output')
    a=p.parse_args()
    if a.action=='prepare':
        if not a.raw_complete or not a.code_root or not a.source_commit:p.error('Actual raw-complete, immutable code-root and published commit required')
        value=prepare(a.base,ref(a.raw_complete),ref(a.controlled_plan),a.code_root,a.source_commit)
    else:
        if not a.selection or not a.selection_sha256 or not a.output:p.error('Actual selection+SHA and output required')
        value=capture(a.selection,a.selection_sha256,a.output)
    print(json.dumps(value,sort_keys=True),flush=True)
if __name__=='__main__':main()
