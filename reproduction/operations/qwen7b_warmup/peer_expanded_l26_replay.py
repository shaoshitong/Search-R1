"""Feature-only expansion: preserve8192 and capture only21824 frozen NQ G10.

The qualified core still computes postblocks20/23/26 in one full causal pass.
Only layer26 is persisted. No rollout, label-based selection, actor or probe
update is performed. Existing8192 packed features remain read-only.
"""
import argparse,fcntl,gzip,json,os,shutil,time
from pathlib import Path
from . import replay_probe_layers as r
from . import peer_newseed_replay as old
from . import peer_newseed_ingest as raw
from .validation_fixedg5_replay import batching_binding
from .replay_bank_prepare import q1_model_binding
from .replay_bank_capture import verify_completed_chunk

TARGET=old.A/'peer-expanded-L26-replay-v1'
IDENTITY=old.A/'peer-expanded-identity-v1/identity-plan.json'
IDENTITY_SHA='b2bcc13b542c3efbd90077a94138c02917940f286fde8f6ffeab2abe7542a047'
PRIOR=old.A/'peer-newseed8192-g10-batched-replay-20261021'
QUESTIONS=30016
NEW_QUESTIONS=21824
SHARDS=4
STATUS='EXPANDED_ADDITIONAL_REAL_G10_QUALIFIED_HF_LAYER26_SHARD_COMPLETE'


def split_new(identity,existing=8192,added=NEW_QUESTIONS,shards=SHARDS):
    """Identity-only projection; never consult labels/probe scores."""
    groups=identity['groups'];new=identity['remaining_groups'];s=identity['summary']
    if (identity.get('role')!='train' or s.get('labels_consulted') is not False or
        s.get('scores_consulted') is not False or s.get('existing8192_preserved_exactly') is not True or
        s['existing_questions']!=existing or s['new_questions']!=added or
        len(groups)!=existing+added or len(new)!=added or groups[existing:]!=new or added%shards):
        raise ValueError('Exact frozen unchanged prior prefix plus declared new identities required')
    qids=set();hashes=set();tids=set()
    for g in groups:
        if set(g)!={'qid','question_hash','trajectory_ids'} or json.loads(g['qid'])[0]!='nq':
            raise ValueError('Frozen identity-only NQ records required')
        if g['qid'] in qids or g['question_hash'] in hashes or len(g['question_hash'])!=64:
            raise ValueError('Repeated question ID/text hash')
        own=g['trajectory_ids']
        if len(own)!=10 or len(set(own))!=10 or tids.intersection(own):
            raise ValueError('Every actual old5+new5 identity must be distinct')
        qids.add(g['qid']);hashes.add(g['question_hash']);tids.update(own)
    parts=[new[i::shards] for i in range(shards)]
    prior={t for g in groups[:existing] for t in g['trajectory_ids']}
    if any(prior.intersection(t for g in part for t in g['trajectory_ids']) for part in parts):
        raise ValueError('Already-qualified prior trajectory must not be recaptured')
    return parts


def prepare(base,code_root,commit,golden):
    base=r.remote(base);code_root=r.remote(code_root)
    if base!=TARGET or code_root.parent!=base or len(commit)!=40:
        raise ValueError('Independent published expansion namespace required')
    identity_ref=r.fingerprints(IDENTITY)
    if identity_ref['sha256']!=IDENTITY_SHA:raise ValueError('Precommitted expansion identity SHA differs')
    plan=r.read(IDENTITY);parts=split_new(plan);prior=r.read(r.checked_file(plan['prior_identity'])['path'])
    if plan['groups'][:8192]!=prior['groups']:raise ValueError('Original8192 changed')
    ingest_ref=plan['raw_complete'];accepted=r.read(r.checked_file(ingest_ref)['path'])
    if accepted['status']!=raw.STATUS or accepted['strict_EM_mismatches']!=0 or accepted['actor_updates']!=0:
        raise ValueError('Actual own strictEM raw acceptance required')
    events=r.read(r.checked_file(accepted['outputs']['source-events'])['path']);sources=[];seen=set()
    for event in events['events']:
        for batch in event['batches']:
            item=batch['prefix']
            if item['path'] not in seen:seen.add(item['path']);sources.append(item)
    batching=batching_binding(golden,32);q1,binding=q1_model_binding()
    prior_completes=[]
    for i in range(4):
        ref=r.fingerprints(PRIOR/('shard-%02d'%i)/'capture/complete.json');done=r.read(ref['path'])
        selection=r.read(r.checked_file(done['source_selection'])['path'])
        if (done['status']!=old.STATUS or done['actor_updates']!=0 or selection['question_plan']!=plan['prior_identity'] or
            selection['source_ingest']!=ingest_ref or done['batching']!=batching):
            raise ValueError('Prior8192 is not actual same-qualified backend')
        prior_completes.append(ref)
    base.mkdir(parents=True,exist_ok=True)
    with (base/'prepare.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (base/'preparation.json').exists():
            done=r.read(base/'preparation.json')
            if done['identity_plan']!=identity_ref or done['source_commit']!=commit:raise ValueError('Existing preparation source differs')
            for item in done['selections']:r.checked_file(item)
            return done
        free=shutil.disk_usage(base).free
        if free<550*1024**3:raise ValueError('Need550GiB shared storage margin for additional layer26')
        wanted={t:g['qid'] for part in parts for g in part for t in g['trajectory_ids']}
        owners={t:i for i,part in enumerate(parts) for g in part for t in g['trajectory_ids']}
        streams=[];paths=[];counts=[0]*SHARDS
        try:
            for i in range(SHARDS):
                root=base/('shard-%02d'%i);(root/'inputs').mkdir(parents=True,exist_ok=True);(root/'logs').mkdir(exist_ok=True)
                path=root/'inputs/prefix.jsonl.gz';temp=path.with_suffix('.gz.partial')
                if path.exists() or temp.exists():raise ValueError('Unknown prefix partial preserved; do not overwrite')
                paths.append((temp,path));streams.append(gzip.open(temp,'xb'))
            for row in old.projected_rows(sources,wanted):
                i=owners[row['trajectory_id']];streams[i].write(r.canonical(row)+b'\n');counts[i]+=1
                if sum(counts)%512==0:raw.atomic(base/'prepare-progress.json',dict(stage='PREFIX_ONLY_ADDITIONAL21824_PROJECTION',trajectories=sum(counts),expected_trajectories=218240,actor_updates=0,time=time.time()))
        finally:
            for stream in streams:stream.close()
        selections=[]
        for i,(temp,path) in enumerate(paths):
            if counts[i]!=54560:raise ValueError('Additional5456 G10 shard coverage differs')
            os.replace(str(temp),str(path));part=parts[i]
            selection=dict(role='peer_newseed_train_G10',model_revision=r.REVISION,questions=len(part),
                trajectories=[dict(qid=g['qid'],trajectory_id=t) for g in part for t in g['trajectory_ids']],
                real_trajectories_per_question=10,label_blind_selection=True,prefix_tokens=20,
                capture_layers=list(r.LAYERS),persist_layers=[26],prefix_files=[r.fingerprints(path)],
                model_files=q1['model_files'],source_model_binding=binding,source_ingest=ingest_ref,
                question_plan=identity_ref,prior_identity_plan=plan['prior_identity'],domain_scope=dict(kind='NQ_only_enrichment',question_counts={'nq':len(part)}),
                source_commit=commit,expected_code_root=str(code_root),shard=i,shards=SHARDS,batching=batching,
                existing8192_reused_not_recaptured=True,new_responses_generated=0,actor_updates=0)
            p=base/('shard-%02d'%i)/'inputs/selection.json';raw.immutable(p,selection);selections.append(r.fingerprints(p))
        done=dict(status='ADDITIONAL21824_G10_PREFIX_ONLY_FOUR_SHARDS_PREPARED',identity_plan=identity_ref,
            raw_complete=ingest_ref,prior_identity_plan=plan['prior_identity'],prior_feature_completes=prior_completes,
            source_commit=commit,code_root=str(code_root),batching=batching,selections=selections,source_prefix_files=sources,
            questions=21824,trajectories=218240,combined_questions=30016,combined_trajectories=300160,
            compute_layers=list(r.LAYERS),persist_layers=[26],free_space_bytes=free,labels_read=False,
            full_or_own_EM_files_opened=False,actor_updates=0,probe_updates=0,new_rollouts_generated=0,final_pool_read=False,time=time.time())
        raw.immutable(base/'preparation.json',done);return done


def capture(selection_path,selection_sha,output):
    import torch
    from transformers import AutoModel
    from .replay_batched_context import BatchedContextCapture
    if not torch.cuda.is_available() or not os.environ.get('SLURM_JOB_ID') or not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise ValueError('Owned real CUDA Slurm allocation required')
    source=r.fingerprints(r.remote(selection_path));selection=r.read(source['path']);base=r.remote(output)
    if source['sha256']!=selection_sha or selection['expected_code_root']!=str(Path(__file__).resolve().parents[3]) or selection['persist_layers']!=[26] or selection['questions']!=5456:
        raise ValueError('Actual isolated5456 G10/layer26 selection required')
    if r.fingerprints(r.__file__)['sha256']!=selection['source_model_binding']['legacy_replay_module_sha256']:
        raise ValueError('Qualified causal primitive differs')
    batching=batching_binding(selection['batching']['golden'],32)
    if batching!=selection['batching']:raise ValueError('Qualified batch32 source differs')
    base.mkdir(parents=True,exist_ok=True)
    with (base/'capture.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (base/'complete.json').exists():raise ValueError('Existing accepted shard preserved')
        selected=old.load_selection(selection);model_dir=selection['source_model_binding']['official_view'];proofs,config=r.verify_model(model_dir,selection)
        if any(t>=config['vocab_size'] for row in selected for seg in row['segments'] for t in seg['context_tokens']):raise ValueError('Actual token outside official vocabulary')
        device=torch.device('cuda:0');torch.cuda.set_device(device)
        model=AutoModel.from_pretrained(model_dir,torch_dtype=torch.bfloat16,attn_implementation='flash_attention_2',local_files_only=True,trust_remote_code=False,low_cpu_mem_usage=True).eval().requires_grad_(False).to(device)
        engine=BatchedContextCapture(model,batch_size=32,max_context_tokens=16384,device=device)
        started=time.monotonic();chunks=[]
        for start in range(0,len(selected),320):
            own=selected[start:start+320];root=base/('chunk-%05d'%(start//320));tids=[row['trajectory_id'] for row in own]
            if (root/'complete.json').exists():
                done=verify_completed_chunk(root,tids)
                if done['source_selection']!=source or done['batching']!=batching:raise ValueError('Prior complete chunk source differs')
                chunks.append(r.fingerprints(root/'complete.json'));continue
            if root.exists():raise ValueError('Unknown partial preserved; isolated recovery required')
            root.mkdir();writer=r.ReplayLayerWriter(root/'layer-26',26,own)
            for index,(row,values) in enumerate(engine.iter_replay(own,window_trajectories=32)):
                writer.write(row,values[26]);raw.atomic(base/'progress.json',dict(status='ACTUAL_ADDITIONAL_G10_LAYER26_FEATURES_ADVANCING',trajectories=start+index+1,expected_trajectories=54560,completed_chunks=len(chunks),compute_layers=list(r.LAYERS),persist_layers=[26],batching=engine.stats(),elapsed_seconds=time.monotonic()-started,actor_updates=0,time=time.time()))
            manifest=writer.finish();raw.immutable(root/'complete.json',dict(source_selection=source,trajectory_ids=tids,feature_manifests={'26':manifest},batching=batching,actor_updates=0,source_runtime='huggingface_frozen_causal_prefill'));chunks.append(r.fingerprints(root/'complete.json'))
        manifests=[];ids=[]
        for item in chunks:
            done=verify_completed_chunk(Path(item['path']).parent,r.read(item['path'])['trajectory_ids']);ids+=done['trajectory_ids'];manifests.append(done['feature_manifests']['26'])
        if ids!=[row['trajectory_id'] for row in selected]:raise ValueError('Additional shard authentic G10 coverage differs')
        from .replay_feature_reader import ReplayFeatureReader
        reader=ReplayFeatureReader(manifests,26)
        if set(reader.records)!=set(ids):raise ValueError('Accepted reader coverage differs')
        reader.get([ids[0],ids[-1]],'cpu');reader.close();raw.immutable(base/'features-layer26.json',manifests)
        done=dict(status=STATUS,source_selection=source,identity_plan=selection['question_plan'],source_ingest=selection['source_ingest'],
            questions=5456,trajectories=len(ids),feature_manifests_by_capture_layer={'26':manifests},features=r.fingerprints(base/'features-layer26.json'),
            chunks=chunks,compute_layers=list(r.LAYERS),capture_layers=[26],persist_layers=[26],human_block_layers=[27],
            model_revision=r.REVISION,model_files=proofs,source_runtime='huggingface_frozen_causal_prefill',attention_implementation='flash_attention_2',batching=batching,actual_batching=engine.stats(),
            actor_frozen=True,actor_updates=0,probe_updates=0,new_responses_generated=0,existing8192_recaptured=False,native_bitexact_equivalence_claim=False,
            runtime_code=r.fingerprints(__file__),job_id=os.environ['SLURM_JOB_ID'],elapsed_seconds=time.monotonic()-started,time=time.time())
        raw.immutable(base/'complete.json',done);return done


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--action',choices=('prepare','capture'),required=True);p.add_argument('--base',default=str(TARGET));p.add_argument('--code-root');p.add_argument('--commit');p.add_argument('--golden');p.add_argument('--selection');p.add_argument('--selection-sha256');p.add_argument('--output');a=p.parse_args()
    try:
        if a.action=='prepare':v=prepare(a.base,a.code_root,a.commit,r.fingerprints(a.golden))
        else:v=capture(a.selection,a.selection_sha256,a.output)
    except BaseException as e:
        if a.action=='capture' and a.output:
            out=Path(a.output);out.mkdir(parents=True,exist_ok=True);raw.atomic(out/'failure.json',dict(error_type=type(e).__name__,error=str(e)[:1000],time=time.time()))
        raise
    print(json.dumps(v,sort_keys=True),flush=True)
if __name__=='__main__':main()
