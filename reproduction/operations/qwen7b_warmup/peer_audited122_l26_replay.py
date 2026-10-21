"""Independent accepted122-batch real G10 lossless layer26-only replay.

No generated responses, filtering by labels, actor/probe update, or change to
existing30016 training. Qualified core computes20/23/26; persists only26.
"""
import argparse,fcntl,gzip,json,os,shutil,time
from pathlib import Path
from . import replay_probe_layers as r
from . import peer_newseed_replay as old
from . import peer_newseed_ingest as raw
from .validation_fixedg5_replay import batching_binding
from .replay_bank_prepare import q1_model_binding
from .replay_bank_capture import verify_completed_chunk
TARGET=old.A/'peer-audited122-L26-replay-v1'
INGEST=old.A/'peer-newseed-ingest-audited122-delta-20261021-673b8cb6'
QUESTIONS=15616
SHARDS=4
STATUS='AUDITED122_REAL_G10_QUALIFIED_HF_LAYER26_SHARD_COMPLETE'
RAW_SHA='13f7e6a24699dd9858387210552a4d922ab521bd9a9c7c99648dab147f36048e'
GROUP_SHA='c289201fd0b5c9c22289817bef94e56baa60b35b806c5aa7872b1624ea21f381'
EVENT_SHA='d6d2de8c275c1a444071066edc993c90fdefae2afd2eeb8bf98819d6124358ac'
AUDIT_SHA='74490ebef994ed17723c87725cc515127922c34588d22d37aacb6cb1dcb878d1'


def split_identity(groups,expected=QUESTIONS,shards=SHARDS):
    if len(groups)!=expected or expected%shards:raise ValueError('Exact complete accepted delta identity budget required')
    rows=[];qids=set();hashes=set();tids=set()
    for g in groups:
        q,h,own=g['qid'],g['question_hash'],list(g['trajectory_ids'])
        if json.loads(q)[0] not in ('nq','hotpotqa'):raise ValueError('Accepted delta NQ/Hotpot domains only')
        if q in qids or h in hashes or len(h)!=64 or len(own)!=10 or len(set(own))!=10 or tids.intersection(own):raise ValueError('Unique accepted qid/hash/authentic ten IDs required')
        qids.add(q);hashes.add(h);tids.update(own);rows.append(dict(qid=q,question_hash=h,trajectory_ids=own))
    return rows,[rows[i::shards] for i in range(shards)]


def geometry(row):
    """Measure actual sealed geometry without truncation or token mutation."""
    prompt=len(row['initial_prompt_token_ids']);policy=len(row['policy_token_ids'])
    contexts=[len(v['context_tokens']) for v in row['prefix_segments']] or [prompt]
    return dict(prompt=prompt,policy=policy,context=max(contexts),overlength=prompt>r.QMAX or max(contexts)>4096)


def prepare(base,code_root,commit,golden,cohort=None):
    # Default is the already qualified122 cohort. New immutable deployments may
    # supply only SHA-bound cohort metadata; causal replay/writer is unchanged.
    c=cohort or dict(target=str(TARGET),ingest=str(INGEST),questions=QUESTIONS,raw_sha=RAW_SHA,audit_sha=AUDIT_SHA,group_sha=GROUP_SHA,event_sha=EVENT_SHA,domains={'nq':12300,'hotpotqa':3316},tag='AUDITED122',audit_status='AUDITED122_DELTA_RECEIVER_AND_ELIGIBLE_RAW_G10_INDEPENDENTLY_ACCEPTED',old_controlled_overlap=584,selection_description='all accepted explicit122 delta; original order; no labels/scores')
    expected=c['questions'];per_shard=expected//SHARDS
    if expected%SHARDS or expected<4:raise ValueError('Exactly divisible explicit accepted cohort required')
    base=r.remote(base);code_root=r.remote(code_root);target=r.remote(c['target']);ingest=r.remote(c['ingest'])
    if base!=target or code_root.parent!=base or len(commit)!=40:raise ValueError('Independent published audited122 namespace required')
    source=r.fingerprints(ingest/'complete.json');audit=r.fingerprints(ingest/'independent-final-audit.json')
    if source['sha256']!=c['raw_sha'] or audit['sha256']!=c['audit_sha']:raise ValueError('Accepted raw/join audit source differs')
    accepted=r.read(source['path']);a=r.read(audit['path'])
    if accepted['status']!=raw.STATUS or accepted['strict_EM_mismatches']!=0 or accepted['actor_updates']!=0 or a['status']!=c['audit_status']:raise ValueError('Actual complete ownEM/G10 accepted sources required')
    gref=accepted['outputs']['groups'];eref=accepted['outputs']['source-events']
    if gref['sha256']!=c['group_sha'] or eref['sha256']!=c['event_sha']:raise ValueError('Frozen accepted groups/source-events differ')
    groups,parts=split_identity(r.read(r.checked_file(gref)['path'])['train'],expected);domains={}
    for g in groups:
        k=json.loads(g['qid'])[0];domains[k]=domains.get(k,0)+1
    if domains!=c['domains']:raise ValueError('Actual accepted domain counts differ')
    events=r.read(r.checked_file(eref)['path']);sources=[];seenpaths=set()
    for event in events['events']:
        for batch in event['batches']:
            item=batch['prefix']
            if item['path'] not in seenpaths:seenpaths.add(item['path']);sources.append(item)
    batching=batching_binding(golden,32);q1,binding=q1_model_binding()
    base.mkdir(parents=True,exist_ok=True)
    with (base/'prepare.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (base/'preparation.json').exists():
            done=r.read(base/'preparation.json')
            if done['raw_complete']!=source or done['source_commit']!=commit:raise ValueError('Existing preparation source differs')
            for item in done['selections']:r.checked_file(item)
            return done
        free=shutil.disk_usage(base).free
        if free<650*1024**3:raise ValueError('Require450GiB feature margin plus200GiB current writer reserve')
        identity=dict(role='train',groups=groups,source_ingest=source,source_groups=gref,summary=dict(questions=expected,trajectories=expected*10,labels_consulted=False,scores_consulted=False,question_selection=c['selection_description'],domains=domains,existing30016_qid_and_hash_overlap=0,old_controlled6528_training_question_overlap=c.get('old_controlled_overlap'),unseen_in_all_previous_training_claim=False,final_labels_or_rollouts_read=False))
        raw.immutable(base/'identity-plan.json',identity);iref=r.fingerprints(base/'identity-plan.json')
        wanted={t:g['qid'] for g in groups for t in g['trajectory_ids']};owners={t:i for i,part in enumerate(parts) for g in part for t in g['trajectory_ids']}
        streams=[];paths=[];counts=[0]*4;seen=set();bad=[0]*4;stats={}
        try:
            for i in range(4):
                p=base/('shard-%02d'%i);(p/'inputs').mkdir(parents=True,exist_ok=True);(p/'logs').mkdir(exist_ok=True)
                path=p/'inputs/prefix.jsonl.gz';temp=path.with_suffix('.gz.partial')
                if path.exists() or temp.exists():raise ValueError('Unknown prior prefix output preserved; no overwrite')
                paths.append((temp,path));streams.append(gzip.open(temp,'xb'))
            for item in sources:
                path=r.checked_file(item)['path']
                with gzip.open(path,'rt') as inp:
                    for line in inp:
                        row=json.loads(line);tid=row['trajectory_id']
                        if tid not in wanted:continue
                        qid=wanted[tid];i=owners[tid]
                        if tid in seen or row['qid']!=qid:raise ValueError('Repeated/wrong accepted own prefix identity')
                        if any(k in row for k in ('strict_em','labels','reward_model','ground_truth','response_token_ids')):raise ValueError('Only actual causal Prefix20 records allowed')
                        v=geometry(row);domain=json.loads(qid)[0];st=stats.setdefault(domain,dict(trajectories=0,min_prompt=10**9,max_prompt=0,min_context=10**9,max_context=0,short_policy=0,overlength_trajectories=0))
                        st['trajectories']+=1;st['min_prompt']=min(st['min_prompt'],v['prompt']);st['max_prompt']=max(st['max_prompt'],v['prompt']);st['min_context']=min(st['min_context'],v['context']);st['max_context']=max(st['max_context'],v['context']);st['short_policy']+=int(v['policy']<20)
                        if v['overlength']:
                            bad[i]+=1;st['overlength_trajectories']+=1
                        else:
                            r.normalize_prefix(row,tid,qid,True);streams[i].write(r.canonical(row)+b'\n');counts[i]+=1
                        seen.add(tid)
                        if len(seen)%512==0:raw.atomic(base/'prepare-progress.json',dict(status='ACTUAL_'+c['tag']+'_PREFIX20_GEOMETRY_AND_PROJECTION_ADVANCING',trajectories=len(seen),expected_trajectories=expected*10,qualified_counts=counts,overlength_counts=bad,geometry=stats,actor_updates=0,time=time.time()))
        finally:
            for stream in streams:stream.close()
        if seen!=set(wanted):raise ValueError('Missing actual accepted own prefix IDs')
        selections=[];blocked=[]
        for i,(temp,path) in enumerate(paths):
            os.replace(str(temp),str(path));part=parts[i]
            if bad[i]:blocked.append(dict(shard=i,overlength_trajectories=bad[i],qualified_trajectories=counts[i]));continue
            if counts[i]!=per_shard*10:raise ValueError('Exact accepted G10 per-shard coverage required')
            scope={}
            for g in part:
                k=json.loads(g['qid'])[0];scope[k]=scope.get(k,0)+1
            selection=dict(role='peer_newseed_train_G10',model_revision=r.REVISION,questions=len(part),trajectories=[dict(qid=g['qid'],trajectory_id=t) for g in part for t in g['trajectory_ids']],real_trajectories_per_question=10,label_blind_selection=True,prefix_tokens=20,capture_layers=list(r.LAYERS),persist_layers=[26],prefix_files=[r.fingerprints(path)],model_files=q1['model_files'],source_model_binding=binding,source_ingest=source,question_plan=iref,domain_scope=dict(kind='accepted_delta_NQ_and_Hotpot',question_counts=scope,overall_question_counts=domains,old_controlled6528_training_overlap=c.get('old_controlled_overlap'),unseen_previous_training_claim=False),source_commit=commit,expected_code_root=str(code_root),shard=i,shards=4,batching=batching,geometry_checked_no_truncation=True,existing30016_changed=False,new_responses_generated=0,actor_updates=0)
            p=base/('shard-%02d'%i)/'inputs/selection.json';raw.immutable(p,selection);selections.append(r.fingerprints(p))
        done=dict(status=c['tag']+'_G10_PREFIX20_GEOMETRY_FOUR_SHARDS_PREPARED' if not blocked else c['tag']+'_G10_GEOMETRY_BLOCKED_SHARDS_PRESERVED',identity_plan=iref,raw_complete=source,raw_groups=gref,source_events=eref,independent_join_audit=audit,source_commit=commit,code_root=str(code_root),batching=batching,selections=selections,blocked_shards=blocked,source_prefix_files=sources,questions=expected,trajectories=expected*10,domains=domains,geometry=stats,compute_layers=list(r.LAYERS),persist_layers=[26],writer_layout='lossless_question_base_xor_policy_v1',free_space_bytes=free,minimum_free_bytes=650*1024**3,labels_used_for_selection=False,full_or_own_EM_files_opened=False,actor_updates=0,probe_updates=0,new_rollouts_generated=0,existing30016_changed=False,final_pool_read=False,time=time.time())
        raw.immutable(base/'preparation.json',done);return done


def capture(selection_path,selection_sha,output,expected_questions=3904,complete_status=STATUS):
    import torch
    from transformers import AutoModel
    from .replay_batched_context import BatchedContextCapture
    if not torch.cuda.is_available() or not os.environ.get('SLURM_JOB_ID') or torch.cuda.device_count()!=1:
        raise ValueError('Owned real CUDA Slurm allocation required')
    source=r.fingerprints(r.remote(selection_path));selection=r.read(source['path']);base=r.remote(output)
    if source['sha256']!=selection_sha or selection['expected_code_root']!=str(Path(__file__).resolve().parents[3]) or selection['persist_layers']!=[26] or selection['questions']!=expected_questions:
        raise ValueError('Exact isolated cohort G10/layer26 selection required')
    if r.fingerprints(r.__file__)['sha256']!=selection['source_model_binding']['legacy_replay_module_sha256']:
        raise ValueError('Qualified causal primitive differs')
    batching=batching_binding(selection['batching']['golden'],32)
    if batching!=selection['batching']:raise ValueError('Qualified batch32 source differs')
    base.mkdir(parents=True,exist_ok=True)
    with (base/'capture.lock').open('a') as guard:
        fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (base/'complete.json').exists():
            done=r.read(base/'complete.json')
            if done['source_selection']!=source or done['status']!=complete_status:raise ValueError('Existing completion source differs')
            for item in done['chunks']:
                chunk=r.read(r.checked_file(item)['path']);verify_completed_chunk(Path(item['path']).parent,chunk['trajectory_ids'])
            return done
        selected=old.load_selection(selection);model_dir=selection['source_model_binding']['official_view'];proofs,config=r.verify_model(model_dir,selection)
        if any(t>=config['vocab_size'] for row in selected for seg in row['segments'] for t in seg['context_tokens']):raise ValueError('Actual token outside official vocabulary')
        device=torch.device('cuda:0');torch.cuda.set_device(device)
        if 'H100' not in torch.cuda.get_device_name(device):raise ValueError('One explicitly isolated H100 per srun task required')
        raw.atomic(base/'device.json',dict(job_id=os.environ['SLURM_JOB_ID'],local_rank=os.environ.get('SLURM_LOCALID'),visible_device_count=torch.cuda.device_count(),device_name=torch.cuda.get_device_name(device),actor_updates=0,time=time.time()))
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
                writer.write(row,values[26]);raw.atomic(base/'progress.json',dict(status='ACTUAL_ADDITIONAL_G10_LAYER26_FEATURES_ADVANCING',trajectories=start+index+1,expected_trajectories=expected_questions*10,completed_chunks=len(chunks),compute_layers=list(r.LAYERS),persist_layers=[26],batching=engine.stats(),elapsed_seconds=time.monotonic()-started,actor_updates=0,time=time.time()))
            manifest=writer.finish();raw.immutable(root/'complete.json',dict(source_selection=source,trajectory_ids=tids,feature_manifests={'26':manifest},batching=batching,actor_updates=0,source_runtime='huggingface_frozen_causal_prefill'));chunks.append(r.fingerprints(root/'complete.json'))
        manifests=[];ids=[]
        for item in chunks:
            done=verify_completed_chunk(Path(item['path']).parent,r.read(item['path'])['trajectory_ids']);ids+=done['trajectory_ids'];manifests.append(done['feature_manifests']['26'])
        if ids!=[row['trajectory_id'] for row in selected]:raise ValueError('Additional shard authentic G10 coverage differs')
        from .replay_feature_reader import ReplayFeatureReader
        reader=ReplayFeatureReader(manifests,26)
        if set(reader.records)!=set(ids):raise ValueError('Accepted reader coverage differs')
        reader.get([ids[0],ids[-1]],'cpu');reader.close();raw.immutable(base/'features-layer26.json',manifests)
        done=dict(status=complete_status,source_selection=source,identity_plan=selection['question_plan'],source_ingest=selection['source_ingest'],
            questions=expected_questions,trajectories=len(ids),feature_manifests_by_capture_layer={'26':manifests},features=r.fingerprints(base/'features-layer26.json'),
            chunks=chunks,compute_layers=list(r.LAYERS),capture_layers=[26],persist_layers=[26],human_block_layers=[27],
            model_revision=r.REVISION,model_files=proofs,source_runtime='huggingface_frozen_causal_prefill',attention_implementation='flash_attention_2',batching=batching,actual_batching=engine.stats(),
            actor_frozen=True,actor_updates=0,probe_updates=0,new_responses_generated=0,existing30016_changed=False,native_bitexact_equivalence_claim=False,
            runtime_code=r.fingerprints(__file__),job_id=os.environ['SLURM_JOB_ID'],elapsed_seconds=time.monotonic()-started,time=time.time())
        raw.immutable(base/'complete.json',done);return done


def worker(preparation_path,preparation_sha,pair):
    if pair not in (0,1) or os.environ.get('SLURM_NTASKS')!='2':raise ValueError('Explicit two-task Slurm pair required')
    local=int(os.environ['SLURM_LOCALID'])
    if local not in (0,1):raise ValueError('One task per pair device required')
    p=r.fingerprints(r.remote(preparation_path))
    if p['sha256']!=preparation_sha:raise ValueError('Actual preparation SHA differs')
    done=r.read(p['path']);shard=pair*2+local;lookup={r.read(v['path'])['shard']:v for v in done['selections']}
    if shard not in lookup:raise ValueError('Shard has blocked original geometry; no silent truncation')
    own=lookup[shard];return capture(own['path'],own['sha256'],TARGET/('shard-%02d'%shard)/'capture')


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--action',choices=('prepare','capture','worker'),required=True);p.add_argument('--base',default=str(TARGET));p.add_argument('--code-root');p.add_argument('--commit');p.add_argument('--golden');p.add_argument('--selection');p.add_argument('--selection-sha256');p.add_argument('--output');p.add_argument('--preparation');p.add_argument('--preparation-sha256');p.add_argument('--pair',type=int);a=p.parse_args()
    try:
        if a.action=='prepare':v=prepare(a.base,a.code_root,a.commit,r.fingerprints(a.golden))
        elif a.action=='worker':v=worker(a.preparation,a.preparation_sha256,a.pair)
        else:v=capture(a.selection,a.selection_sha256,a.output)
    except BaseException as e:
        out=Path(a.output) if a.output else TARGET/('shard-%02d'%(a.pair*2+int(os.environ.get('SLURM_LOCALID','0'))))/'capture' if a.action=='worker' else TARGET
        out.mkdir(parents=True,exist_ok=True);raw.atomic(out/'failure.json',dict(error_type=type(e).__name__,error=str(e)[:1000],time=time.time()));raise
    print(json.dumps(v,sort_keys=True),flush=True)
if __name__=='__main__':main()
