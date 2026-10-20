"""Accept original real G5 plus three independent real G5 events as G20.

This CPU-only consumer never generates, selects virtual groups, touches actors,
reads final-question files, or repacks features. All twelve extra shards must be
sealed before bulk IO. Original and extra own EM/seed journals are rechecked;
features remain the original independent native G5 packing events.
"""
import argparse
import copy
import fcntl
import gzip
import hashlib
import inspect
import json
import os
from pathlib import Path
import time
from .extra_g5_prepare import ROLE,REVISION,validate_config
from .extra_g5_merge import verify_extra_shard
from reproduction.prefix_selection200.rollout import seed_for
from reproduction.prefix20_online.rollout import strict_label

PARENT='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019'
A=PARENT+'/algorithm-improvement-20261021'
PHASES={o:A+'/training-extraG5-matched6528'+('' if o==5 else '-offset%d'%o)+'-20261021/materialized' for o in (5,10,15)}
TARGET=A+'/training-G20-matched6528-20261021/materialized'
STATUS='ACTUAL_ORIGINAL5_PLUS_THREE_INDEPENDENT_G5_G20_SHA_OWN_EM_SEEDS_VERIFIED'


def read(p):return json.loads(Path(p).read_text())
def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()
def digest(v):return hashlib.sha256(canonical(v)).hexdigest()
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for raw in iter(lambda:f.read(1024*1024),b''):h.update(raw)
    return h.hexdigest()
def ref(p):
    p=Path(p);return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def checked(item):
    p=Path(item['path'])
    if p.stat().st_size!=item['bytes'] or sha(p)!=item['sha256']:raise ValueError('Source SHA/size differs: '+str(p))
    return p
def atomic(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix('.tmp')
    with tmp.open('wb') as f:f.write(canonical(v)+b'\n');f.flush();os.fsync(f.fileno())
    tmp.replace(p)
def immutable(p,v):
    p=Path(p)
    if p.exists():
        if read(p)!=v:raise ValueError('Refusing to overwrite different G20 materialization: '+str(p))
    else:atomic(p,v)

def gate(phases,expected_questions=6528):
    """Only small completion/config files; fail before any bulk read if partial."""
    if set(phases)!={5,10,15}:raise ValueError('Require exact real offsets5/10/15')
    for offset in (5,10,15):
        base=Path(phases[offset])
        for shard in range(4):
            root=base/'sampling'/('shard-%02d'%shard)
            for name in ('complete.json','extra-g5-complete.json','extra-g5-adapter-controller.json'):
                if not (root/name).is_file():raise ValueError('All twelve genuine extra shards must already be sealed: '+str(root/name))
    configs={};identity=[];reference=None
    shared=('original_base','original_run_id','original_sampling_attempt_id','original_bank','original_bank_complete',
            'original_config','original_actor_complete','model_dir','model_revision','actor_version','seed',
            'original_protocol_contract','matched_cohort_plan','source_commit','code_root','code_files_sha256')
    for offset in (5,10,15):
        base=Path(phases[offset]);c=validate_config(read(base/'config.json'));configs[offset]=c
        if (c['base']!=str(base) or c['extra_response_offset']!=offset or c['extra_response_indices']!=list(range(offset,offset+5)) or
                c['train_questions']!=expected_questions or c['train_trajectories']!=5*expected_questions):raise ValueError('Wrong real phase identity/counts')
        own={k:c.get(k) for k in shared}
        if reference is None:reference=own
        elif reference!=own:raise ValueError('Three real events must share frozen actor/protocol/source/seed/cohort')
        identity.append((c['run_id'],c['sampling_attempt_id']))
        for shard in range(4):
            root=base/'sampling'/('shard-%02d'%shard);generic=read(root/'complete.json');extra=read(root/'extra-g5-complete.json');controller=read(root/'extra-g5-adapter-controller.json')
            assigned=(expected_questions+3-shard)//4
            if (generic.get('shard')!=shard or generic.get('questions')!=assigned or generic.get('trajectories')!=5*assigned or
                extra.get('questions')!=assigned or extra.get('trajectories')!=5*assigned or len(extra.get('groups',[]))!=assigned or
                extra.get('model_revision')!=REVISION or controller.get('response_indices')!=list(range(offset,offset+5)) or generic['status']!='ALL_ASSIGNED_CANONICAL_G5_COMPLETE' or generic['run_id']!=c['run_id'] or generic['model_revision']!=REVISION or
                extra['status']!='EXTRA_REAL_G5_SHARD_NATIVE_FULL_OWN_EM_SHA_VERIFIED' or extra['run_id']!=c['run_id'] or extra['sampling_attempt_id']!=c['sampling_attempt_id'] or
                extra['shard']!=shard or extra['actor_updates']!=0 or extra['synthetic_EM'] is not False or
                controller['run_id']!=c['run_id'] or controller['sampling_attempt_id']!=c['sampling_attempt_id'] or
                controller['source_commit']!=c['source_commit'] or controller['original_actor_fresh_all_file_SHA'] is not True):raise ValueError('Wrong/missing actual sealed extra event evidence')
    if (len({x[0] for x in identity})!=3 or len({x[1] for x in identity})!=3 or
        reference['original_run_id'] in {x[0] for x in identity} or reference['original_sampling_attempt_id'] in {x[1] for x in identity}):raise ValueError('Original/extra run or attempt namespace collides')
    return configs


def assemble(original,events,qids):
    """Join authentic own labels/IDs only; no label-dependent G5 selection."""
    if set(events)!={5,10,15} or len(qids)!=len(set(qids)):raise ValueError('Exact ordered cohort and three independent events required')
    if not set(qids)<=set(original) or any(set(event)!=set(qids) for event in events.values()):raise ValueError('Missing/foreign authentic event question')
    result=[];seen=set()
    for qid in qids:
        old=original[qid];parts=[old]+[events[o][qid] for o in (5,10,15)]
        for i,g in enumerate(parts):
            if (g['qid']!=qid or g['question_hash']!=old['question_hash'] or len(g['trajectory_ids'])!=5 or len(set(g['trajectory_ids']))!=5 or
                    len(g['labels'])!=5 or any(type(y) is not int or y not in (0,1) for y in g['labels'])):raise ValueError('Incomplete authentic G5 identities/own labels')
            if i and g['response_indices']!=list(range(5*i,5*i+5)):raise ValueError('Real event response slots differ')
        tids=[t for g in parts for t in g['trajectory_ids']];labels=[y for g in parts for y in g['labels']]
        if len(set(tids))!=20 or seen.intersection(tids):raise ValueError('G20 must have twenty globally unique genuine trajectory IDs per question')
        seen.update(tids)
        result.append(dict(qid=qid,question_hash=old['question_hash'],trajectory_ids=tids,labels=labels,
            original_fixed5_trajectory_ids=list(old['trajectory_ids']),original_fixed5_labels=list(old['labels']),
            real_G5_events=[dict(response_indices=list(range(5*i,5*i+5)),trajectory_ids=list(g['trajectory_ids']),labels=list(g['labels'])) for i,g in enumerate(parts)]))
    return result


def audit_rows(rows,event_groups,config,prompts,ledger,*,event_offset,score,seed_function=seed_for):
    """Stream real full journals; shared ledger rejects every same-qid collision."""
    expected={tid:(qid,label,event_offset+j) for qid,g in event_groups.items() for j,(tid,label) in enumerate(zip(g['trajectory_ids'],g['labels']))}
    seen=set();chunks=0;seeds_by_qid={qid:set() for qid in event_groups}
    for row in rows:
        tid=row.get('trajectory_id')
        if tid not in expected:continue
        qid,label,response_index=expected[tid]
        if tid in seen:raise ValueError('Repeated authentic full trajectory')
        seen.add(tid);group=event_groups[qid];source=prompts[qid]
        if (row['qid']!=qid or row['split']!='train' or row['question_hash']!=group['question_hash'] or
                row['response_index']!=response_index or row['run_id']!=config['run_id'] or
                row['policy_snapshot_id']!='official7b-'+REVISION or row['initial_prompt_token_ids']!=source['initial_prompt_token_ids'] or
                row['reward_model']!=source['reward_model'] or type(row['strict_em']) is not int or row['strict_em']!=label or
                len(row['response_token_ids'])!=len(row['response_info_mask']) or not row['behavior_turns'] or not row['sampling_journal']):raise ValueError('Full trajectory own-EM/actor/prompt/identity/provenance differs')
        if score(row)!=label:raise ValueError('Strict EM recomputation differs from authentic own label')
        own=ledger.setdefault(qid,{})
        for chunk in row['sampling_journal']:
            if (chunk.get('sampling_attempt_id')!=config.get('sampling_attempt_id') or
                    chunk.get('actor_version')!='official7b-'+REVISION or chunk.get('policy_snapshot_id')!='official7b-'+REVISION):raise ValueError('Actual journal actor/attempt identity differs')
            seed=seed_function(config['run_id'],1,tid,chunk['logical_turn_index'],chunk['native_chunk_index'],config['seed'],sampling_attempt_id=config.get('sampling_attempt_id'))
            if chunk['seed']!=seed or chunk['sampling']['seed']!=seed:raise ValueError('Actual registered native seed differs from event derivation')
            if seed in own:raise ValueError('Actual same-question seed overlap within/between original and three extra real events')
            own[seed]=dict(event_offset=event_offset,trajectory_id=tid);seeds_by_qid[qid].add(seed);chunks+=1
    if seen!=set(expected):raise ValueError('Incomplete authentic full trajectory coverage')
    return dict(trajectories=len(seen),seed_chunks=chunks,seeds_by_qid=seeds_by_qid)


def full_rows(refs):
    """Refs were SHA checked by the producer audit immediately before iteration."""
    for item in refs:
        with gzip.open(item['path'],'rt') as stream:
            for line in stream:yield json.loads(line)


def feature_membership(manifests,wanted,*,fresh_all_payloads):
    found=set();proofs=[]
    for path in manifests:
        p=Path(path);m=read(p)
        if (m['status']!='LOSSLESS_NATIVE_FEATURES_COMPLETE' or m['feature_layer']!=26 or m['prefix_tokens']!=20 or
            m['dimension']!=3584 or m['question_max_tokens']!=273):raise ValueError('Native G5 feature geometry/source differs')
        index=checked(m['files']['index.jsonl'])
        if index.resolve()!=(p.parent/'index.jsonl').resolve():raise ValueError('Native feature index escapes its genuine packing event')
        # Extra payloads were already fully hashed by verify_extra_shard. Old
        # source retains its accepted packing; every selected row is checked by
        # the native FeatureReader when consumed, never fabricated/repacked here.
        if fresh_all_payloads:
            for item in m['files'].values():checked(item)
        with index.open() as f:
            for line in f:
                row=json.loads(line);tid=row['trajectory_id']
                if tid in wanted:
                    if tid in found:raise ValueError('A real trajectory occurs in multiple native feature events')
                    found.add(tid)
        proofs.append(ref(p))
    if found!=set(wanted):raise ValueError('Missing authentic native feature rows for accepted real trajectories')
    return proofs


def merge(base,phases=PHASES,expected_questions=6528):
    base=Path(base);configs=gate(phases,expected_questions)  # completion gate precedes bulk IO/output creation
    if (not str(base).startswith('/data/user/') or base.resolve()!=base or
        any(base==Path(v) or str(base).startswith(str(Path(v))+'/') for v in phases.values())):raise ValueError('Separate remote G20 output required')
    base.mkdir(parents=True,exist_ok=True);lock=(base/'g20-merge.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (base/'bank/complete.json').exists():raise ValueError('G20 already accepted; use the sealed consumer, never overwrite it')
    started=time.monotonic();c=configs[5]
    try:
        atomic(base/'merge-progress.json',dict(stage='ALL_TWELVE_REAL_EXTRA_SHARDS_SEALED_AUDITING',epoch=time.time()))
        for conf in configs.values():
            for name,value in conf['code_files_sha256'].items():
                if sha(Path(conf['code_root'])/name)!=value:raise ValueError('Frozen qualified collector code differs')
        used=inspect.getfile(verify_extra_shard)
        if sha(used)!=c['code_files_sha256']['reproduction/operations/qwen7b_warmup/extra_g5_merge.py']:raise ValueError('Reused extra verifier differs from qualified immutable collector')
        plans={o:read(Path(phases[o])/'inputs/questions.json') for o in (5,10,15)}
        if len({sha(Path(phases[o])/'inputs/questions.json') for o in (5,10,15)})!=1:raise ValueError('Real events must reuse exactly the same original question/prompt/seed plan')
        plan=plans[5];qids=[g['qid'] for g in plan['groups']]
        if len(qids)!=expected_questions or digest(qids)!=plan['ordered_qid_sha256'] or not plan['validation_qids_excluded'] or not plan['final_qids_excluded']:raise ValueError('Wrong training-only sealed cohort')
        original_bank=read(checked(c['original_bank']));original_config=read(checked(c['original_config']));original_complete=read(checked(c['original_bank_complete']))
        if original_complete['model_revision']!=REVISION or original_complete['actor_updates']!=0 or original_config['run_id']!=c['original_run_id']:raise ValueError('Original G5 is not the frozen official actor')
        wanted_qids=set(qids);original={g['qid']:g for g in original_bank['train'] if g['qid'] in wanted_qids}
        if [original[q] for q in qids]!=plan['groups']:raise ValueError('Original own G5 bank differs from sealed selected labels/IDs')
        prompts={}
        for offset in (5,10,15):
            manifest=read(Path(phases[offset])/'inputs/manifest.json')
            if manifest['run_id']!=configs[offset]['run_id'] or manifest['ordered_qid_sha256']!=plan['ordered_qid_sha256']:raise ValueError('Phase input manifest identity differs')
            own={}
            for item in manifest['batches']:
                v=read(checked(item))
                for row in v['rows']:
                    if row['qid'] in own:raise ValueError('Repeated phase input question')
                    own[row['qid']]=row
            if set(own)!=set(qids):raise ValueError('Real phase input question coverage differs')
            if prompts and prompts!=own:raise ValueError('Real events do not share the exact original token prompts/gold/source rows')
            prompts=own
        # No final_plan path is opened or hashed by this merger.
        events={};extra_refs=[];extra_generic_refs=[];extra_full={};extra_feature_paths={}
        for offset in (5,10,15):
            conf=configs[offset];groups={};sources=[];full=[]
            for shard in range(4):
                result=verify_extra_shard(phases[offset],shard)
                receipt_path=Path(phases[offset])/'sampling'/('shard-%02d'%shard)/'extra-g5-complete.json';published=read(receipt_path)
                if published['groups']!=result['groups'] or published['batches']!=result['batches'] or published['run_id']!=result['run_id']:raise ValueError('Published/fresh genuine extra group evidence differs')
                extra_refs.append(ref(receipt_path));extra_generic_refs.append(ref(Path(phases[offset])/'sampling'/('shard-%02d'%shard)/'complete.json'))
                for g in result['groups']:
                    if g['qid'] in groups:raise ValueError('Repeated real extra question across shards')
                    groups[g['qid']]=g
                sources.extend(str(checked(item)) for item in result['independent_G5_feature_sources'])
                for item in result['batches']:full.append(read(checked(item))['full']['full'])
            events[offset]=groups;extra_full[offset]=full;extra_feature_paths[offset]=sources
            atomic(base/'merge-progress.json',dict(stage='EXTRA_EVENT_PAYLOADS_SHA_OWN_EM_SOURCE_VERIFIED',offset=offset,epoch=time.time()))
        joined=assemble(original,events,qids)
        actor=read(checked(c['original_actor_complete']))
        if actor['revision']!=REVISION or actor['status']!='MIGRATION_ALL_SELECTED_ASSETS_VERIFIED':raise ValueError('Frozen official actor source receipt differs')
        for name,item in actor['files'].items():
            if name.endswith(('.json','.txt','.model')):
                if Path(name).name!=name:raise ValueError('Official tokenizer metadata path escapes snapshot')
                checked(dict(item,path=str(Path(c['model_dir'])/name)))
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(c['model_dir'],local_files_only=True,trust_remote_code=False)
        score=lambda r:strict_label(tokenizer,r['response_token_ids'],r['response_info_mask'],r['reward_model']['ground_truth'])
        # Fresh hash/scan original selected source receipts/full journals. This
        # is intentional ONCE at final G20 acceptance, not repeated at fork.
        original_full=[];original_input_shas={item['sha256'] for item in plan['original_input_batches']}
        for item in plan['original_batches']:
            p=checked(item)
            if str(p).endswith('.gz'):original_full.append(item)
            else:
                v=read(p)
                if v['run_id']!=original_config['run_id'] or v['actor_optimizer_updates']!=0 or v['input_sha256'] not in original_input_shas:raise ValueError('Original source receipt actor/run differs')
                for phase in (v['prefix'],v['full']):
                    if {k:phase[k] for k in c['original_protocol_contract']}!=c['original_protocol_contract']:raise ValueError('Original source protocol differs')
        for item in plan['original_input_batches']:checked(item)
        ledger={};audits={}
        audits[0]=audit_rows(full_rows(original_full),original,original_config,prompts,ledger,event_offset=0,score=score)
        for qid,seeds in audits[0]['seeds_by_qid'].items():
            if sorted(seeds)!=plan['original_native_seeds_by_qid'][qid]:raise ValueError('Fresh original actual seed journals differ from inherited preparation')
        for offset in (5,10,15):audits[offset]=audit_rows(full_rows(extra_full[offset]),events[offset],configs[offset],prompts,ledger,event_offset=offset,score=score)
        original_manifest=Path(c['original_base'])/'bank/features.json';original_paths=read(original_manifest)
        if not isinstance(original_paths,list) or any(not isinstance(p,str) for p in original_paths):raise ValueError('Original genuine native event list required')
        old_tids={t for g in original.values() for t in g['trajectory_ids']}
        feature_events=[dict(event_offset=0,run_id=original_config['run_id'],sampling_attempt_id=original_config.get('sampling_attempt_id'),manifest=ref(original_manifest),native_G5_manifests=feature_membership(original_paths,old_tids,fresh_all_payloads=False),all_bulk_feature_SHA_fresh=False)]
        all_paths=list(original_paths)
        for offset in (5,10,15):
            tids={t for g in events[offset].values() for t in g['trajectory_ids']}
            feature_events.append(dict(event_offset=offset,run_id=configs[offset]['run_id'],sampling_attempt_id=configs[offset]['sampling_attempt_id'],native_G5_manifests=feature_membership(extra_feature_paths[offset],tids,fresh_all_payloads=False),all_bulk_feature_SHA_fresh=True))
            all_paths.extend(extra_feature_paths[offset])
        if len(set(all_paths))!=len(all_paths):raise ValueError('Independent native G5 feature event paths collide')
        immutable(base/'bank/groups.json',dict(train=joined));immutable(base/'bank/feature-events.json',dict(events=feature_events,repacked=False,all_source_events_remain_independent=True))
        immutable(base/'bank/features.json',all_paths)
        hist=[0]*21
        for g in joined:hist[sum(g['labels'])]+=1
        result=dict(status=STATUS,role='train',questions=len(qids),trajectories=20*len(qids),real_trajectories_per_question=20,extra_real_trajectories=15*len(qids),
            groups=ref(base/'bank/groups.json'),features=ref(base/'bank/features.json'),feature_events=ref(base/'bank/feature-events.json'),features_format='flat_list_of_independent_native_G5_complete_paths',
            original_G5_bank=c['original_bank'],original_feature_manifest=ref(original_manifest),extra_shards=extra_refs,extra_generic_shards=extra_generic_refs,
            source_configs={str(o):ref(Path(phases[o])/'config.json') for o in (5,10,15)},source_question_plan=ref(Path(phases[5])/'inputs/questions.json'),
            original_config=c['original_config'],ordered_qid_sha256=digest(qids),model_revision=REVISION,actor_updates=0,synthetic_EM=False,
            original_fixed5_preserved=True,validation_or_final_questions_generated=False,final_pool_read=False,groups_chosen_using_EM=False,
            original_source_full_SHA_fresh=True,original_native_feature_indexes_SHA_fresh=True,original_all_bulk_feature_SHA_fresh=False,
            extra_all_bulk_native_feature_SHA_fresh=True,feature_repacking=False,feature_row_SHA_checked_by_consumer_on_fetch=True,
            strict_EM_recomputed=20*len(qids),strict_EM_mismatches=0,actual_seed_chunks=sum(v['seed_chunks'] for v in audits.values()),actual_same_question_cross_phase_seed_collisions=0,
            native_seed_counts_per_event={str(k):v['seed_chunks'] for k,v in audits.items()},seed_namespace_digest=digest({q:sorted(ledger[q]) for q in qids}),
            histogram_K0_to20=hist,heterogeneous_questions=sum(hist[1:20]),independent_real_G5_events=4,model_tensor_loaded=False,
            source_commit=c['source_commit'],merge_implementation=ref(__file__),strict_scorer_implementation=ref(inspect.getfile(strict_label)),elapsed_seconds=time.monotonic()-started,time=time.time())
        immutable(base/'bank/complete.json',result);atomic(base/'merge-progress.json',dict(stage=STATUS,questions=len(qids),trajectories=20*len(qids),epoch=time.time()))
        return result
    except BaseException as exc:
        atomic(base/'merge-failure.json',dict(error_type=type(exc).__name__,error=str(exc)[:700],epoch=time.time()));raise


def accepted_g20(reference):
    """Lightweight training consumer: SHA/rejoin identities; never select a G5."""
    done=read(checked(reference))
    if (done['status']!=STATUS or done['role']!='train' or done['actor_updates']!=0 or done['synthetic_EM'] or
            done['final_pool_read'] or not done['original_fixed5_preserved'] or done['feature_repacking'] or
            done['actual_same_question_cross_phase_seed_collisions']!=0):raise ValueError('Accepted actual training-only G20 required')
    groups=read(checked(done['groups']))['train'];original={g['qid']:g for g in read(checked(done['original_G5_bank']))['train']};events={o:{} for o in (5,10,15)}
    configs={int(o):read(checked(item)) for o,item in done['source_configs'].items()}
    for item in done['extra_shards']:
        row=read(checked(item));offset=next((o for o,c in configs.items() if c['run_id']==row['run_id']),None)
        if offset is None or row['sampling_attempt_id']!=configs[offset]['sampling_attempt_id'] or row['synthetic_EM'] or row['actor_updates']!=0:raise ValueError('Extra event source identity differs')
        for g in row['groups']:
            if g['qid'] in events[offset]:raise ValueError('Repeated extra event question')
            events[offset][g['qid']]=g
    qids=[g['qid'] for g in groups]
    if (assemble(original,events,qids)!=groups or done['questions']!=len(groups) or done['trajectories']!=20*len(groups) or digest(qids)!=done['ordered_qid_sha256']):raise ValueError('Accepted G20 own labels/IDs/counts differ')
    feature_events=read(checked(done['feature_events']))
    if feature_events['repacked'] or len(feature_events['events'])!=4:raise ValueError('Four independent authentic G5 feature events required')
    paths=read(checked(done['features']))
    if paths!=[p['path'] for event in feature_events['events'] for p in event['native_G5_manifests']]:raise ValueError('Flat features must preserve event order and exact native source paths')
    for event in feature_events['events']:
        for item in event['native_G5_manifests']:checked(item)
    return groups,paths,done


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',default=TARGET);a=p.parse_args()
    if a.base!=TARGET:p.error('Only independent declared G20 target allowed')
    print(json.dumps(merge(a.base)),flush=True)
if __name__=='__main__':main()
