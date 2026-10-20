"""CPU-only, label-blind preparation of four real matched multilayer banks.

No GPU/model/tensor loading, full response/EM reading, submission or generation.
Inputs are actual matched128 plan/bank identities and sealed native prefix-only
metadata. Outputs must stay in a NEW remote directory. Preparation alone is not
recapture completion or acceptance of HF replay as a native-equivalent backend.
"""
import argparse
from collections import Counter
import fcntl
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import shlex
import time
from .replay_probe_layers import (REVISION,MODEL,LAYERS,canonical,fingerprints,
    remote,read,atomic,normalize_prefix,checked_file)

PARENT='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019'
Q1=PARENT+'/replay-layer-qualification-20261021'
RUNTIME_COMMIT='18c46affb9c6c69330aa19eb1e00124e1f39d019'
MODEL_VIEW=Q1+'/official-snapshot-view'
ROLE_COUNTS={'fit':6528,'diagnostic512':512,'selection2048':2048}


def digest(value):return hashlib.sha256(canonical(value)).hexdigest()


def checked(ref):return Path(checked_file(ref)['path'])


def once(path,value):
    path=Path(path)
    if path.exists():
        if read(path)!=value:raise ValueError('Immutable preparation conflict: '+str(path))
    else:atomic(path,value)
    return fingerprints(path)


def partition_questions(plan,train,diagnostic,selection,expected_counts=None):
    """Choose only the explicit matched cohort; do not consult own EM/scores."""
    counts=ROLE_COUNTS if expected_counts is None else expected_counts
    lookup={g['qid']:g for g in train}
    if len(lookup)!=len(train):raise ValueError('Duplicate source train qid')
    flattened=[q for batch in plan['batches'] for q in batch]
    cohort=plan['cohort_qids']
    if (len(plan['batches'])!=128 or any(len(b)!=51 or len(set(b))!=51 for b in plan['batches']) or
            len(cohort)!=6528 or len(set(cohort))!=6528 or len(flattened)!=6528 or
            len(set(flattened))!=6528 or set(flattened)!=set(cohort)):
        if expected_counts is None:raise ValueError('Require the actual distinct matched128/B51/6528 plan')
        if len(cohort)!=len(set(cohort)) or flattened!=cohort:
            raise ValueError('Synthetic test plan must also contain unique declared cohort')
    pools={'fit':[lookup[q] for q in cohort],'diagnostic512':diagnostic,'selection2048':selection}
    if {k:len(v) for k,v in pools.items()}!=counts:raise ValueError('Exact declared pool counts required; never truncate')
    seen_qids=set();seen_hashes=set();seen_tids=set();shards=[[] for _ in range(4)]
    for role,groups in pools.items():
        for ordinal,g in enumerate(groups):
            qid=g['qid'];tids=g['trajectory_ids'];qhash=g['question_hash']
            if (not isinstance(qid,str) or not qid or qid in seen_qids or qhash in seen_hashes or
                    len(tids)!=5 or len(set(tids))!=5 or seen_tids.intersection(tids)):
                raise ValueError('Pool identity overlap or incomplete own authentic G5')
            seen_qids.add(qid);seen_hashes.add(qhash);seen_tids.update(tids)
            shards[ordinal%4].append(dict(role=role,qid=qid,question_hash=qhash,trajectory_ids=list(tids)))
    return shards,dict(roles=counts,questions=len(seen_qids),trajectories=len(seen_tids),
        ordered_qid_sha256=digest([g['qid'] for part in shards for g in part]),
        qid_set_sha256=digest(sorted(seen_qids)),question_hash_set_sha256=digest(sorted(seen_hashes)),
        trajectory_set_sha256=digest(sorted(seen_tids)),labels_consulted_for_selection=False,
        qid_disjoint=True,question_hash_disjoint=True,trajectory_disjoint=True)


def q1_model_binding(q1=Q1):
    q1=remote(q1);pub=read(q1/'source-publication.json');prepared=read(q1/'preparation.json')
    complete=read(q1/'qualification-output/complete.json')
    if pub['commit']!=RUNTIME_COMMIT or complete['status']!='BOUNDED_HF_REPLAY_CAPTURE_AND_NATIVE_COMPARISON_COMPLETE':
        raise ValueError('Previously completed Q1 official model view required')
    selection=read(checked(prepared['selection']))
    if (complete['source_selection']!=fingerprints(prepared['selection']['path']) or
            complete['actor_optimizer_updates']!=0 or not complete['actor_frozen'] or
            prepared['model_view']!=MODEL_VIEW or
            complete['code']['sha256']!=pub['files']['reproduction/operations/qwen7b_warmup/replay_probe_layers.py']['sha256']):
        raise ValueError('Q1 completed selection/source/model-view binding differs')
    for name,ref in selection['model_files'].items():
        path=Path(MODEL_VIEW)/name
        if Path(name).name!=name or path.stat().st_size!=ref['bytes']:
            raise ValueError('Q1 official model-view file size/name differs')
        actual=complete['model_files'][name]
        if actual['sha256']!=ref['sha256'] or actual['bytes']!=ref['bytes']:
            raise ValueError('Q1 full model SHA acceptance differs')
        if not name.endswith('.safetensors') and fingerprints(path)['sha256']!=ref['sha256']:
            raise ValueError('Q1 small model metadata changed')
    return selection,dict(q1_publication=fingerprints(q1/'source-publication.json'),
        q1_preparation=fingerprints(q1/'preparation.json'),q1_complete=fingerprints(q1/'qualification-output/complete.json'),
        q1_selection=prepared['selection'],official_view=MODEL_VIEW,reference_runtime_commit=RUNTIME_COMMIT,
        legacy_replay_module_sha256=complete['code']['sha256'],
        GPU_rechecks_full_official_model_SHA=True,native_bitexact_equivalence_claim=False)


def render_sbatch(shard_root,code_root,selection_sha,shard,hours=1):
    if type(shard) is not int or not 0<=shard<4 or hours not in (1,2):raise ValueError('Four shards; explicit one or two hour budget')
    for p in (shard_root,code_root):
        if not Path(p).is_absolute() or any(c.isspace() for c in str(p)):raise ValueError('Absolute whitespace-free deployment roots')
    if len(selection_sha)!=64 or any(c not in '0123456789abcdef' for c in selection_sha):raise ValueError('Actual fixture SHA required')
    script='''#!/usr/bin/env bash
#SBATCH --job-name=q7b-replay-bank-@@SHARD@@
#SBATCH --partition=acd_u
#SBATCH --account=mou176
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=0@@HOURS@@:00:00
#SBATCH --output=@@BASE_RAW@@/logs/replay-%j.log
set -euo pipefail
REPLAY_BASE=@@BASE_Q@@
export QWEN7B_CODE_ROOT=@@CODE_Q@@
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONFAULTHANDLER=1
exec 9>"$REPLAY_BASE/replay-run.lock"
flock -n 9
python3 - "$QWEN7B_CODE_ROOT" <<'VERIFY_IMMUTABLE_CODE'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]);m=json.loads((root/'_replay_bank_source_manifest.json').read_text())
for name,expected in m['files'].items():
 p=root/name;assert p.stat().st_size==expected['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==expected['sha256'],name
VERIFY_IMMUTABLE_CODE
srun --kill-on-bad-exit=1 bash "$QWEN7B_CODE_ROOT/reproduction/operations/qwen7b_warmup/runtime.sh" -u -m reproduction.operations.qwen7b_warmup.replay_bank_capture --selection "$REPLAY_BASE/inputs/selection.json" --selection-sha256 @@SHA@@ --output "$REPLAY_BASE/capture" --model @@MODEL_Q@@ --device cuda:0
'''
    for token,value in {'@@SHARD@@':str(shard),'@@HOURS@@':str(hours),'@@BASE_RAW@@':str(shard_root),
        '@@BASE_Q@@':shlex.quote(str(shard_root)),'@@CODE_Q@@':shlex.quote(str(code_root)),
        '@@SHA@@':selection_sha,'@@MODEL_Q@@':shlex.quote(MODEL_VIEW)}.items():script=script.replace(token,value)
    if '@@' in script:raise ValueError('Unrendered unique shell token')
    return script


def prepare(base,training_base,matched_config,matched_sha,code_root,source_commit,hours=1):
    base=remote(base);training_base=remote(training_base);code_root=remote(code_root)
    if not str(base).startswith(PARENT+'/') or base==training_base:raise ValueError('New isolated experiment directory required')
    base.mkdir(exist_ok=True,parents=True)
    lock=(base/'prepare.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (base/'preparation.json').exists():
        old=read(base/'preparation.json')
        if old['source_commit']!=source_commit or old['matched_config']['sha256']!=matched_sha:
            raise ValueError('Existing prepared version/config conflict')
        for ref in old['selections']:checked(ref)
        return old
    cpath=remote(matched_config)
    if fingerprints(cpath)['sha256']!=matched_sha:raise ValueError('Matched config SHA differs')
    c=read(cpath);plan=read(checked(c['plan']));bank=read(checked(c['bank']));held=read(checked(c['selection_bank']))
    parts,identity=partition_questions(plan,bank['train'],bank['diagnostic'],held['selection_validation'])
    if identity['questions']!=9088 or identity['trajectories']!=45440:
        raise ValueError('Actual 6528+512+2048 / G5 coverage required')
    q1_selection,model_binding=q1_model_binding()
    wanted={tid:g for part in parts for g in part for tid in g['trajectory_ids']}
    wanted_qids={g['qid'] for part in parts for g in part}
    found={};native_refs={};sources=[];protocol=None
    geometry=dict(multisegment_trajectories=0,short_terminal_prefix_trajectories=0,
                  max_actual_causal_context_tokens=0,retained_policy_tokens=0)
    selection_base=Path(c['selection_bank']['path']).parent.parent
    # Only sealed batch/prefix/packed metadata; never full responses or own EM.
    for producer in (training_base,selection_base):
        pc=read(producer/'config.json');shard_paths=sorted((producer/'sampling').glob('shard-*/complete.json'))
        if not shard_paths:raise ValueError('Actual sealed source sampling receipts required')
        for sp in shard_paths:
            shard=read(sp)
            if shard['run_id']!=pc['run_id'] or shard['model_revision']!=REVISION:raise ValueError('Source shard actor/run differs')
            for batch_path in shard['batches']:
                bp=Path(batch_path);batch=read(bp)
                if (batch['run_id']!=pc['run_id'] or batch['model_revision']!=REVISION or
                        batch['actor_optimizer_updates']!=0):raise ValueError('Source batch is not frozen official7B')
                ip=producer/'inputs'/sp.parent.name/(batch['batch']+'.json')
                if fingerprints(ip)['sha256']!=batch['input_sha256']:raise ValueError('Actual prefix input SHA differs')
                inputs={r['qid']:r for r in read(ip)['rows']}
                relevant=set(inputs)&wanted_qids
                if not relevant:continue
                pref=batch['prefix']['prefix'];native=batch['prefix']['actor_hidden']
                own_protocol={k:batch['prefix'][k] for k in
                    ('sampling_contract','interaction_protocol','compute_dtype','actual_compute_dtype')
                    if k in batch['prefix']}
                if not {'sampling_contract','interaction_protocol'}<=set(own_protocol):
                    raise ValueError('Original actual prefix decoder/interaction contract is missing')
                if protocol is None:protocol=own_protocol
                elif own_protocol!=protocol:
                    raise ValueError('Train/diagnostic/selection actual sampling contracts differ')
                checked(pref);checked(native);nm=read(native['path'])
                if (nm['status']!='LOSSLESS_NATIVE_FEATURES_COMPLETE' or nm['feature_layer']!=26 or
                        nm['prefix_tokens']!=20 or nm['dimension']!=3584 or nm['question_max_tokens']!=273):
                    raise ValueError('Actual source is not native L26/Prefix20')
                # Verify the packed index identity metadata, not feature tensors.
                checked(nm['files']['index.jsonl'])
                native_refs[native['path']]=native
                batch_found=0
                with gzip.open(pref['path'],'rt') as f:
                    for line in f:
                        row=json.loads(line);tid=row['trajectory_id']
                        if tid not in wanted:continue
                        g=wanted[tid];qid=g['qid']
                        if tid in found or row['qid']!=qid or qid not in inputs:raise ValueError('Repeated/wrong selected original trajectory')
                        inp=inputs[qid]
                        if (row['question_hash']!=g['question_hash'] or inp['question_hash']!=g['question_hash'] or
                                row['initial_prompt_token_ids']!=inp['initial_prompt_token_ids'] or
                                row['feature_file']!=native['path']):raise ValueError('Question/prompt/native metadata binding differs')
                        if any(k in row for k in ('strict_em','labels','predicted_mixed')):raise ValueError('Prefix-only fixture cannot carry EM or probe scores')
                        normalized=normalize_prefix(row,tid,qid,True)
                        geometry['multisegment_trajectories']+=int(len(normalized['segments'])>1)
                        geometry['short_terminal_prefix_trajectories']+=int(normalized['policy_length']<20)
                        geometry['retained_policy_tokens']+=normalized['policy_length']
                        geometry['max_actual_causal_context_tokens']=max(geometry['max_actual_causal_context_tokens'],
                            max(len(segment['context_tokens']) for segment in normalized['segments']))
                        found[tid]=row;batch_found+=1
                sources.append(dict(shard=fingerprints(sp),batch=fingerprints(bp),input=fingerprints(ip),
                    prefix=pref,native_metadata=native,selected_trajectories=batch_found))
                atomic(base/'progress.json',dict(status='CPU_REAL_PREFIX_FIXTURE_PREPARATION_ADVANCING',
                    selected_found=len(found),expected=45440,source_batches=len(sources),time=time.time(),GPU_used=False))
    if set(found)!=set(wanted):raise ValueError('Missing real saved prefix trajectories; no synthetic or rerollout fallback')
    selections=[];scripts=[]
    for shard,groups in enumerate(parts):
        root=base/('shard-%02d'%shard);(root/'inputs').mkdir(parents=True,exist_ok=True);(root/'logs').mkdir(exist_ok=True)
        if any((root/name).exists() for name in ('job.id','submission-intent.json','capture')):
            raise ValueError('Existing writer/output identity; never reprepare or overwrite it')
        tids=[tid for g in groups for tid in g['trajectory_ids']];pp=root/'inputs/prefix-only.jsonl.gz'
        temporary=pp.with_suffix('.tmp')
        with temporary.open('wb') as raw:
            with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0,filename='',compresslevel=1) as stream:
                for tid in tids:stream.write(canonical(found[tid])+b'\n')
            raw.flush();os.fsync(raw.fileno())
        if pp.exists():
            if fingerprints(pp)['sha256']!=fingerprints(temporary)['sha256']:raise ValueError('Immutable prefix subset changed')
            temporary.unlink()
        else:temporary.replace(pp)
        native={found[tid]['feature_file'] for tid in tids}
        selection=dict(schema_version=1,role='matched128_all_variants_uniform_HF_causal_replay_backend',
            model_revision=REVISION,prefix_files=[fingerprints(pp)],
            trajectories=[dict(trajectory_id=tid,qid=wanted[tid]['qid']) for tid in tids],
            native_layer26_manifests=[native_refs[p] for p in sorted(native)],
            model_files=q1_selection['model_files'],source_groups=groups,
            shard=shard,shards=4,questions=len(groups),expected_trajectories=len(tids),
            role_counts=dict(Counter(g['role'] for g in groups)),capture_layers=list(LAYERS),
            shared_identity=identity,source_model_binding=model_binding,
            source_commit=source_commit,expected_code_root=str(code_root),
            reference_runtime_commit=RUNTIME_COMMIT,label_blind_selection=True,
            original_sampling_protocol=protocol,
            training_native_equivalence_claim=False,production_accepted=False,
            source_prefix_metadata= sources,
            source_matched_config=fingerprints(cpath),source_matched_plan=c['plan'])
        ref=once(root/'inputs/selection.json',selection);selections.append(ref)
        script=render_sbatch(root,code_root,ref['sha256'],shard,hours)
        path=root/'replay.sbatch'
        if path.exists() and path.read_text()!=script:raise ValueError('Immutable Slurm script changed')
        if not path.exists():path.write_text(script)
        scripts.append(fingerprints(path))
    result=dict(status='REAL_9088Q_45440_TRAJECTORY_FOUR_SHARD_REPLAY_FIXTURES_PREPARED_NOT_CAPTURED',
        source_commit=source_commit,matched_config=fingerprints(cpath),matched_plan=c['plan'],
        source_banks=dict(train_and_diagnostic=c['bank'],selection=c['selection_bank']),
        actual_prefix_geometry=geometry,
        selections=selections,scripts=scripts,identity=identity,model_binding=model_binding,
        per_shard_questions=[len(part) for part in parts],per_shard_trajectories=[5*len(part) for part in parts],
        capture_layers=list(LAYERS),human_block_layers=[21,24,27],
        original_sampling_protocol=protocol,
        Slurm_partition='acd_u',qos='normal',gpus_per_shard=1,cpus_per_shard=8,memory_GB=64,hours=hours,
        jobs_submitted=False,new_responses_generated=0,replay_features_generated=0,
        HF_native_bitexact_claim=False,all_variant_train_eval_backend_must_match=True,time=time.time())
    once(base/'preparation.json',result);return result


def merge_capture_bank(base):
    """Post-capture CPU merge; never marks native equivalence or changes source."""
    base=remote(base);prepared=read(base/'preparation.json');layers={str(k):[] for k in LAYERS}
    groups=[];seen_qids=set();seen_tids=set();sources=[]
    for shard,selection_ref in enumerate(prepared['selections']):
        selection=read(checked(selection_ref));root=base/('shard-%02d'%shard)
        complete_path=root/'capture/complete.json';complete=read(complete_path)
        if (complete['status']!='MATCHED_REAL_G5_MULTILAYER_HF_CAUSAL_REPLAY_SHARD_COMPLETE' or
                complete['source_selection']!=selection_ref or complete['actor_updates']!=0 or
                complete['native_bitexact_equivalence_claim'] is not False):
            raise ValueError('All four actual complete immutable replay shards required')
        own_groups=selection['source_groups'];own_qids={g['qid'] for g in own_groups}
        own_tids={tid for g in own_groups for tid in g['trajectory_ids']}
        if seen_qids&own_qids or seen_tids&own_tids:raise ValueError('Completed replay shards overlap')
        if complete['questions']!=len(own_groups) or complete['trajectories']!=len(own_tids):
            raise ValueError('Completed replay scope differs from its prepared identities')
        seen_qids|=own_qids;seen_tids|=own_tids;groups+=own_groups
        for layer in layers:layers[layer]+=complete['feature_manifests_by_capture_layer'][layer]
        sources.append(fingerprints(complete_path))
    if len(seen_qids)!=9088 or len(seen_tids)!=45440:raise ValueError('Incomplete full matched replay bank')
    # This is a full shared-file SHA acceptance via the actual production reader.
    # Run remotely after capture; never carry features through the controller.
    from .replay_feature_reader import ReplayFeatureReader
    for layer,refs in layers.items():
        reader=ReplayFeatureReader(refs,int(layer))
        if set(reader.records)!=seen_tids:raise ValueError('Merged reader exact real trajectory coverage differs')
        reader.close()
        once(base/('features-layer-%s.json'%layer),refs)
    own_groups=once(base/'group-identities.json',dict(groups=groups,labels_included=False))
    result=dict(status='ALL_FOUR_MATCHED9088Q_MULTILAYER_HF_REPLAY_BANKS_SHA_VERIFIED',
        source_preparation=fingerprints(base/'preparation.json'),capture_completions=sources,
        group_identities=own_groups,questions=9088,trajectories=45440,roles=ROLE_COUNTS,
        features_by_capture_layer={layer:fingerprints(base/('features-layer-%s.json'%layer)) for layer in layers},
        full_replay_shared_file_SHA_verified=True,source_runtime='huggingface_frozen_causal_prefill',
        native_bitexact_equivalence_claim=False,accepted_for_production=False,
        permitted_use='ALL variant train and validation use this same new HF replay backend',
        actor_updates=0,new_responses_generated=0)
    once(base/'bank-complete.json',result);return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--training-base',default=PARENT)
    p.add_argument('--matched-config');p.add_argument('--matched-config-sha256')
    p.add_argument('--code-root');p.add_argument('--source-commit');p.add_argument('--hours',type=int,choices=(1,2),default=1);p.add_argument('--merge',action='store_true')
    a=p.parse_args()
    if a.merge:
        r=merge_capture_bank(a.base)
        print(json.dumps({k:r[k] for k in ('status','questions','trajectories','features_by_capture_layer','full_replay_shared_file_SHA_verified','native_bitexact_equivalence_claim')}),flush=True);return
    if any(v is None for v in (a.matched_config,a.matched_config_sha256,a.code_root,a.source_commit)):
        p.error('Preparation requires matched config/SHA and immutable code-root/commit')
    r=prepare(a.base,a.training_base,a.matched_config,a.matched_config_sha256,a.code_root,a.source_commit,a.hours)
    print(json.dumps({k:r[k] for k in ('status','identity','selections','scripts','per_shard_questions','per_shard_trajectories','jobs_submitted','replay_features_generated')}),flush=True)


if __name__=='__main__':main()
