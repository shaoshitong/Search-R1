"""Fork sealed extra-G5 *static inputs*, without rereading original bulk rollouts.

Only metadata/token-input JSON and the immutable collector source are freshly
hashed here. Original full-rollout proofs were accepted by the parent preparation
and are explicitly inherited, not represented as newly checked. GPU collection
still runs the existing qualified collector, with a distinct run/attempt/slot.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import time
import uuid
ROLE="ISOLATED_OFFICIAL7B_EXTRA_REAL_G5_FOR_TRAINING_G10"
REVISION="a09a35458c702b33eeacc393d103063234e8bc28"

def canonical_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(",",":")).encode()).hexdigest()

def validate_config(c):
    """Use the already bound immutable collector validator, never change it."""
    path=Path(c["code_root"])/"reproduction/operations/qwen7b_warmup/extra_g5_prepare.py"
    relative="reproduction/operations/qwen7b_warmup/extra_g5_prepare.py"
    if sha(path)!=c["code_files_sha256"][relative]:raise ValueError("Collector validator SHA differs")
    spec=importlib.util.spec_from_file_location("qualified_extra_g5_validator",path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module.validate_config(c)

PARENT='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019'
A=PARENT+'/algorithm-improvement-20261021'
PARENT_BASE=A+'/training-extraG5-matched6528-20261021/materialized'
COLLECTOR_COMMIT='2b293d6e6a0be5a8b99f976b3ac0c2b15e19ee40'
PINNED_PARENT_SHA={
 'config.json':'3957642964d692329b98b2de216c9b7cb2ebe9866a667283e0055bb36223aa67',
 'prepared.json':'3c4c3bb8b20f08ca342deecf11ccbd6e2cb27fd02a6eaddea36cbed85910fb96',
 'inputs/manifest.json':'d8537ce397f48c13294f4d420d66c5cd975011c259c15a0d3c0470a6c02d1d4b',
 'inputs/questions.json':'265ce78e1e19e87263256348ef7fa54bf09987f7be65e331b3541e427a11f703'}


def target(offset):
    if offset not in (10,15):raise ValueError('Only new offset10/15 phases are allowed')
    return A+'/training-extraG5-matched6528-offset%d-20261021/materialized'%offset


def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for v in iter(lambda:f.read(1024*1024),b''):h.update(v)
    return h.hexdigest()
def proof(p):
    p=Path(p);return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def verify(ref):
    p=Path(ref['path'])
    if p.stat().st_size!=ref['bytes'] or sha(p)!=ref['sha256']:raise ValueError('Static input proof differs: '+str(p))
    return p

def write_once(p,raw):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists():
        if p.read_bytes()!=raw:raise ValueError('Refusing to overwrite different immutable fork file: '+str(p))
        return
    with p.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
def json_once(p,v):write_once(p,(json.dumps(v,indent=2,sort_keys=True)+'\n').encode())


def render_sbatch(base,code_root,offset):
    if offset not in (10,15):raise ValueError('Only offset10/15')
    for p in (base,code_root):
        if not Path(p).is_absolute() or any(c in str(p) for c in '\n\r\t ;`$'):raise ValueError('Safe absolute path required')
    return '''#!/usr/bin/env bash
#SBATCH --job-name=q7b-g5-m6528-o@@OFFSET@@
#SBATCH --partition=acd_u
#SBATCH --account=mou176
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00
#SBATCH --array=0-3%2
#SBATCH --output=@@BASE@@/logs/sampling-%A_%a.log
set -euo pipefail
export QWEN7B_CODE_ROOT=@@CODE@@
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
srun --kill-on-bad-exit=1 bash "$QWEN7B_CODE_ROOT/reproduction/operations/qwen7b_warmup/runtime.sh" -u -m reproduction.operations.qwen7b_warmup.extra_g5_sampler --config @@CONFIG@@ --shard "$SLURM_ARRAY_TASK_ID"
'''.replace('@@OFFSET@@',str(offset)).replace('@@BASE@@',str(base)).replace('@@CODE@@',shlex.quote(str(code_root))).replace('@@CONFIG@@',shlex.quote(str(Path(base)/'config.json')))


def inspect_parent(parent,pins,expected_questions=6528):
    """Fresh static checks only; deliberately never open original_batches paths."""
    parent=Path(parent);refs={}
    for relative,expected in pins.items():
        p=parent/relative;ref=proof(p)
        if ref['sha256']!=expected:raise ValueError('Pinned parent static SHA differs: '+relative)
        refs[relative]=ref
    c=validate_config(read(parent/'config.json'));done=read(parent/'prepared.json')
    manifest=read(parent/'inputs/manifest.json');questions=read(parent/'inputs/questions.json')
    if (c['base']!=str(parent) or c['extra_response_offset']!=5 or c['source_commit']!=COLLECTOR_COMMIT or
        c['train_questions']!=expected_questions or c['train_trajectories']!=5*expected_questions):raise ValueError('Not the accepted matched parent offset5')
    if (done['source_commit']!=c['source_commit'] or done['extra_run_id']!=c['run_id'] or
        done['config']!=refs['config.json'] or done['input_manifest']!=refs['inputs/manifest.json']):raise ValueError('Parent preparation/config binding differs')
    groups=questions['groups'];qids=[g['qid'] for g in groups]
    if (len(qids)!=expected_questions or len(set(qids))!=expected_questions or
        questions['ordered_qid_sha256']!=canonical_sha(qids) or manifest['ordered_qid_sha256']!=canonical_sha(qids) or
        done['source_qid_plan_sha256']!=canonical_sha(qids) or manifest['run_id']!=c['run_id'] or
        manifest['response_indices']!=list(range(5,10)) or manifest['questions']!=expected_questions or
        manifest['trajectories']!=5*expected_questions or not manifest['originalG5_preserved']):raise ValueError('Matched question identity/count binding differs')
    if not questions['validation_qids_excluded'] or not questions['final_qids_excluded']:raise ValueError('Inherited exclusions required')
    if questions['matched_cohort_plan']!=c['matched_cohort_plan']:raise ValueError('Matched plan differs')
    verify(c['matched_cohort_plan'])
    for g in groups:
        if (len(g['trajectory_ids'])!=5 or len(set(g['trajectory_ids']))!=5 or len(g['labels'])!=5 or
            any(type(y) is not int or y not in (0,1) for y in g['labels']) or
            not questions['original_native_seeds_by_qid'].get(g['qid'])):raise ValueError('Incomplete original genuine G5/own-EM/seed metadata')
    batch_refs=[];partitions=[];per_shard={i:[] for i in range(4)};byqid={g['qid']:g for g in groups}
    for item in manifest['batches']:
        p=verify(item)
        relative=p.relative_to(parent)
        if relative.parts[:1]!=('inputs',) or p.name[:6]!='batch-':raise ValueError('Parent input path escape')
        value=read(p);shard=item['shard']
        if shard not in per_shard or value['shard']!=shard or value['split']!='train' or len(value['rows'])!=item['questions']:raise ValueError('Static batch counts/split differ')
        for row in value['rows']:
            g=byqid[row['qid']]
            if (row['question_hash']!=g['question_hash'] or row['original_trajectory_ids']!=g['trajectory_ids'] or not row['initial_prompt_token_ids']):raise ValueError('Input row identity/original prompt binding differs')
            per_shard[shard].append(row['qid'])
        batch_refs.append(dict(item,relative=str(relative)))
    for shard in range(4):
        expected=qids[shard::4]
        if per_shard[shard]!=expected:raise ValueError('Shard question order differs from sealed matched cohort')
        p=parent/'inputs'/('shard-%02d'%shard)/'partition.json';v=read(p)
        if v!=dict(shard=shard,questions=len(expected),trajectories=5*len(expected)):raise ValueError('Partition counts differ')
        partitions.append(dict(proof(p),relative=str(p.relative_to(parent))))
    code=Path(c['code_root'])
    for relative,digest in c['code_files_sha256'].items():
        if sha(code/relative)!=digest:raise ValueError('Qualified immutable collector code differs: '+relative)
    return c,manifest,questions,dict(parent_static=refs,batches=batch_refs,partitions=partitions,
        collector_code_files_fresh_SHA=True,original_bulk_payloads_freshly_read=False,
        original_bulk_proofs_inherited_count=len(questions['original_batches']),
        inherited_original_bulk_proofs_sha256=canonical_sha(questions['original_batches']))


def validate_destination(parent,base):
    if not str(base).startswith('/data/user/') or base==parent or base.resolve()!=base:raise ValueError('Separate remote shared-storage destination required')


def prepare(parent,base,offset,pins=PINNED_PARENT_SHA,preparer_publication=None,expected_questions=6528):
    parent,base=Path(parent),Path(base)
    validate_destination(parent,base)
    if offset not in (10,15):raise ValueError('Only new offset10/15')
    c,manifest,questions,audit=inspect_parent(parent,pins,expected_questions)
    intent_path=base/'fork-intent.json'
    if base.exists() and not intent_path.exists() and any(base.iterdir()):raise ValueError('Existing unowned destination must be reconciled before fork preparation')
    base.mkdir(parents=True,exist_ok=True);(base/'logs').mkdir(exist_ok=True)
    identity=dict(parent=str(parent),base=str(base),offset=offset,parent_static=audit['parent_static'])
    if intent_path.exists():
        intent=read(intent_path)
        if any(intent.get(k)!=v for k,v in identity.items()):raise ValueError('Conflicting or unknown fork preparation identity')
    else:
        intent=dict(identity,run_id=uuid.uuid4().hex,time=time.time());intent['sampling_attempt_id']='extra-real-g5-'+intent['run_id'];json_once(intent_path,intent)
    if intent['run_id'] in (c['run_id'],c['original_run_id']) or intent['sampling_attempt_id'] in (c['sampling_attempt_id'],c.get('original_sampling_attempt_id')):raise ValueError('Fork run/attempt identity collides')
    for item in audit['batches']+audit['partitions']:
        write_once(base/item['relative'],Path(item['path']).read_bytes())
    write_once(base/'inputs/questions.json',Path(audit['parent_static']['inputs/questions.json']['path']).read_bytes())
    newc=copy.deepcopy(c);newc.update(base=str(base),run_id=intent['run_id'],sampling_attempt_id=intent['sampling_attempt_id'],
        extra_response_offset=offset,extra_response_indices=list(range(offset,offset+5)),
        extra_g5_fork=dict(parent_prepared=audit['parent_static']['prepared.json'],parent_run_id=c['run_id'],
            static_inputs_reused=True,original_bulk_SHA_freshly_rechecked=False,
            original_bulk_proofs_inherited=True,preparer_publication=preparer_publication,
            future_G20_acceptance_requires_actual_same_question_all_phase_seed_disjointness=True))
    validate_config(newc)
    entries=[dict(proof(base/item['relative']),questions=item['questions'],shard=item['shard']) for item in audit['batches']]
    newm=copy.deepcopy(manifest);newm.update(batches=entries,run_id=newc['run_id'],response_indices=newc['extra_response_indices'])
    json_once(base/'inputs/manifest.json',newm);json_once(base/'config.json',newc)
    write_once(base/'sampling.sbatch',render_sbatch(base,c['code_root'],offset).encode())
    provenance=dict(audit,role='STATIC_MATCHED6528_EXTRA_G5_FORK_NO_ORIGINAL_BULK_REREAD',parent_run_id=c['run_id'],
        run_id=newc['run_id'],sampling_attempt_id=newc['sampling_attempt_id'],response_indices=newc['extra_response_indices'],
        same_original_questions_prompt_tokens_own_EM_and_original_seeds=True,original_G5_modified=False,
        collector_source_commit=c['source_commit'],collector_code_root=c['code_root'],preparer_publication=preparer_publication,
        seed_disjointness=dict(status='REQUIRED_AT_FUTURE_G20_MERGE_NOT_YET_VERIFIED',
            phases=['original0..4','extra5..9','extra10..14','extra15..19'],scope='actual sampling_journal seeds pairwise disjoint per original question; IDs/run/attempt must also be distinct'),
        fresh_child_inputs=[proof(base/item['relative']) for item in audit['batches']+audit['partitions']],
        questions=proof(base/'inputs/questions.json'))
    json_once(base/'fork-provenance.json',provenance)
    result=dict(status='EXTRA_REAL_G5_STATIC_FORK_PREPARED_NOT_SUBMITTED',role=ROLE,base=str(base),questions=expected_questions,
        trajectories=5*expected_questions,source_commit=c['source_commit'],source_qid_plan_sha256=manifest['ordered_qid_sha256'],
        original_run_id=c['original_run_id'],extra_run_id=newc['run_id'],response_indices=newc['extra_response_indices'],
        actor_updates=0,actual_extra_rollouts_generated=0,sampler_runtime_qualified=False,original_bulk_payloads_freshly_read=False,
        fork_provenance=proof(base/'fork-provenance.json'),config=proof(base/'config.json'),input_manifest=proof(base/'inputs/manifest.json'),
        sampling_script=proof(base/'sampling.sbatch'),resources=dict(qos='normal',partition='acd_u',gpus_per_job=2,array='0-3%2',dependency=None))
    json_once(base/'prepared.json',result)
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--parent',default=PARENT_BASE);p.add_argument('--base',required=True)
    p.add_argument('--offset',type=int,choices=[10,15],required=True);p.add_argument('--preparer-publication',required=True);a=p.parse_args()
    if a.parent!=PARENT_BASE or a.base!=target(a.offset):p.error('Only the pinned parent and separate declared offset target are allowed')
    pub=proof(a.preparer_publication);result=prepare(a.parent,a.base,a.offset,preparer_publication=pub)
    print(json.dumps(result),flush=True)
if __name__=='__main__':main()
