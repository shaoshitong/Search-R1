"""Materialize frozen untouched question pools into two independent realG5 plans.

Preparation only: no download, generation, fitting, scoring, or submission.
The two new events produce ten planned IDs; the fixed label-blind five are
frozen before any generated EM exists. There is no fabricated old/source G5.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shlex
import time
import unicodedata
import uuid

from .g10_grouping import fixed_g5


REVISION = 'a09a35458c702b33eeacc393d103063234e8bc28'
SOURCE_SHA = '10d5ee25e813f3e2092247a5d2797793dd47ef5853959989cb34bf80a110d607'
ROLE = 'UNTOUCHED_CONFIRMATION_SAME_ACTOR_TWO_INDEPENDENT_G5_NOT_FOR_FIT'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def proof(path):
    path = Path(path)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def normalize(question):
    return ' '.join(unicodedata.normalize('NFKC', question).casefold().strip().rstrip('?').split())


def remote(path):
    path = Path(path).resolve()
    if not str(path).startswith('/data/user/'):
        raise ValueError('Question/rollout materialization must remain on remote shared storage')
    return path


def question_template(template, original_question):
    messages = []
    for message in template['prompt']:
        if message['content'].count(original_question) != 1:
            raise ValueError('Canonical prompt differs from the original heldout materialization recipe')
        messages.append(dict(message, content=message['content'].replace(original_question, '<QUESTION>')))
    if set(template['reward_model']['ground_truth']) != {'target'}:
        raise ValueError('Canonical strict-EM target schema changed')
    return messages


def build_rows(pool, template, original_question, tokenize, identity, qmax=273, expected=2048):
    """Pure injected tokenizer/identity boundary, preserving all frozen questions."""
    if pool.get('role') != 'UNTOUCHED_FINAL_CONFIRMATION_NOT_FOR_TUNING':
        raise ValueError('Not an untouched confirmation source question pool')
    if pool.get('source_sha256') != SOURCE_SHA or pool.get('generation_not_started') is not True or pool.get('selection_before_any_rollout_or_probe_score') is not True:
        raise ValueError('Source pool is not bound to a pre-generation untouched freeze')
    if len(pool['questions']) != expected:
        raise ValueError('Frozen pool question count differs; do not drop or replace rows')
    messages = question_template(template, original_question)
    result, seen_ids, seen_qids, seen_text = [], set(), set(), set()
    overlong = []
    for ordinal, own in enumerate(pool['questions']):
        raw = own['source_row']
        source_id = 'nq:dev:'+str(raw['id'])
        normalized_hash = hashlib.sha256(normalize(raw['question']).encode()).hexdigest()
        if own['source_question_id'] != source_id or own['canonical_question_sha256'] != normalized_hash:
            raise ValueError('Frozen source row identity/text hash changed')
        answers = raw['golden_answers']
        if not isinstance(answers, list) or not answers or any(not isinstance(a, str) for a in answers):
            raise ValueError('Source answer targets missing for authentic strict EM')
        row = dict(data_source='nq',
            prompt=[dict(m, content=m['content'].replace('<QUESTION>', raw['question'])) for m in messages],
            ability=template['ability'],
            reward_model=dict(style=template['reward_model']['style'], ground_truth=dict(target=list(answers))),
            extra_info=dict(index=source_id, split='untouched_confirmation'))
        own_identity = identity(row)
        ids = list(tokenize(row['prompt']))
        if not ids or any(type(t) is not int or t < 0 for t in ids):
            raise ValueError('Official tokenizer produced invalid prompt IDs')
        if len(ids) > qmax:
            overlong.append(dict(source_question_id=source_id, tokens=len(ids)))
        if source_id in seen_ids or own_identity['qid'] in seen_qids or normalized_hash in seen_text:
            raise ValueError('Duplicate question within untouched source pool')
        seen_ids.add(source_id)
        seen_qids.add(own_identity['qid'])
        seen_text.add(normalized_hash)
        row.update(own_identity, split='confirmation', source_ordinal=ordinal, candidate_ordinal=ordinal,
                   source_question_id=source_id, canonical_question_sha256=normalized_hash,
                   initial_prompt_token_ids=ids)
        result.append(row)
    if overlong:
        raise ValueError('Frozen pool has %d prompts longer than QMAX%d (maximum%d); cannot drop/replace frozen questions or silently change capture geometry' %
                         (len(overlong), qmax, max(v['tokens'] for v in overlong)))
    return result


def plan_events(rows, run_ids, attempt_ids, fixed_seed=20261021):
    """IDs and fixedG5 are chosen without receiving/consulting any own EM."""
    if len(run_ids) != 2 or len(set(run_ids)) != 2 or any(not x for x in run_ids):
        raise ValueError('Two distinct fresh generation run IDs required')
    if len(attempt_ids) != 2 or len(set(attempt_ids)) != 2 or any(not x for x in attempt_ids):
        raise ValueError('Two distinct explicit native sampling attempt IDs required')
    g10, all_tids = [], set()
    for row in rows:
        event_tids = []
        for run in run_ids:
            event_tids.append(['%s-outer%06d-c%04d-g%d' % (run, 1, row['candidate_ordinal'], j) for j in range(5)])
        tids = event_tids[0]+event_tids[1]
        if len(set(tids)) != 10 or all_tids.intersection(tids):
            raise ValueError('Planned native trajectory IDs collide')
        all_tids.update(tids)
        g10.append(dict(qid=row['qid'], question_hash=row['question_hash'], trajectory_ids=tids,
                        event_trajectory_ids=event_tids, canonical_question_sha256=row['canonical_question_sha256']))
    fixed = fixed_g5(g10, selection='random', seed=fixed_seed, epoch=0, role='final')
    return dict(planned_G10=g10, fixed_G5=fixed, run_ids=list(run_ids), sampling_attempt_ids=list(attempt_ids),
                fixed_seed=fixed_seed, actual_responses_generated=0, labels_observed=False,
                event_local_response_indices=[0, 1, 2, 3, 4], global_index_mapping='eventA local0..4 ->0..4; eventB local0..4 ->5..9',
                generation_events_same_official_actor_and_protocol=True)


def checked(item):
    path = Path(item['path'])
    if path.stat().st_size != item['bytes'] or sha(path) != item['sha256']:
        raise ValueError('Pinned preparation artifact changed: '+str(path))
    return path


def preflight_event(config_path):
    """Read-only launch gate, usable in prepared sbatch before generic collector."""
    c = read(remote(config_path))
    if c.get('confirmation_role') != ROLE or not c.get('evaluation_only') or c.get('fitting_authorized') is not False or c.get('fit_epochs') != 0:
        raise ValueError('Confirmation collector cannot fit or tune a probe')
    if c['model_revision'] != REVISION or c.get('actor_updates_authorized') is not False or c.get('actor_optimizer_updates') != 0:
        raise ValueError('Confirmation requires the unchanged official initial actor')
    if c['capture_prefix_tokens'] != 20 or c['question_max_tokens'] != 273 or c.get('prefix_diversity'):
        raise ValueError('Confirmation rollout form differs from the frozen training prefix20 contract')
    for item in c['immutable_preparation_sources']:
        checked(item)
    for relative, expected_sha in c['collector_code_files_sha256'].items():
        if sha(Path(c['code_root'])/relative) != expected_sha:
            raise ValueError('Confirmation collector source changed: '+relative)
    actor = read(c['actor_complete']['path'])
    checked(c['actor_complete'])
    if actor['revision'] != REVISION or actor['status'] != 'MIGRATION_ALL_SELECTED_ASSETS_VERIFIED':
        raise ValueError('Official actor file proof missing')
    for name, expected in actor['files'].items():
        if Path(name).name != name:
            raise ValueError('Actor file escaped its snapshot')
        checked(dict(expected, path=str(Path(c['model_dir'])/name)))
    return dict(status='CONFIRMATION_EVENT_SOURCE_ACTOR_SHA_PREFLIGHT_PASS', run_id=c['run_id'],
                generation_not_performed=True, actor_updates=0, fitting_allowed=False)


def render_event_script(event_root, code_root):
    runtime = Path(code_root)/'reproduction/operations/qwen7b_warmup/runtime.sh'
    configuration = Path(event_root)/'config.json'
    script = '''#!/usr/bin/env bash
#SBATCH --job-name=q7b-confirm-G5
#SBATCH --partition=acd_u
#SBATCH --account=mou176
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH --time=12:00:00
#SBATCH --array=0-3
#SBATCH --output=__EVENT_ROOT__/logs/sampling-%A_%a.log
set -euo pipefail
export QWEN7B_CODE_ROOT=__CODE_ROOT__
bash __RUNTIME__ -u -m reproduction.operations.qwen7b_warmup.confirmation_materialize --check-event-config __CONFIG__
srun --kill-on-bad-exit=1 bash __RUNTIME__ -u -m reproduction.operations.qwen7b_warmup.shard --config __CONFIG__ --shard "$SLURM_ARRAY_TASK_ID"
'''
    for key, value in [('__EVENT_ROOT__', str(event_root)), ('__CODE_ROOT__', shlex.quote(str(code_root))),
                       ('__RUNTIME__', shlex.quote(str(runtime))), ('__CONFIG__', shlex.quote(str(configuration)))]:
        script = script.replace(key, value)
    return script


def prepare(training, frozen_manifest, destination, code_root, source_commit, fixed_seed=20261021):
    training, frozen_manifest, destination, code_root = map(remote, (training, frozen_manifest, destination, code_root))
    if destination.exists():
        raise ValueError('Materialized output already exists; verify/reuse its exact plan instead of overwriting')
    if len(source_commit) != 40 or any(v not in '0123456789abcdef' for v in source_commit):
        raise ValueError('Exact published source commit required')
    manifest, source = read(frozen_manifest), read(training/'config.json')
    if manifest['status'] != 'TWO_DISJOINT_UNTOUCHED_2048_QUESTION_POOLS_FROZEN' or manifest['source_sha256'] != SOURCE_SHA:
        raise ValueError('Wrong frozen untouched-pool manifest')
    if manifest['score_or_generated_EM_observed'] or manifest['fit_or_tuning_allowed']:
        raise ValueError('These source pools are not untouched final confirmation')
    if source['model_revision'] != REVISION or source['capture_prefix_tokens'] != 20 or source['question_max_tokens'] != 273:
        raise ValueError('Training official actor/capture contract differs')
    old = training/'selection-2048-20261020'
    if sha(old/'inputs/questions.json') != manifest['excluded_viewed2048_sha256']:
        raise ValueError('Previously viewed validation exclusion proof changed')
    excluded_text = set()
    for item in manifest['excluded_canonical'].values():
        if sha(item['path']) != item['sha256']:
            raise ValueError('Canonical train/development/final exclusion SHA changed')
        with Path(item['path']).open() as f:
            for line in f:
                excluded_text.add(hashlib.sha256(normalize(json.loads(line)['question']).encode()).hexdigest())
    previous = read(old/'inputs/questions.json')['rows']
    previous_qids = {r['qid'] for r in previous}
    previous_text = {r['canonical_question_sha256'] for r in previous}
    if sha(manifest['source_path']) != SOURCE_SHA:
        raise ValueError('Original heldout source dataset SHA changed')
    source_records = {}
    with Path(manifest['source_path']).open() as f:
        for line in f:
            row = json.loads(line)
            key = str(row['id'])
            if key in source_records:
                raise ValueError('Repeated official NQ source ID')
            source_records[key] = row
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer
    from reproduction.prefix20_online.pool import identity
    from reproduction.prefix20_online.rollout import prompt_tokens
    common = Path(source['canonical_train']).parent
    canonical = read(common/'manifest.json')
    if sha(common/'train.parquet') != canonical['splits']['train']['sha256']:
        raise ValueError('Canonical prompt template Parquet changed')
    template = next(pq.ParquetFile(str(common/'train.parquet')).iter_batches(batch_size=1)).to_pylist()[0]
    first_question = json.loads((common/'train.jsonl').open().readline())['question']
    tokenizer = AutoTokenizer.from_pretrained(source['model_dir'], local_files_only=True, trust_remote_code=False)
    all_rows, all_sources, all_qids, all_text = {}, {}, set(), set()
    for name in ('confirmation2048', 'stability2048'):
        item = manifest['pools'][name]
        if sha(item['path']) != item['sha256']:
            raise ValueError('Frozen pool payload SHA changed')
        pool = read(item['path'])
        for own in pool['questions']:
            if own['source_row'] != source_records.get(str(own['source_row']['id'])):
                raise ValueError('Frozen question/answer row differs from SHA-verified official dataset')
        ids = [r['source_question_id'] for r in pool['questions']]
        if hashlib.sha256(json.dumps(ids, separators=(',', ':')).encode()).hexdigest() != item['ordered_source_id_sha256']:
            raise ValueError('Frozen source question order changed')
        rows = build_rows(pool, template, first_question, lambda messages: prompt_tokens(tokenizer, messages), identity)
        qids = {r['qid'] for r in rows}
        text = {r['canonical_question_sha256'] for r in rows}
        if qids & (all_qids | previous_qids) or text & (all_text | previous_text | excluded_text):
            raise ValueError('Confirmation/stability overlaps training/viewedvalidation/final or each other')
        all_qids.update(qids)
        all_text.update(text)
        all_rows[name] = rows
        all_sources[name] = proof(item['path'])
    files = ('confirmation_materialize.py', 'g10_grouping.py', 'shard.py', 'sampling.py', 'native_capture.py', 'packed_features.py', 'runtime.sh')
    code_files = {'reproduction/operations/qwen7b_warmup/'+name:sha(code_root/'reproduction/operations/qwen7b_warmup'/name) for name in files}
    for name in ('reproduction/prefix_selection200/rollout.py', 'reproduction/prefix_selection200/interaction_protocol.py',
                 'reproduction/prefix_selection200/contract.py', 'reproduction/prefix_selection200/b_progressive.py',
                 'reproduction/prefix20_online/pool.py', 'reproduction/prefix20_online/rollout.py',
                 'search_g1/async_rollout_runtime.py', 'search_r1/llm_agent/generation.py'):
        code_files[name] = sha(code_root/name)
    actor_complete = proof(Path(source['model_dir'])/'complete.json')
    destination.mkdir(parents=True, exist_ok=False)
    source_proofs = [proof(frozen_manifest), proof(training/'config.json'), proof(old/'inputs/questions.json')]+list(all_sources.values())
    plans = {}
    for name, rows in all_rows.items():
        pool_root = destination/name
        run_ids = [uuid.uuid4().hex, uuid.uuid4().hex]
        attempts = ['confirmation-'+run for run in run_ids]
        planned = plan_events(rows, run_ids, attempts, fixed_seed)
        atomic(pool_root/'inputs/questions.json', dict(rows=rows, source_pool=all_sources[name],
            ordered_qid_sha256=digest([r['qid'] for r in rows]),
            ordered_source_id_sha256=digest([r['source_question_id'] for r in rows]),
            ordered_normalized_question_sha256=digest([r['canonical_question_sha256'] for r in rows])))
        atomic(pool_root/'planned-g10-and-fixed-g5.json', planned)
        event_configs = []
        for event_index in range(2):
            event_root = pool_root/('event-%s' % ('A' if event_index == 0 else 'B'))
            event_entries = []
            for shard in range(4):
                assigned = rows[shard::4]
                for offset in range(0, len(assigned), 128):
                    path = event_root/'inputs'/('shard-%02d' % shard)/('batch-%06d.json' % (offset//128))
                    atomic(path, dict(shard=shard, split='confirmation', rows=assigned[offset:offset+128], model_revision=REVISION))
                    event_entries.append(dict(proof(path), questions=len(assigned[offset:offset+128]), shard=shard))
            atomic(event_root/'inputs/manifest.json', dict(batches=event_entries, questions=2048, trajectories=10240,
                role=ROLE, ordered_qid_sha256=digest([r['qid'] for r in rows]), new_event=True))
            configuration = dict(copy.deepcopy(source), base=str(event_root),
                confirmation_role=ROLE, run_id=run_ids[event_index], sampling_attempt_id=attempts[event_index],
                actor_version='official7b-'+REVISION, actor_updates_authorized=False, actor_optimizer_updates=0,
                warmup_only=True, evaluation_only=True, diagnostic_only=False,
                train_questions=0, train_trajectories=0, diagnostic_questions=0, diagnostic_trajectories=0,
                evaluation_questions=2048, evaluation_trajectories=10240, fit_epochs=0,
                fitting_authorized=False, tuning_allowed=False, prefix_diversity={}, prefix_diversity_selection=False,
                sampling_shards=4, gpus_per_sampling_job=2, source_commit=source_commit,
                code_root=str(code_root), collector_code_files_sha256=code_files,
                actor_complete=actor_complete, immutable_preparation_sources=source_proofs+
                    [proof(pool_root/'inputs/questions.json'), proof(pool_root/'planned-g10-and-fixed-g5.json'),
                     proof(event_root/'inputs/manifest.json')]+event_entries,
                fixed_view_precommitted=True, fixed_view_seed=fixed_seed, event_index=event_index,
                actual_responses_generated=0, same_generation_protocol_as_training=True)
            atomic(event_root/'config.json', configuration)
            event_configs.append(proof(event_root/'config.json'))
            (event_root/'logs').mkdir()
            (event_root/'sampling.sbatch').write_text(render_event_script(event_root, code_root))
        plans[name] = dict(question_plan=proof(pool_root/'inputs/questions.json'),
            planned_G10_and_fixed_G5=proof(pool_root/'planned-g10-and-fixed-g5.json'),
            event_configs=event_configs, questions=2048, planned_real_trajectories=20480,
            actual_responses_generated=0, source_pool=all_sources[name])
    result = dict(status='CONFIRMATION_STABILITY_TWO_REAL_G5_EVENT_INPUTS_PREPARED_NOT_GENERATED',
        role=ROLE, source_commit=source_commit, source_freeze=proof(frozen_manifest), pools=plans,
        isolation=dict(disjoint_materialized_qids=True, disjoint_normalized_questions=True,
            excluded_canonical_source_SHA_verified=True, viewed_validation2048_excluded=True,
            all_materialized_qids_sha256=digest(sorted(all_qids)),
            all_normalized_question_hashes_sha256=digest(sorted(all_text))),
        fixed_G5='label-blind deterministic random5of10, precommitted before generation',
        no_fake_original_G5=True, fit_or_tuning_allowed=False, actor_updates=0,
        actual_responses_generated=0, jobs_submitted=False, runtime_qualified=False,
        remaining='Run/qualify real collectors, fresh-audit actual event manifests and ownEM, then bind actual G10/fixedG5; this preparation is not generation acceptance.',
        time=time.time())
    atomic(destination/'materialization.json', result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--training-base')
    p.add_argument('--frozen-manifest')
    p.add_argument('--base')
    p.add_argument('--code-root')
    p.add_argument('--source-commit')
    p.add_argument('--fixed-seed', type=int, default=20261021)
    p.add_argument('--check-event-config')
    a = p.parse_args()
    if a.check_event_config:
        result = preflight_event(a.check_event_config)
    else:
        if not all((a.training_base, a.frozen_manifest, a.base, a.code_root, a.source_commit)):
            p.error('Preparation requires training-base, frozen-manifest, base, code-root and source-commit')
        result = prepare(a.training_base, a.frozen_manifest, a.base, a.code_root, a.source_commit, a.fixed_seed)
    print(json.dumps(result), flush=True)
