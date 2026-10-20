"""Two fresh G5 events for the already-viewed recipe-selection validation pool.

Never claims this2048 is untouched final. Reads question/prompt inputs only,
never old rollouts/EM/probe predictions. Both G5 events are newly generated;
their label-blind fixedG5 view is committed before any new response exists.
Also exports a label-free final-exclusion plan from SHA-bound materialized
confirmation/stability question inputs for training-only extra-G5 preparation.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time
import uuid

from .confirmation_materialize import (REVISION, atomic, checked, digest, plan_events,
    proof, read, remote, render_event_script, sha)
from .g10_grouping import fixed_g5


ROLE = 'RECIPE_SELECTION_VALIDATION_TWO_FRESH_G5_NOT_UNTOUCHED_FINAL'
MODULE = 'reproduction.operations.qwen7b_warmup.validation_g10_materialize'


def question_rows_only(rows, expected=2048):
    """Whitelist source-question fields; generated labels/scores are not accessed."""
    if len(rows) != expected:
        raise ValueError('Keep exactly the original frozen validation question pool')
    result, seen, seen_text = [], set(), set()
    for ordinal, source in enumerate(rows):
        # No copying of source.get(labels/p_j/strict_em/predicted_mixed/...):
        # those fields are neither required nor consulted even if supplied.
        row = {key:copy.deepcopy(source[key]) for key in (
            'qid', 'question_hash', 'data_source', 'prompt', 'ability', 'reward_model',
            'source_question_id', 'canonical_question_sha256', 'initial_prompt_token_ids')}
        extra = source['extra_info']
        row['extra_info'] = dict(index=extra['index'], split='recipe_selection_validation')
        if row['qid'] in seen or row['canonical_question_sha256'] in seen_text:
            raise ValueError('Validation question repeats')
        if row['qid'] != json.dumps([str(row['data_source']), str(extra['index'])]):
            raise ValueError('Validation qid differs from its source question identity')
        prompt_hash = hashlib.sha256('\n'.join(m['content'] for m in row['prompt']).encode()).hexdigest()
        if row['question_hash'] != prompt_hash:
            raise ValueError('Validation prompt differs from its pinned question hash')
        tokens = row['initial_prompt_token_ids']
        if not 0 < len(tokens) <= 273 or any(type(t) is not int or t < 0 for t in tokens):
            raise ValueError('Validation prompt token geometry differs from training QMAX273')
        if 'normalized_question_hash' in source:
            row['normalized_question_hash'] = source['normalized_question_hash']
        row.update(split='recipe_selection_validation', source_ordinal=ordinal, candidate_ordinal=ordinal)
        seen.add(row['qid'])
        seen_text.add(row['canonical_question_sha256'])
        result.append(row)
    return result


def final_exclusion_from_rows(pools, source_proofs):
    """Construct only question identity exclusions; never access generated EM."""
    questions, qids, hashes, normalized = [], set(), set(), set()
    for name, rows in pools.items():
        for row in rows:
            qid, question_hash = row['qid'], row['question_hash']
            norm = row['canonical_question_sha256']
            if qid in qids or question_hash in hashes or norm in normalized:
                raise ValueError('Materialized final question pools overlap')
            questions.append(dict(qid=qid, question_hash=question_hash,
                                  canonical_question_sha256=norm, pool=name))
            qids.add(qid)
            hashes.add(question_hash)
            normalized.add(norm)
    return dict(role='final', frozen=True, questions=questions,
        ordered_qid_sha256=digest([r['qid'] for r in questions]),
        ordered_question_hash_sha256=digest([r['question_hash'] for r in questions]),
        ordered_normalized_question_sha256=digest([r['canonical_question_sha256'] for r in questions]),
        source_proofs=source_proofs, purpose='training-generation exclusion only; not a fitted/scored final result',
        labels_read=False, generated_responses_required=False)


def load_final_exclusion(materialization_path, expected_sha=None):
    path = remote(materialization_path)
    if expected_sha and sha(path) != expected_sha:
        raise ValueError('Known materialized final question manifest SHA differs')
    manifest = read(path)
    if manifest['status'] != 'CONFIRMATION_STABILITY_TWO_REAL_G5_EVENT_INPUTS_PREPARED_NOT_GENERATED' or manifest['fit_or_tuning_allowed'] is not False:
        raise ValueError('Not an immutable final confirmation/stability question materialization')
    pools, proofs = {}, [proof(path)]
    for name in ('confirmation2048', 'stability2048'):
        item = manifest['pools'][name]['question_plan']
        checked(item)
        payload = read(item['path'])
        rows = payload['rows']
        if len(rows) != 2048 or digest([r['qid'] for r in rows]) != payload['ordered_qid_sha256']:
            raise ValueError('Materialized final question count/order SHA differs')
        if digest([r['canonical_question_sha256'] for r in rows]) != payload['ordered_normalized_question_sha256']:
            raise ValueError('Materialized final normalized question SHA differs')
        pools[name] = rows
        proofs.append(item)
    return final_exclusion_from_rows(pools, proofs)


def write_final_exclusion(materialization_path, output, expected_sha=None):
    output = remote(output)
    value = load_final_exclusion(materialization_path, expected_sha)
    if output.exists():
        if read(output) != value:
            raise ValueError('Existing final-exclusion plan is a different immutable identity set')
    else:
        atomic(output, value)
    return dict(status='FINAL_QUESTION_EXCLUSION_PLAN_SHA_VERIFIED', plan=proof(output),
                questions=len(value['questions']), labels_read=False)


def validate_isolation(rows, train_and_diagnostic, final_plan):
    qids = {r['qid'] for r in rows}
    hashes = {r['question_hash'] for r in rows}
    norms = {r['canonical_question_sha256'] for r in rows}
    for other in (train_and_diagnostic, final_plan['questions']):
        if qids & {r['qid'] for r in other} or hashes & {r['question_hash'] for r in other}:
            raise ValueError('Recipe-selection validation overlaps training/512diagnostic/final question identities')
        if norms & {r['canonical_question_sha256'] for r in other if 'canonical_question_sha256' in r}:
            raise ValueError('Recipe-selection validation overlaps normalized final question text')
    return dict(validation_questions=len(rows), validation_qid_sha256=digest([r['qid'] for r in rows]),
        validation_question_hash_sha256=digest([r['question_hash'] for r in rows]),
        validation_normalized_question_sha256=digest([r['canonical_question_sha256'] for r in rows]),
        disjoint_train_and512diagnostic=True, disjoint_confirmation_and_stability=True)


def validation_plan_events(rows, run_ids, attempt_ids, fixed_seed):
    result = plan_events(rows, run_ids, attempt_ids, fixed_seed)
    # Reuse ID construction, but assign the genuine validation role by invoking
    # the label-blind selector as validation. No final role/config is relabeled.
    result['fixed_G5'] = fixed_g5(result['planned_G10'], selection='random', seed=fixed_seed,
                                 epoch=0, role='validation')
    result.update(role=ROLE, untouched_final=False, recipe_selection_allowed=True,
                  fitting_on_validation_allowed=False)
    return result


def render_validation_script(event_root, code_root):
    script = render_event_script(event_root, code_root)
    script = script.replace('reproduction.operations.qwen7b_warmup.confirmation_materialize --check-event-config',
                            MODULE+' --check-event-config')
    return script.replace('#SBATCH --job-name=q7b-confirm-G5', '#SBATCH --job-name=q7b-valid-G5')


def validation_preflight(config_path):
    """Own explicit validation gate; never impersonates the final-only gate."""
    c = read(remote(config_path))
    if c.get('validation_role') != ROLE or c.get('confirmation_role') is not None:
        raise ValueError('Validation collector role differs; cannot relabel a final config')
    if c.get('tuning_allowed') is not True or c.get('untouched_final') is not False:
        raise ValueError('Must explicitly identify recipe-selection validation')
    if not c.get('evaluation_only') or c.get('fitting_authorized') is not False or c.get('fit_epochs') != 0:
        raise ValueError('Validation responses cannot fit the probe or actor')
    if c['model_revision'] != REVISION or c.get('actor_updates_authorized') is not False or c.get('actor_optimizer_updates') != 0:
        raise ValueError('Validation must use the same frozen official initial actor')
    if c['capture_prefix_tokens'] != 20 or c['question_max_tokens'] != 273 or c.get('prefix_diversity'):
        raise ValueError('Validation rollout form differs from training prefix20')
    for item in c['immutable_preparation_sources']:
        checked(item)
    for relative, expected in c['collector_code_files_sha256'].items():
        if sha(Path(c['code_root'])/relative) != expected:
            raise ValueError('Frozen validation collector source changed')
    checked(c['actor_complete'])
    actor = read(c['actor_complete']['path'])
    if actor['revision'] != REVISION or actor['status'] != 'MIGRATION_ALL_SELECTED_ASSETS_VERIFIED':
        raise ValueError('Official actor identity proof differs')
    for name, item in actor['files'].items():
        if Path(name).name != name:
            raise ValueError('Actor snapshot path escaped')
        checked(dict(item, path=str(Path(c['model_dir'])/name)))
    return dict(status='RECIPE_SELECTION_VALIDATION_ACTOR_SOURCE_PREFLIGHT_PASS',
        run_id=c['run_id'], tuning_allowed=True, untouched_final=False,
        fitting_allowed=False, generation_not_performed=True)


def prepare(training, old_selection, final_materialization, destination, code_root,
            source_commit, fixed_seed=20261021, final_materialization_sha=None):
    training, old_selection, destination, code_root = map(remote, (training, old_selection, destination, code_root))
    if destination.exists():
        raise ValueError('Use a new isolated validation plan; never overwrite an existing writer/output')
    if len(source_commit) != 40 or any(c not in '0123456789abcdef' for c in source_commit):
        raise ValueError('Exact published source commit required')
    source, previous = read(training/'config.json'), read(old_selection/'config.json')
    if source['model_revision'] != REVISION or previous['model_revision'] != REVISION:
        raise ValueError('Training/previous validation actor revisions differ')
    if source['capture_prefix_tokens'] != 20 or source['question_max_tokens'] != 273:
        raise ValueError('Validation must retain the training prefix20/QMAX273 generation contract')
    for key in ('model_revision', 'model_dir', 'capture_prefix_tokens', 'question_max_tokens',
                'retrieval_index', 'retrieval_corpus', 'retrieval_encoder', 'retrieval_data_cache'):
        if source[key] != previous[key]:
            raise ValueError('Previous validation and training protocol differ: '+key)
    old_input = read(old_selection/'inputs/questions.json')
    rows = question_rows_only(old_input['rows'])
    old_order_sha = hashlib.sha256(json.dumps([r['qid'] for r in rows]).encode()).hexdigest()
    if old_input['ordered_qid_sha256'] != old_order_sha:
        raise ValueError('Original viewed-validation question identity/order SHA differs')
    final_plan = load_final_exclusion(final_materialization, final_materialization_sha)
    # Source input batches contain prompt/target data, never generated ownEM.
    # Do not read bank/groups.json, old prediction files, or rollout labels.
    input_manifest_path = training/'inputs/manifest.json'
    input_manifest = read(input_manifest_path)
    train_and_diagnostic, seen, original_inputs = [], set(), []
    for item in input_manifest['batches']:
        if sha(item['path']) != item['sha256']:
            raise ValueError('Original training/diagnostic question input SHA differs')
        batch = read(item['path'])
        original_inputs.append(proof(item['path']))
        for row in batch['rows']:
            if row['qid'] in seen:
                raise ValueError('Original training/diagnostic source qid repeats')
            seen.add(row['qid'])
            train_and_diagnostic.append(dict(qid=row['qid'], question_hash=row['question_hash']))
    if len(train_and_diagnostic) != source['train_questions']+source['diagnostic_questions']:
        raise ValueError('Original training/512diagnostic question coverage differs')
    isolation = validate_isolation(rows, train_and_diagnostic, final_plan)
    files = ('validation_g10_materialize.py', 'confirmation_materialize.py', 'g10_grouping.py',
             'shard.py', 'sampling.py', 'native_capture.py', 'packed_features.py', 'runtime.sh')
    code_files = {'reproduction/operations/qwen7b_warmup/'+name:sha(code_root/'reproduction/operations/qwen7b_warmup'/name) for name in files}
    for name in ('reproduction/prefix_selection200/rollout.py', 'reproduction/prefix_selection200/interaction_protocol.py',
                 'reproduction/prefix_selection200/contract.py', 'reproduction/prefix_selection200/b_progressive.py',
                 'reproduction/prefix20_online/pool.py', 'reproduction/prefix20_online/rollout.py',
                 'search_g1/async_rollout_runtime.py', 'search_r1/llm_agent/generation.py'):
        code_files[name] = sha(code_root/name)
    run_ids = [uuid.uuid4().hex, uuid.uuid4().hex]
    attempts = ['recipe-validation-'+run for run in run_ids]
    planned = validation_plan_events(rows, run_ids, attempts, fixed_seed)
    destination.mkdir(parents=True, exist_ok=False)
    atomic(destination/'final-exclusion-plan.json', final_plan)
    atomic(destination/'inputs/questions.json', dict(role=ROLE, rows=rows,
        ordered_qid_sha256=digest([r['qid'] for r in rows]), original_source_questions=proof(old_selection/'inputs/questions.json'),
        old_generated_EM_or_probe_predictions_read=False, untouched_final=False))
    atomic(destination/'planned-g10-and-fixed-g5.json', planned)
    source_proofs = [proof(training/'config.json'), proof(old_selection/'config.json'),
        proof(old_selection/'inputs/questions.json'), proof(input_manifest_path), proof(destination/'final-exclusion-plan.json'),
        proof(destination/'inputs/questions.json'), proof(destination/'planned-g10-and-fixed-g5.json')]+original_inputs
    configs = []
    for event_index in range(2):
        event_root = destination/('event-%s' % ('A' if event_index == 0 else 'B'))
        entries = []
        for shard in range(4):
            assigned = rows[shard::4]
            for offset in range(0, len(assigned), 128):
                path = event_root/'inputs'/('shard-%02d' % shard)/('batch-%06d.json' % (offset//128))
                atomic(path, dict(shard=shard, split='recipe_selection_validation', rows=assigned[offset:offset+128], model_revision=REVISION))
                entries.append(dict(proof(path), questions=len(assigned[offset:offset+128]), shard=shard))
        atomic(event_root/'inputs/manifest.json', dict(role=ROLE, batches=entries, questions=2048, trajectories=10240,
                                                     new_event=True, ordered_qid_sha256=digest([r['qid'] for r in rows])))
        configuration = dict(copy.deepcopy(source), base=str(event_root), validation_role=ROLE,
            run_id=run_ids[event_index], sampling_attempt_id=attempts[event_index],
            actor_version='official7b-'+REVISION, actor_updates_authorized=False, actor_optimizer_updates=0,
            warmup_only=True, evaluation_only=True, diagnostic_only=False, fit_epochs=0, fitting_authorized=False,
            tuning_allowed=True, untouched_final=False, train_questions=0, train_trajectories=0,
            diagnostic_questions=0, diagnostic_trajectories=0, evaluation_questions=2048, evaluation_trajectories=10240,
            prefix_diversity={}, prefix_diversity_selection=False, sampling_shards=4, gpus_per_sampling_job=2,
            source_commit=source_commit, code_root=str(code_root), collector_code_files_sha256=code_files,
            actor_complete=proof(Path(source['model_dir'])/'complete.json'),
            immutable_preparation_sources=source_proofs+[proof(event_root/'inputs/manifest.json')]+entries,
            fixed_view_precommitted=True, fixed_view_seed=fixed_seed, event_index=event_index,
            actual_responses_generated=0, same_generation_protocol_as_training=True)
        configuration.pop('confirmation_role', None)
        atomic(event_root/'config.json', configuration)
        (event_root/'logs').mkdir()
        (event_root/'sampling.sbatch').write_text(render_validation_script(event_root, code_root))
        configs.append(proof(event_root/'config.json'))
    result = dict(status='RECIPE_SELECTION_VALIDATION_TWO_FRESH_G5_PLANS_PREPARED_NOT_GENERATED', role=ROLE,
        base=str(destination), source_commit=source_commit, viewed_validation_questions_reused=True,
        untouched_final=False, tuning_allowed=True, fitting_on_validation_allowed=False,
        old_generated_EM_or_probe_predictions_read=False, source_question_plan=proof(old_selection/'inputs/questions.json'),
        question_plan=proof(destination/'inputs/questions.json'), fixed_G5_plan=proof(destination/'planned-g10-and-fixed-g5.json'),
        event_configs=configs, final_exclusion_plan=proof(destination/'final-exclusion-plan.json'), isolation=isolation,
        questions=2048, planned_real_trajectories=20480, actual_responses_generated=0, jobs_submitted=False,
        runtime_qualified=False, remaining='Generate and fresh-verify both new G5 events, then bind actual G10/fixedG5; no response/metric is created by preparation.', time=time.time())
    atomic(destination/'materialization.json', result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--training-base')
    p.add_argument('--old-selection-base')
    p.add_argument('--final-materialization')
    p.add_argument('--final-materialization-sha256')
    p.add_argument('--base')
    p.add_argument('--code-root')
    p.add_argument('--source-commit')
    p.add_argument('--fixed-seed', type=int, default=20261021)
    p.add_argument('--check-event-config')
    p.add_argument('--export-final-exclusion')
    a = p.parse_args()
    if a.check_event_config:
        result = validation_preflight(a.check_event_config)
    elif a.export_final_exclusion:
        if not a.final_materialization:
            p.error('Final-exclusion export requires --final-materialization')
        result = write_final_exclusion(a.final_materialization, a.export_final_exclusion,
                                      a.final_materialization_sha256)
    else:
        if not all((a.training_base, a.old_selection_base, a.final_materialization, a.base, a.code_root, a.source_commit)):
            p.error('Validation preparation requires training/old-selection/final-materialization/base/code-root/source-commit')
        result = prepare(a.training_base, a.old_selection_base, a.final_materialization, a.base,
                         a.code_root, a.source_commit, a.fixed_seed, a.final_materialization_sha256)
    print(json.dumps(result), flush=True)
