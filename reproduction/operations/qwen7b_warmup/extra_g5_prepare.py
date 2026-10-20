"""Freeze an isolated extra-real-G5 pilot; never submit or alter old outputs.

Input questions are chosen by identity hash, not EM/probe scores. Existing five
answers remain untouched; new generation uses response indices5..9 and its own
run/attempt/trajectory seed namespace. All payloads remain on remote storage.
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import shlex
import time
import uuid


REVISION = 'a09a35458c702b33eeacc393d103063234e8bc28'
ROLE = 'ISOLATED_OFFICIAL7B_EXTRA_REAL_G5_FOR_TRAINING_G10'


def render_extra_sbatch(base, code_root):
    package=Path('reproduction/operations/qwen7b_warmup')
    return '''#!/usr/bin/env bash
#SBATCH --job-name=q7b-extra-realG5
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
#SBATCH --output=OUTPUT/logs/sampling-%A_%a.log
set -euo pipefail
export QWEN7B_CODE_ROOT=__CODE_ROOT__
srun --kill-on-bad-exit=1 bash RUNTIME -u -m reproduction.operations.qwen7b_warmup.extra_g5_sampler --config CONFIG --shard "$SLURM_ARRAY_TASK_ID"
'''.replace('OUTPUT', str(base)).replace('__CODE_ROOT__', shlex.quote(str(code_root))).replace('RUNTIME', shlex.quote(str(code_root/package/'runtime.sh'))).replace('CONFIG', shlex.quote(str(base/'config.json')))


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    os.replace(str(temporary), str(path))


def file_proof(path):
    path = Path(path)
    return dict(path=str(path), bytes=path.stat().st_size, sha256=sha(path))


def remote(path):
    path = Path(path).resolve()
    if not str(path).startswith('/data/user/'):
        raise ValueError('Extra rollout preparation/data must stay on remote shared storage')
    return path


def validate_config(c):
    if c.get('role') != ROLE or c.get('model_revision') != REVISION:
        raise ValueError('Not an isolated official7B extra-G5 phase')
    if c['base'] == c['original_base'] or c['run_id'] == c['original_run_id']:
        raise ValueError('Extra phase must not reuse original output/run identity')
    if not c.get('sampling_attempt_id') or c['sampling_attempt_id'] == c.get('original_sampling_attempt_id'):
        raise ValueError('Distinct explicit extra sampling attempt identity required')
    offset = c.get('extra_response_offset', 5)
    if offset not in (5, 10, 15) or c.get('extra_response_indices') != list(range(offset, offset+5)):
        raise ValueError('Five extra real response slots require offset5,10,or15')
    if not c.get('warmup_only') or c.get('actor_updates_authorized') is not False or c.get('actor_optimizer_updates') != 0:
        raise ValueError('Actor updates are forbidden in extra generation')
    if c.get('capture_prefix_tokens') != 20 or c.get('feature_layer') != 26:
        raise ValueError('Extra native capture must retain the original20/layer26 contract')
    if c.get('question_max_tokens') != 273 or c.get('feature_dimension') != 3584:
        raise ValueError('Official7B original representation geometry changed')
    if c.get('extra_shards') != 4 or c.get('gpus_per_sampling_job') != 2:
        raise ValueError('Pilot recipe is four independently admitted two-card shards')
    if c.get('evaluation_only') or c.get('prefix_diversity_selection') or c.get('prefix_diversity'):
        raise ValueError('Training-only extra-G5 cannot run final evaluation or prefix resampling')
    return c


def extra_initial_state(c, outer, group, local_index, tokenizer, factory=None):
    """Strict pure state adapter; shifts only response index, not rollout logic."""
    validate_config(c)
    if type(local_index) is not int or not 0 <= local_index < 5:
        raise ValueError('Extra local response slot must be0..4')
    if outer != 1 or group.get('split') != 'train':
        raise ValueError('Only declared training questions at isolated sampling outer1')
    if factory is None:
        from reproduction.prefix_selection200.rollout import initial_state
        factory = initial_state
    response_index = local_index+c.get('extra_response_offset', 5)
    value = factory(c, outer, group, response_index, tokenizer)
    if value['response_index'] != response_index or value['run_id'] != c['run_id']:
        raise ValueError('Underlying collector did not preserve extra response identity')
    if value['actor_version'] != 'official7b-'+REVISION or value.get('sampling_attempt_id') != c['sampling_attempt_id']:
        raise ValueError('Underlying state lost official actor/attempt identity')
    if value['trajectory_id'] in group.get('original_trajectory_ids', []):
        raise ValueError('Extra trajectory ID collides with retained original G5')
    value['extra_generation_role'] = ROLE
    return value


def choose_questions(train, diagnostic, heldout, final_questions, budget, seed, ordered_qids=None):
    """Label-blind question selection with explicit exclusion and identity checks."""
    if type(budget) is not int or budget < 4 or budget > len(train):
        raise ValueError('Pilot requires4..len(train) distinct training questions')
    sources = [train, diagnostic, heldout, final_questions]
    for rows in sources:
        if len({r['qid'] for r in rows}) != len(rows):
            raise ValueError('Duplicate question identity in a declared source split')
    excluded = {r['qid'] for rows in sources[1:] for r in rows}
    excluded_hashes = {r['question_hash'] for rows in sources[1:] for r in rows}
    eligible = [r for r in train if r['qid'] not in excluded and r['question_hash'] not in excluded_hashes]
    if len(eligible) < budget:
        raise ValueError('Insufficient training-only questions after validation/final exclusions')
    if ordered_qids is None:
        selected = sorted(eligible, key=lambda r: (canonical_sha(['extra-g5-question-plan', seed, r['qid']]), r['qid']))[:budget]
    else:
        if (not isinstance(ordered_qids,list) or len(ordered_qids)!=budget or
                any(not isinstance(q,str) or not q for q in ordered_qids) or
                len(set(ordered_qids))!=budget):
            raise ValueError('Matched cohort must declare exactly budget distinct question IDs')
        lookup={r['qid']:r for r in eligible}
        if not set(ordered_qids)<=set(lookup):
            raise ValueError('Matched cohort contains excluded or nontraining questions')
        selected=[lookup[q] for q in ordered_qids]
    for row in selected:
        if len(row['trajectory_ids']) != 5 or len(set(row['trajectory_ids'])) != 5:
            raise ValueError('Original G5 is incomplete')
        if len(row['labels']) != 5 or any(type(y) is not int or y not in (0, 1) for y in row['labels']):
            raise ValueError('Original own EM labels are incomplete')
    return selected


def prepare(original, selection_base, final_plan, base, code_root, source_commit, budget, seed, response_offset=5,
            cohort_plan=None,cohort_plan_sha256=None):
    original, selection_base, base, code_root = map(remote, (original, selection_base, base, code_root))
    final_plan = remote(final_plan)
    if base.exists():
        raise ValueError('Extra phase output already exists; inspect identity/locks/receipts, never reprepare over it')
    if len(source_commit) != 40 or any(c not in '0123456789abcdef' for c in source_commit):
        raise ValueError('Exact published source commit required')
    c = read(original/'config.json')
    bank, bank_proof = read(original/'bank/groups.json'), read(original/'bank/complete.json')
    if c['model_revision'] != REVISION or bank_proof['model_revision'] != REVISION or bank_proof['actor_updates'] != 0:
        raise ValueError('Original source is not frozen official initial7B')
    if sha(original/'bank/groups.json') != bank_proof['groups']['sha256']:
        raise ValueError('Original completeG5 bank SHA differs')
    selected_proof = read(selection_base/'bank/complete.json')
    if sha(selection_base/'bank/groups.json') != selected_proof['groups_sha256']:
        raise ValueError('Selection-validation bank SHA differs')
    heldout = read(selection_base/'bank/groups.json')['selection_validation']
    final = read(final_plan)
    if final.get('role') != 'final' or final.get('frozen') is not True or not final.get('questions'):
        raise ValueError('Explicit frozen final question-plan manifest required before extra training generation')
    if final.get('ordered_qid_sha256') != canonical_sha([r['qid'] for r in final['questions']]):
        raise ValueError('Final ordered question IDs do not match frozen manifest SHA')
    cohort_proof=None;ordered_qids=None
    if (cohort_plan is None)!=(cohort_plan_sha256 is None):
        raise ValueError('Matched cohort requires both an explicit plan and pinned SHA')
    if cohort_plan is not None:
        cohort_plan=remote(cohort_plan)
        if sha(cohort_plan)!=cohort_plan_sha256:
            raise ValueError('Predeclared matched cohort plan SHA differs')
        plan=read(cohort_plan);ordered_qids=plan['cohort_qids']
        if not isinstance(plan.get('batches'),list) or set(q for batch in plan['batches'] for q in batch)!=set(ordered_qids):
            raise ValueError('Matched cohort differs from its predeclared update batches')
        cohort_proof=file_proof(cohort_plan)
    chosen = choose_questions(bank['train'], bank['diagnostic'], heldout, final['questions'], budget, seed,ordered_qids)
    selection_rule=('exact predeclared matched-cohort IDs; no EM/probe scores' if cohort_proof else
                    'deterministic identity hash; no EM/probe scores')
    wanted = {r['qid']:r for r in chosen}
    actor_proof = Path(c['model_dir'])/'complete.json'
    actor = read(actor_proof)
    if actor['status'] != 'MIGRATION_ALL_SELECTED_ASSETS_VERIFIED' or actor['revision'] != REVISION or not actor['files']:
        raise ValueError('Official initial actor complete-file proof missing')
    publication_files = {}
    package = Path('reproduction/operations/qwen7b_warmup')
    for name in ('extra_g5_prepare.py', 'extra_g5_sampler.py', 'extra_g5_merge.py',
                 'shard.py', 'sampling.py', 'native_capture.py', 'packed_features.py', 'runtime.sh'):
        path = code_root/package/name
        if not path.is_file():
            raise ValueError('Prepared collector source missing: '+str(path))
        publication_files[str(package/name)] = sha(path)
    for relative in ('reproduction/prefix_selection200/rollout.py',
                     'reproduction/prefix_selection200/interaction_protocol.py',
                     'reproduction/prefix_selection200/contract.py',
                     'reproduction/prefix_selection200/b_progressive.py',
                     'reproduction/prefix_selection200/prefix_diversity.py',
                     'reproduction/prefix20_online/rollout.py', 'reproduction/prefix20_online/pool.py',
                     'search_g1/async_rollout_runtime.py', 'search_r1/llm_agent/generation.py'):
        publication_files[relative] = sha(code_root/relative)
    if response_offset not in (5, 10, 15):
        raise ValueError('Only independent extraG5 offsets5/10/15 are supported')
    found, inputs, original_batches, original_seeds, original_contract = {}, {}, [], {}, None
    manifest = read(original/'inputs/manifest.json')
    for item in manifest['batches']:
        if item['split'] != 'train':
            continue
        path = Path(item['path'])
        rows = read(path)['rows']
        accepted = [r for r in rows if r['qid'] in wanted]
        if not accepted:
            continue
        if sha(path) != item['sha256']:
            raise ValueError('Original tokenizer/prompt input batch SHA differs')
        source_receipt = original/'sampling'/('shard-%02d' % item['shard'])/'batches'/path.stem/'complete.json'
        receipt = read(source_receipt)
        if receipt['run_id'] != c['run_id'] or receipt['input_sha256'] != item['sha256'] or receipt['actor_optimizer_updates'] != 0:
            raise ValueError('Original batch identity/actor-update proof differs')
        original_batches.append(file_proof(source_receipt))
        prefix_proof = receipt['prefix']
        contract = {k:prefix_proof[k] for k in ('sampling_contract', 'action_boundary', 'interaction_protocol', 'actual_compute_dtype')}
        if original_contract is None:
            original_contract = contract
        elif original_contract != contract:
            raise ValueError('Original selected generation batches use different protocols')
        full_item = receipt['full']['full']
        if sha(full_item['path']) != full_item['sha256']:
            raise ValueError('Original genuine fullrollout payload SHA differs')
        original_batches.append(file_proof(full_item['path']))
        original_rows = {}
        with gzip.open(full_item['path'], 'rt') as f:
            for line in f:
                row = json.loads(line)
                if row['qid'] in wanted:
                    original_rows[row['trajectory_id']] = row
        for row in accepted:
            if row['qid'] in found or row['split'] != 'train' or row['question_hash'] != wanted[row['qid']]['question_hash']:
                raise ValueError('Pilot original question identity differs/repeats')
            if not row['initial_prompt_token_ids']:
                raise ValueError('Original actual prompt tokens missing')
            found[row['qid']] = dict(row, original_trajectory_ids=wanted[row['qid']]['trajectory_ids'])
            inputs[row['qid']] = file_proof(path)
            own_seeds = set()
            for tid, label in zip(wanted[row['qid']]['trajectory_ids'], wanted[row['qid']]['labels']):
                full_row = original_rows.get(tid)
                if full_row is None or full_row['strict_em'] != label or full_row['policy_snapshot_id'] != 'official7b-'+REVISION:
                    raise ValueError('Original bank label/actor differs from its own genuine full trajectory')
                own_seeds.update(chunk['seed'] for chunk in full_row['sampling_journal'])
            original_seeds[row['qid']] = sorted(own_seeds)
    if set(found) != set(wanted):
        raise ValueError('Not every pilot question has a sealed original input batch')
    new_run = uuid.uuid4().hex
    configuration = dict(c, role=ROLE, base=str(base), original_base=str(original),
        original_run_id=c['run_id'], original_sampling_attempt_id=c.get('sampling_attempt_id'),
        run_id=new_run, sampling_attempt_id='extra-real-g5-'+new_run, seed=seed,
        model_revision=REVISION, actor_version='official7b-'+REVISION,
        actor_model_dir=c['model_dir'], actor_updates_authorized=False, actor_optimizer_updates=0,
        warmup_only=True, evaluation_only=False, extra_response_offset=response_offset,
        extra_response_indices=list(range(response_offset, response_offset+5)),
        train_questions=budget, train_trajectories=5*budget, diagnostic_questions=0, diagnostic_trajectories=0,
        extra_shards=4, sampling_shards=4, gpus_per_sampling_job=2, fit_epochs=0,
        source_commit=source_commit, code_root=str(code_root), code_files_sha256=publication_files,
        final_plan=file_proof(final_plan), selection_base=str(selection_base),
        matched_cohort_plan=cohort_proof,question_selection=selection_rule,
        original_actor_complete=file_proof(actor_proof), full_actor_SHA_recheck_required_at_sampler_start=True,
        original_bank=file_proof(original/'bank/groups.json'),
        original_config=file_proof(original/'config.json'),
        original_bank_complete=file_proof(original/'bank/complete.json'),
        original_protocol_contract=original_contract,
        prefix_diversity_selection=False, prefix_diversity={}, fitting_authorized=False)
    validate_config(configuration)
    # Only now create an exclusive new output tree after all static checks.
    base.mkdir(parents=True, exist_ok=False)
    (base/'logs').mkdir()
    entries = []
    for shard in range(4):
        rows = [found[g['qid']] for g in chosen[shard::4]]
        for offset in range(0, len(rows), 128):
            path = base/'inputs'/('shard-%02d' % shard)/('batch-%06d.json' % (offset//128))
            atomic(path, dict(shard=shard, split='train', rows=rows[offset:offset+128], model_revision=REVISION, role=ROLE))
            entries.append(dict(file_proof(path), questions=len(rows[offset:offset+128]), shard=shard))
        atomic(base/'inputs'/('shard-%02d' % shard)/'partition.json', dict(shard=shard, questions=len(rows), trajectories=5*len(rows)))
    qids = [g['qid'] for g in chosen]
    atomic(base/'inputs/questions.json', dict(role=ROLE, groups=chosen, ordered_qid_sha256=canonical_sha(qids),
        final_plan=configuration['final_plan'], validation_qids_excluded=True, final_qids_excluded=True,
        question_selection=selection_rule,matched_cohort_plan=cohort_proof,original_batches=original_batches,
        original_native_seeds_by_qid=original_seeds,
        original_input_batches=list({r['path']:r for r in inputs.values()}.values())))
    atomic(base/'inputs/manifest.json', dict(role=ROLE, batches=entries, questions=budget, trajectories=5*budget,
        ordered_qid_sha256=canonical_sha(qids), run_id=new_run, response_indices=list(range(response_offset, response_offset+5)),
        originalG5_preserved=True, model_revision=REVISION))
    atomic(base/'config.json', configuration)
    script=render_extra_sbatch(base,code_root)
    (base/'sampling.sbatch').write_text(script)
    result = dict(status='EXTRA_REAL_G5_PILOT_PREPARED_NOT_SUBMITTED', role=ROLE,
        base=str(base), questions=budget, trajectories=5*budget, source_commit=source_commit,
        source_qid_plan_sha256=canonical_sha(qids), original_run_id=c['run_id'], extra_run_id=new_run,
        sampler_runtime_qualified=False, actual_extra_rollouts_generated=0, actor_updates=0,
        config=file_proof(base/'config.json'), input_manifest=file_proof(base/'inputs/manifest.json'),
        sampling_script=file_proof(base/'sampling.sbatch'), time=time.time())
    atomic(base/'prepared.json', result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--original-base', required=True)
    p.add_argument('--selection-base', required=True)
    p.add_argument('--final-plan', required=True)
    p.add_argument('--base', required=True)
    p.add_argument('--code-root', required=True)
    p.add_argument('--source-commit', required=True)
    p.add_argument('--questions', type=int, default=128)
    p.add_argument('--seed', type=int, default=20261021)
    p.add_argument('--response-offset', type=int, choices=(5, 10, 15), default=5)
    p.add_argument('--cohort-plan')
    p.add_argument('--cohort-plan-sha256')
    a = p.parse_args()
    print(json.dumps(prepare(a.original_base, a.selection_base, a.final_plan, a.base, a.code_root,
                             a.source_commit, a.questions, a.seed, a.response_offset,
                             a.cohort_plan,a.cohort_plan_sha256)), flush=True)
