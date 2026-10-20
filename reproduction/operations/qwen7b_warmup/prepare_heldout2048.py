"""Freeze unseen NQ-dev2048 before labels or checkpoint scores are observed.

Run on HPC3 with the qualified warmup runtime. The original training config,
bank, model, all four checkpoints and reserved final set remain untouched.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import random
import shlex
import time
import unicodedata
import uuid


SOURCE_SHA = '10d5ee25e813f3e2092247a5d2797793dd47ef5853959989cb34bf80a110d607'
REVISION = 'a09a35458c702b33eeacc393d103063234e8bc28'


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2))
    temporary.replace(path)


def normalized(question):
    return ' '.join(unicodedata.normalize('NFKC', question).casefold().strip().rstrip('?').split())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--training-base', required=True, type=Path)
    parser.add_argument('--base', required=True, type=Path)
    parser.add_argument('--source-commit', required=True)
    args = parser.parse_args()
    base, training = args.base, args.training_base
    assert base == training / 'selection-2048-20261020'
    config = json.loads((training / 'config.json').read_text())
    assert config['model_revision'] == REVISION and config['question_max_tokens'] == 273
    request_path = training / 'control/human-epoch4-selection-request.json'
    request = json.loads(request_path.read_text())
    assert request['enabled'] and request['stop_after_epoch'] == 4 and request['exclude_all_warmup_questions']
    source = base / 'source/nq-dev-bcafb8dd07d4.jsonl'
    assert source.stat().st_size == 1073443 and sha(source) == SOURCE_SHA
    if (base / 'config.json').exists():
        existing = json.loads((base / 'config.json').read_text())
        assert existing['heldout_source_sha256'] == SOURCE_SHA and existing['evaluation_questions'] == 2048
        assert existing['human_request_sha256'] == sha(request_path)
        print('EXISTING_IMMUTABLE_HELDOUT2048_PLAN_NOT_REPARTITIONED')
        return
    from transformers import AutoTokenizer
    import pyarrow.parquet as pq
    from reproduction.prefix20_online.pool import identity
    from reproduction.prefix20_online.rollout import prompt_tokens
    common = Path(config['canonical_train']).parent
    manifest = json.loads((common / 'manifest.json').read_text())
    template = next(pq.ParquetFile(str(common / 'train.parquet')).iter_batches(batch_size=1)).to_pylist()[0]
    first_question = json.loads((common / 'train.jsonl').open().readline())['question']
    assert set(template['reward_model']['ground_truth']) == {'target'}
    messages = []
    for message in template['prompt']:
        assert message['content'].count(first_question) == 1
        messages.append(dict(message, content=message['content'].replace(first_question, '<QUESTION>')))
    excluded_ids, excluded_qids, excluded_text, counts, exclusion_sha = set(), set(), set(), {}, {}
    for split in ('train', 'validation', 'stage', 'final'):
        parquet = common / (split + '.parquet')
        assert sha(parquet) == manifest['splits'][split]['sha256']
        rows = [json.loads(line) for line in (common / (split + '.jsonl')).open()]
        assert len(rows) == manifest['splits'][split]['rows']
        counts[split] = len(rows)
        exclusion_sha[split] = sha(common / (split + '.jsonl'))
        for row in rows:
            excluded_ids.add(str(row['question_id']))
            excluded_text.add(normalized(row['question']))
        ordinal = 0
        for batch in pq.ParquetFile(str(parquet)).iter_batches(batch_size=2048):
            for own in batch.to_pylist():
                raw_question = rows[ordinal]['question']
                expected_prompt = [dict(m, content=m['content'].replace('<QUESTION>', raw_question)) for m in messages]
                assert own['prompt'] == expected_prompt, 'Parquet/JSONL question identity mismatch'
                excluded_qids.add(identity(own)['qid'])
                ordinal += 1
        assert ordinal == len(rows)
    tokenizer = AutoTokenizer.from_pretrained(config['model_dir'], local_files_only=True, trust_remote_code=False)
    raw = [json.loads(line) for line in source.open()]
    assert len(raw) == 8757
    available, seen, removed = [], set(), dict(existing_id=0, existing_qid=0, existing_text=0, internal_duplicate=0, overlong_prompt=0)
    for ordinal, own in enumerate(raw):
        question = own['question']
        key = 'nq:dev:' + str(own['id'])
        text = normalized(question)
        if key in excluded_ids:
            removed['existing_id'] += 1
            continue
        if text in excluded_text:
            removed['existing_text'] += 1
            continue
        if text in seen:
            removed['internal_duplicate'] += 1
            continue
        seen.add(text)
        answers = own['golden_answers']
        assert isinstance(answers, list) and answers and all(isinstance(a, str) for a in answers)
        row = dict(data_source='nq', prompt=[dict(m, content=m['content'].replace('<QUESTION>', question)) for m in messages],
                   ability=template['ability'], reward_model=dict(style=template['reward_model']['style'], ground_truth=dict(target=answers)),
                   extra_info=dict(index=key, split='unseen_selection_validation'))
        if identity(row)['qid'] in excluded_qids:
            removed['existing_qid'] += 1
            continue
        tokens = prompt_tokens(tokenizer, row['prompt'])
        if len(tokens) > config['question_max_tokens']:
            removed['overlong_prompt'] += 1
            continue
        assert len(tokens) > 0 and text not in excluded_text and key not in excluded_ids
        row.update(identity(row), split='selection_validation', source_ordinal=ordinal,
                   initial_prompt_token_ids=tokens, source_question_id=key,
                   canonical_question_sha256=hashlib.sha256(text.encode()).hexdigest())
        available.append(row)
    assert len(available) >= 2048, (len(available), removed)
    random.Random(20261020).shuffle(available)
    selected = available[:2048]
    assert len({r['qid'] for r in selected}) == len({r['canonical_question_sha256'] for r in selected}) == 2048
    assert not {r['qid'] for r in selected}.intersection(excluded_qids)
    for index, row in enumerate(selected):
        row['candidate_ordinal'] = index
    new = dict(config, base=str(base), run_id=uuid.uuid4().hex, seed=20261020,
               warmup_only=True, evaluation_only=True, actor_updates_authorized=False,
               actor_optimizer_updates=0, train_questions=0, train_trajectories=0,
               diagnostic_questions=0, diagnostic_trajectories=0, fit_epochs=0,
               evaluation_questions=2048, evaluation_trajectories=10240,
               sampling_shards=4, sampling_partition='acd_u', selected_questions=512,
               training_base=str(training), training_config_sha256=sha(training / 'config.json'),
               human_request_sha256=sha(request_path), heldout_source_sha256=SOURCE_SHA,
               plan_source_commit=args.source_commit, sampling_source_commit='96df5a23f23f6da0f24e79e67dcf0e3039f22de0',
               selection_rule='top512_by_EMA_predicted_Mixed_probability_no_EM',
               dataset_role='checkpoint_selection_validation_not_final_QA')
    entries = []
    for shard in range(4):
        directory = base / 'inputs' / ('shard-%02d' % shard)
        assigned = selected[shard::4]
        assert len(assigned) == 512
        for start in range(0, 512, 128):
            path = directory / ('batch-%06d.json' % (start // 128))
            write(path, dict(shard=shard, split='selection_validation', rows=assigned[start:start + 128], model_revision=REVISION))
            entries.append(dict(path=str(path), sha256=sha(path), questions=128, shard=shard))
        write(directory / 'partition.json', dict(shard=shard, counts=dict(selection_validation=512), batches=4))
    ordered_sha = hashlib.sha256(json.dumps([r['qid'] for r in selected]).encode()).hexdigest()
    write(base / 'inputs/questions.json', dict(rows=selected, ordered_qid_sha256=ordered_sha))
    write(base / 'inputs/manifest.json', dict(batches=entries, questions=2048, trajectories=10240,
        question_max_tokens=273, model_revision=REVISION, ordered_qid_sha256=ordered_sha,
        dataset_revision='bcafb8dd07d453be3cbeeeb3f78be1841bddf92c', source_sha256=SOURCE_SHA))
    write(base / 'config.json', new)
    (base / 'logs').mkdir(exist_ok=True)
    runtime = training / 'repo-postprocess-96df5a23f23f/reproduction/operations/qwen7b_warmup/runtime.sh'
    script = '''#!/usr/bin/env bash
#SBATCH --job-name=q7b-unseen2048
#SBATCH --partition=acd_u
#SBATCH --account=mou176
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH --time=08:00:00
#SBATCH --array=0-3
#SBATCH --output=BASE/logs/sampling-%A_%a.log
set -euo pipefail
export QWEN7B_CODE_ROOT=TRAINING/repo-postprocess-96df5a23f23f
srun --kill-on-bad-exit=1 bash RUNTIME -u -m reproduction.operations.qwen7b_warmup.shard --config BASE/config.json --shard "$SLURM_ARRAY_TASK_ID"
'''.replace('BASE', str(base)).replace('TRAINING', str(training)).replace('RUNTIME', shlex.quote(str(runtime)))
    (base / 'sampling.sbatch').write_text(script)
    acceptance = dict(status='FROZEN_UNSEEN_2048_G5_QUESTION_PLAN_VERIFIED', original_rows=8757,
        eligible_unique_rows=len(available), removed=removed, excluded_sets=counts,
        excluded_jsonl_sha256=exclusion_sha, selected_questions=2048, trajectories=10240,
        source_sha256=SOURCE_SHA, ordered_qid_sha256=ordered_sha,
        selection_before_any_generated_EM=True, no_final_QA_rows_used=True,
        normalized_identity='NFKC_casefold_trim_questionmark_collapse_whitespace',
        source_commit=args.source_commit, epoch_time=time.time())
    write(base / 'preparation/question-plan-acceptance.json', acceptance)
    print(json.dumps(acceptance), flush=True)


if __name__ == '__main__':
    main()
