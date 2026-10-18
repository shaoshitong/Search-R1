#!/usr/bin/env python3
"""Read-only integrity/identity and stored strict-score aggregation on HPC3.

Run with the existing remote environment. Full responses stay remote. This
does not generate answers, change scoring, or claim a new model evaluation.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import sys
import time

FAMILY = Path('/data/user/jxu957/experiments/dynrpo-prefix5-thread5-20261017')
VALIDATORS = Path(str(FAMILY) + '-r3')
RUNTIME = '89c97ff2a5a60c7c3d6e1b0bf825d122f0ff874c'
RUN_ID = 'cc0de51f2d68495c9664aa888074839c'


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 ** 2), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default=str(VALIDATORS))
    parser.add_argument('--outer', type=int, choices=(5, 50, 100, 150, 200), required=True)
    args = parser.parse_args()
    base = Path(args.base)
    assert (base.parent == FAMILY.parent and base.resolve(strict=True) == base
            and re.fullmatch(re.escape(FAMILY.name) + r'(?:-r[0-9]+)?', base.name))
    assert read(VALIDATORS / 'preparation/source-publication.json')['commit'] == RUNTIME
    root = base / 'outers' / ('outer-%06d' % args.outer)
    target = root / 'validation/complete.json'
    if not target.is_file():
        print(json.dumps(dict(status='PENDING_NO_COMPLETE_FIXED512_QA', outer=args.outer)))
        return
    sys.path.insert(0, str(VALIDATORS / 'repo'))
    import os
    from reproduction.prefix_selection200.progressive_prefix import MODE,after_boundary
    os.environ['DYNRPO_PROGRESSIVE_PREFIX']=MODE
    os.environ['DYNRPO_PREFIX_TOKENS']=str(after_boundary(args.outer))
    from reproduction.prefix_selection200.eval import load_questions, CONTRACT
    from reproduction.prefix_selection200.interaction_protocol import CONTRACT as PROTOCOL
    config = read(base / 'preparation/launch-config.json')
    joint = read(root / 'complete.json')
    result = read(target)
    assert config['run_id'] == joint['run_id'] == result['run_id'] == RUN_ID
    assert joint['outer'] == args.outer and result['actor_version'] == joint['actor']['actor_version']
    assert result['actor_model_dir'] == joint['actor']['hf_model_dir']
    questions_sha = sha(config['validation_questions'])
    assert questions_sha == config['validation_questions_sha256'] == result['question_source_sha256']
    questions = load_questions(config['validation_questions'], 512)
    assert result['evaluation_contract'] == CONTRACT and result['interaction_protocol'] == PROTOCOL
    assert result['actual_compute_dtype'] == 'torch.bfloat16'
    assert len(result['rank_receipts']) == len(result['output_shards']) == 4
    files, rows, rank_seconds = [], [], []

    def verify(path, expected):
        path = Path(path)
        assert path.resolve(strict=True) == path and path.parent == target.parent
        digest = sha(path)
        assert digest == expected['sha256']
        if 'bytes' in expected:
            assert path.stat().st_size == expected['bytes']
        files.append(dict(path=str(path), bytes=path.stat().st_size, sha256=digest))

    for rank, expected in enumerate(result['rank_receipts']):
        path = target.parent / ('rank-%d-complete.json' % rank)
        assert expected['path'] == str(path)
        verify(path, expected)
        receipt = read(path)
        assert receipt['rank'] == rank and receipt['qa_count'] == receipt['question_count'] == 128
        for key in ('run_id', 'actor_version', 'actor_model_dir', 'question_source_sha256',
                    'evaluation_contract', 'interaction_protocol', 'actual_compute_dtype'):
            assert receipt[key] == result[key], key
        full = target.parent / ('rank-%d-full.jsonl.gz' % rank)
        assert receipt['full_output'] == result['output_shards'][rank]
        assert receipt['full_output']['path'] == str(full)
        verify(full, receipt['full_output'])
        with gzip.open(full, 'rt') as handle:
            full_rows = [json.loads(line) for line in handle if line.strip()]
        full_rows.sort(key=lambda row: row['evaluation_ordinal'])
        assert len(full_rows) == len(receipt['rows']) == 128
        assert [row['evaluation_ordinal'] for row in full_rows] == list(range(rank, 512, 4))
        for saved, full_row in zip(receipt['rows'], full_rows):
            assert all(full_row[key] == value for key, value in saved.items())
            canonical = questions[saved['evaluation_ordinal']]
            assert saved['qid'] == canonical['qid'] and saved['data_source'] == canonical['data_source']
            assert type(saved['em']) is int and saved['em'] in (0, 1)
            assert math.isfinite(saved['f1']) and 0 <= saved['f1'] <= 1
        rows.extend(receipt['rows'])
        rank_seconds.append(receipt['elapsed_seconds'])
    assert len({row['qid'] for row in rows}) == 512
    assert Counter(row['data_source'] for row in rows) == dict(nq=256, hotpotqa=256)
    assert result['qa_count'] == 512
    metrics = dict(em=sum(row['em'] for row in rows) / 512,
                   f1=sum(row['f1'] for row in rows) / 512)
    for key, value in metrics.items():
        assert math.isclose(result[key], value, rel_tol=0, abs_tol=1e-12)
    per_source = {}
    for source in ('nq', 'hotpotqa'):
        subset = [row for row in rows if row['data_source'] == source]
        measured = dict(qa_count=256, em=sum(row['em'] for row in subset) / 256,
                        f1=sum(row['f1'] for row in subset) / 256)
        assert result['per_source'][source]['qa_count'] == 256
        for key in ('em', 'f1'):
            assert math.isclose(result['per_source'][source][key], measured[key], rel_tol=0, abs_tol=1e-12)
        per_source[source] = measured
    assert all(value == 0 for value in result['actual_worker_exit_codes'].values())
    phase = read(base / 'logs/phase-timings.json').get('outer-%06d-validation' % args.outer)
    pending_path = base / 'logs/qa-pending.json'
    pending = read(pending_path) if pending_path.is_file() else {}
    print(json.dumps(dict(status='FIXED512_QA_FRESH_SHA_IDENTITY_AND_STORED_STRICT_AGGREGATION_PASS',
        epoch=time.time(), base=str(base), outer=args.outer, run_id=RUN_ID,
        actor_version=result['actor_version'], question_source_sha256=questions_sha,
        complete_sha256=sha(target), verified_files=files, qa_count=512,
        strict_correct=sum(row['em'] for row in rows), **metrics, per_source=per_source,
        actual_compute_dtype=result['actual_compute_dtype'], rank_inference_seconds=rank_seconds,
        phase_including_initialization=phase, actual_worker_exit_codes=result['actual_worker_exit_codes'],
        qa_pending_entry=pending.get(str(args.outer)), model_evaluation_rerun=False,
        scope='fresh file SHA, canonical identities and stored strict-score reaggregation; no new generation or token rescoring'), allow_nan=False))


if __name__ == '__main__':
    main()
