"""Read-only remote CPU recomputation of completed milestone evaluation rows.

Does not import Torch, load a checkpoint, allocate a GPU, or run retrieval.
Only a small score/provenance receipt is returned; raw rows remain remote.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-milestone100-150-native2-eval-20261025')
COMMIT = 'd5c8a152b709805ad36946759d60023191bad3d9'
BASE = Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fa3-batch-searchstop-r1-20261023')


def read(path):
    raw = path.read_bytes()
    return json.loads(raw), dict(path=str(path), bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())


def main():
    sys.path.insert(0, str(ROOT / 'source' / COMMIT))
    from reproduction.easy2560_joint4.final_cpu_acceptance import canonical, scores
    identity = canonical(BASE, 'final')
    results = {}
    for step in (100, 150):
        directory = ROOT / ('eval-' + str(step))
        if not (directory / 'result.json').is_file():
            continue
        result, result_proof = read(directory / 'result.json')
        launch, launch_proof = read(directory / 'launch.json')
        workers, worker_proof = read(directory / 'workers.json')
        assert launch['source_run_id'] == '232f20e112ac4a978c5aba52a0367195'
        assert result['evaluation_commit'] == launch['evaluation_commit'] == COMMIT
        assert result['training_commit'] == launch['source_training_commit'] == 'dde1452d2ec65fb3f4a0464d4beacf3b8136414c'
        assert result['job'] == launch['job'] == '760313'
        assert result['step'] == step and result['split'] == 'final' and result['world_size'] == 2
        assert result['greedy'] and not result['training_modified'] and not result['optimizer_loaded']
        assert result['optimizer_writes'] == 0
        assert len(workers) == 2 and {x['rank'] for x in workers} == {0, 1}
        assert all(x['checkpoint'] == launch['checkpoint'] == result['checkpoint'] and
                   not x['optimizer_loaded'] and not x['actor_model_loaded'] for x in workers)
        rows_path = directory / 'rows.jsonl'
        before = rows_path.stat()
        rows, h = [], hashlib.sha256()
        with rows_path.open('rb') as stream:
            for line in stream:
                assert line.endswith(b'\n')
                h.update(line); rows.append(json.loads(line))
        after = rows_path.stat()
        assert (before.st_ino, before.st_size, before.st_mtime_ns) == (after.st_ino, after.st_size, after.st_mtime_ns)
        ids = [x['question_id'] for x in rows]
        sources = [x['data_source'] for x in rows]
        em, f1 = [x['em'] for x in rows], [x['f1'] for x in rows]
        value = dict(run_id=launch['run_id'], step=step, split='final', final=True,
            question_ids=ids, data_sources=sources, em=em, f1=f1,
            validation_sha256=result['validation_sha256'], counts=dict(Counter(sources)),
            correct=result['correct'], weighted_em=result['weighted_em'], weighted_f1=result['weighted_f1'])
        scored = scores(value, identity, {'run_id': launch['run_id']}, step, split='final', final=True)
        for key in ('question_count', 'correct', 'weighted_em', 'weighted_f1', 'macro_em', 'macro_f1', 'per_source'):
            assert scored[key] == result[key], key
        ids_sha = hashlib.sha256(json.dumps(ids, sort_keys=True, ensure_ascii=False,
            separators=(',', ':'), allow_nan=False).encode()).hexdigest()
        assert ids_sha == result['ordered_question_ids_sha256']
        assert len(list((directory / 'raw').glob('batch-*.pt'))) == 809
        results[str(step)] = dict(status='INDEPENDENT_CPU_ORDERED51713_SCORES_VERIFIED',
            scores=scored, validation_sha256=identity['sha256'], ordered_question_ids_sha256=ids_sha,
            result=result_proof, launch=launch_proof, workers=worker_proof,
            rows=dict(path=str(rows_path), bytes=before.st_size, sha256=h.hexdigest(), rows=len(rows)),
            raw_batches=809, last_batch_expected_questions=1, raw_tensors_loaded=False,
            retrieval_queries_sent=0, model_loaded=False, gpu_allocated=False,
            torch_imported='torch' in sys.modules, training_modified=False)
    assert 'torch' not in sys.modules
    print(json.dumps(dict(checked_unix=time.time(), root=str(ROOT), completed_steps=results,
        both_scores_verified=set(results) == {'100', '150'}, training_modified=False)))


if __name__ == '__main__':
    main()
