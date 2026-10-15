"""Exact ordered shared stage/dev/final evaluation; no resampling or dedup."""
import copy
import json
import math
import os
import time
from collections import Counter
from pathlib import Path
from reproduction.common8.verify_data import digest, probe_question_id, COUNTS, MANIFEST_SHA

_ACTIVE_SPLIT = 'stage'
_ACTIVE_SOURCE_COUNTS = None


def expected(base, split):
    import pyarrow.parquet as pq
    manifest_path = base / 'data/manifest.json'
    assert digest(manifest_path) == MANIFEST_SHA
    record = json.loads(manifest_path.read_text())['splits'][split]
    path = base / 'data' / record['path']
    assert digest(path) == record['sha256']
    frame = pq.read_table(path).to_pandas()
    assert len(frame) == COUNTS[split] == record['rows']
    return record, frame, [probe_question_id(x) for x in frame['prompt']]


def bind_validation_dataframe(frame, base, split='stage'):
    record, original, ids = expected(base, split)
    assert [probe_question_id(x) for x in frame['prompt']] == ids
    assert frame['data_source'].tolist() == original['data_source'].tolist()
    assert dict(Counter(frame['data_source'])) == record['source_counts']
    result = frame.copy()
    result['__g1_question_id'] = ids
    return result


def configure_validation(trainer, final=False):
    from torch.utils.data import DataLoader
    from verl.utils.dataset.rl_dataset import collate_fn
    global _ACTIVE_SPLIT, _ACTIVE_SOURCE_COUNTS
    step = int(trainer.global_steps)
    from reproduction.common8.calibration_runtime import active_plan
    phase = active_plan()
    if phase is not None:
        assert not final and step in (0, 1), 'A calibration child has only paired fixed512 evaluations'
        split = 'validation'
    else:
        split = 'final' if final else ('validation' if step > 0 and step % 50 == 0 else 'stage')
    base = Path(os.environ['SEARCH_G1_BASE'])
    record, frame, _ = expected(base, split)
    dataset = copy.copy(trainer.val_dataset)
    dataset.dataframe = bind_validation_dataframe(frame, base, split)
    trainer.val_dataloader = DataLoader(dataset, batch_size=64, shuffle=False,
        drop_last=False, collate_fn=collate_fn)
    _ACTIVE_SPLIT = split
    _ACTIVE_SOURCE_COUNTS = dict(record['source_counts'])
    print('SEARCH_G1_EVALUATION_SPLIT=' + split, flush=True)


def rows_from_validation_batch(batch, tokenizer):
    from reproduction.common8.calibration_evaluation import rows_from_batch
    assert _ACTIVE_SOURCE_COUNTS, 'Exact shared validation split must be bound before scoring'
    return rows_from_batch(batch, tokenizer, allowed_sources=_ACTIVE_SOURCE_COUNTS)


def record_validation(base, step, ids, sources, rewards, final=False, *, f1=None):
    split = _ACTIVE_SPLIT
    assert final == (split == 'final')
    record, frame, expected_ids = expected(base, split)
    ids, sources, scores = list(ids), list(sources), [float(x) for x in rewards]
    assert ids == expected_ids and len(scores) == COUNTS[split]
    assert sources == frame['data_source'].tolist()
    assert all(math.isfinite(x) and x in (0., 1.) for x in scores)
    assert f1 is not None, 'Actual generated-answer F1 is required; EM is not F1'
    f1 = [float(x) for x in f1]
    assert len(f1)==len(scores) and all(math.isfinite(x) and 0<=x<=1 for x in f1)
    assert all(em<=score for em,score in zip(scores,f1)), 'EM and alias-normalized F1 disagree'
    counts = dict(Counter(sources))
    assert counts == record['source_counts']
    run = json.loads((base / 'logs/run-manifest.json').read_text())
    proof = dict(run_id=run['run_id'], split=split, step=int(step), final=final,
        question_ids=ids, data_sources=sources, em=scores, counts=counts,
        f1=f1, weighted_f1=math.fsum(f1)/len(f1),
        correct=sum(scores), weighted_em=sum(scores)/len(scores),
        validation_sha256=record['sha256'], checked_unix=time.time())
    if (run.get('question_probe',{}).get('head_kind')=='qwen_native_periodic_lastblock_binary_v1' and
            split=='validation' and int(step) in (5,50,100,150,200)):
        from search_g1.qprobe_runtime import _write_json
        _write_json(base/'logs'/('validation-step'+str(int(step))+'.json'),proof)
    # Repeated normalized questions in the canonical final product are retained.
    filename = 'final-identity.json' if final else 'latest-' + split + '-identity.json'
    path = base / 'logs' / filename
    if final:
        with path.open('x') as f:
            json.dump(proof, f,allow_nan=False);f.flush();os.fsync(f.fileno())
    else:
        temp = path.with_suffix('.tmp')
        with temp.open('w') as f:
            json.dump(proof,f,allow_nan=False);f.flush();os.fsync(f.fileno())
        temp.replace(path)
    parent_fd=os.open(path.parent,os.O_RDONLY|getattr(os,'O_DIRECTORY',0))
    try:os.fsync(parent_fd)
    finally:os.close(parent_fd)
    ledger=base/'logs/validation-identities.jsonl'
    with ledger.open('a') as f:
        f.write(json.dumps(proof,allow_nan=False) + '\n');f.flush();os.fsync(f.fileno())
    parent_fd=os.open(ledger.parent,os.O_RDONLY|getattr(os,'O_DIRECTORY',0))
    try:os.fsync(parent_fd)
    finally:os.close(parent_fd)
    return proof
