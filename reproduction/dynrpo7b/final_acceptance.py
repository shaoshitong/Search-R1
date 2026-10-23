"""CPU acceptance of actual 200-outers, probe updates and complete QA outputs.

This reads committed producer SHA receipts and file metadata. It does not load
7B Adam tensors, repeat GPU work, or require continuous occupancy sampling.
"""
import argparse
import json
import math
from pathlib import Path

from .deploy import publish, sha
from .infra_profile import native4
from .prefix_schedule import DEFAULT_SCHEDULE
from .recipe import MILESTONES


def read(path):
    return json.loads(Path(path).read_text())


def rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def history(base):
    """Newest segment first, each parent cut at the actual restored boundary."""
    segments, seen, upper = [], set(), 200
    while True:
        base = Path(base).resolve()
        if base in seen:
            raise ValueError('Restore source chain contains a cycle')
        seen.add(base); segments.append((base, upper))
        run = read(base / 'logs/run-manifest.json')
        restored = run.get('full_state_restore')
        if not restored:
            return segments
        upper = min(upper, restored['completed_outer'])
        base = Path(restored['source_base'])


def historical_rows(segments, relative, key='step'):
    result = {}
    for directory, upper in reversed(segments):
        path = directory / relative
        if path.exists():
            for row in rows(path):
                if row[key] <= upper:
                    # A suffix is authoritative only within its own segment;
                    # parent rows beyond the restore point were excluded above.
                    result[(row[key], row.get('split'), row.get('kind'))] = row
    return sorted(result.values(), key=lambda row: row[key])


def historical_file(segments, step, relative):
    for directory, upper in segments:
        path = directory / relative
        if step <= upper and path.is_file():
            return path
    raise FileNotFoundError('Required historical artifact is missing: ' + str(relative))


def verify_evaluation(proof, run, ids, sources, *, step, split):
    if (proof['run_id'] != run['run_id'] or proof['step'] != step or
            proof['question_ids'] != ids or proof['data_sources'] != sources):
        raise ValueError('Actual evaluation question order/actor differs at outer ' + str(step))
    em, f1 = proof['em'], proof['f1']
    if (len(em) != len(ids) or len(f1) != len(ids) or
            any(v not in (0., 1.) for v in em) or
            any(not math.isfinite(v) or not 0 <= v <= 1 for v in f1) or
            any(a > b for a, b in zip(em, f1)) or proof['split'] != split):
        raise ValueError('Incomplete strict EM/F1 evaluation at outer ' + str(step))
    return dict(questions=len(ids), correct=sum(em), EM=sum(em) / len(ids),
                F1=math.fsum(f1) / len(ids))


def verify(base):
    base = Path(base); run = read(base / 'logs/run-manifest.json')
    if not native4(run):
        raise ValueError('Final acceptance requires this native4 DynRPO recipe')
    segments = history(base)
    updates = historical_rows(segments, 'logs/actual-updates.jsonl')
    if [row['step'] for row in updates] != list(range(1, 201)):
        raise ValueError('Expected all 200 completed actual actor outers')
    if any(row['run_id'] != run['run_id'] or row['opportunities_processed'] != 20
           or not 1 <= row['actual_optimizer_writes'] <= 20 for row in updates):
        raise ValueError('Actual actor update counts differ from the formal budget')
    writes = {}
    for rank in range(4):
        complete = [r for r in historical_rows(segments, 'logs/scheduler-rank-%d.jsonl' % rank, key='logical_update')
                    if r['kind'] == 'COMPLETE']
        if [r['logical_update'] for r in complete] != list(range(1, 201)):
            raise ValueError('A rank has not completed every outer: ' + str(rank))
        writes[str(rank)] = complete[-1]['writes_after']
    if len(set(writes.values())) != 1:
        raise ValueError('Final rank Adam counts disagree')
    checkpoints = {}
    for step in MILESTONES:
        path = historical_file(segments, step,
            Path('preparation/dynrpo7b-checkpoints') / ('global_step_%d' % step) / 'boundary.json')
        boundary = read(path)
        actor = boundary['actor_ref']
        if (boundary['run_id'] != run['run_id'] or boundary['completed_outer'] != step or
                boundary['status'] != 'DYNRPO7B_ROLLING_READY' or
                boundary['phase']['probe_pending'] or boundary['phase']['grounding_pending']):
            raise ValueError('Unfinished checkpoint callbacks at milestone ' + str(step))
        from .restore import _check_files
        _check_files(actor, Path(boundary['checkpoint']))
        complete = read(actor['complete_path'])
        if complete['files_sha256'] != actor['files_sha256'] or complete['step'] != step:
            raise ValueError('Model/state producer hashes differ at milestone ' + str(step))
        checkpoints[str(step)] = dict(boundary_path=str(path), boundary_sha256=sha(path),
            checkpoint=boundary['checkpoint'], producer_file_count=len(actor['files_sha256']))
    probe_updates = {}
    for step in range(10, 200, 10):
        published_path = historical_file(segments, step,
            Path('artifacts/question-probe') / ('update-%04d' % step) / 'published.json')
        directory = published_path.parent
        expected = DEFAULT_SCHEDULE.update_binding(step)
        summaries = [read(directory / ('epoch-%02d.json' % epoch)) for epoch in (1, 2, 3)]
        published = read(directory / 'published.json')
        if (published['status'] != 'PROBE_UPDATE_PUBLISHED' or
                any(published.get(k) != v for k, v in expected.items()) or
                [s['epoch'] for s in summaries] != [1, 2, 3] or
                any(s['actual_questions'] != 2560 or s['actual_optimizer_updates'] != 20
                    or any(s.get(k) != v for k, v in expected.items()) for s in summaries)):
            raise ValueError('Missing full three-epoch probe update at outer ' + str(step))
        probe_updates[str(step)] = dict(prefix=expected['prefix'], epochs=3,
            publication_sha256=sha(directory / 'published.json'))
    from reproduction.common8.validation_identity import expected as evaluation_expected
    _, dev_frame, dev_ids = evaluation_expected(base, 'validation')
    ledger = historical_rows(segments, 'logs/validation-identities.jsonl')
    evaluations = {}
    first = read(historical_file(segments, 5, 'logs/formal-inline-development-after.json'))
    if first['step'] != 5 or first['run_id'] != run['run_id'] or len(first['rows']) != 512:
        raise ValueError('Actual fixed512 outer5 evaluation is missing')
    if [r['question_id'] for r in first['rows']] != dev_ids:
        raise ValueError('Outer5 evaluation is not the fixed dev512 order')
    evaluations['5'] = dict(questions=512, EM=first['correct_count'] / 512,
                            F1=first['f1_sum'] / 512)
    for step in (50, 100, 150, 200):
        chosen = [p for p in ledger if p['step'] == step and p['split'] == 'validation']
        if len(chosen) != 1:
            raise ValueError('Exactly one fixed dev512 evaluation required at outer ' + str(step))
        evaluations[str(step)] = verify_evaluation(chosen[0], run, dev_ids,
            dev_frame['data_source'].tolist(), step=step, split='validation')
    _, frame, final_ids = evaluation_expected(base, 'final')
    final = read(base / 'logs/final-identity.json')
    if len(final_ids) != 51713 or final.get('final') is not True:
        raise ValueError('Final evaluation must contain all 51713 canonical questions')
    final_metrics = verify_evaluation(final, run, final_ids,
        frame['data_source'].tolist(), step=200, split='final')
    result = dict(status='DYNRPO7B_FORMAL_200_AND_FINAL51713_VERIFIED',
        run_id=run['run_id'], source_commit=run['source_commit'], completed_outers=200,
        rank_actual_optimizer_writes=writes, checkpoints=checkpoints,
        probe_updates=probe_updates, fixed512=evaluations, final51713=final_metrics,
        checkpoint_validation='committed producer SHA map plus unchanged file bytes/mtime',
        repeated_tensor_deserialization=False, continuous_GPU_sampler_required=False)
    publish(base / 'logs/dynrpo7b-final-acceptance.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.base), sort_keys=True), flush=True)


if __name__ == '__main__':
    main()
