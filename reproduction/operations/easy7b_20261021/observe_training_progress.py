"""Read-only compact progress observer for the immutable CP1 dense recovery."""
import json
import re
import subprocess
import time
from collections import Counter
from pathlib import Path

BASE = Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-exact-cp1-r2-20261021')
SERVICE = Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-retrieval-service2-20261021')


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as error:
        return {'observation_error': str(error)}


def tail(path, limit=14000):
    try:
        stat = path.stat()
        with path.open('rb') as stream:
            stream.seek(max(0, stat.st_size-limit))
            content = stream.read().decode(errors='replace')
        return dict(bytes=stat.st_size, mtime=stat.st_mtime, tail=content)
    except OSError as error:
        return {'observation_error': str(error)}


def command(args):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=12)
        return dict(rc=result.returncode, stdout=result.stdout, stderr=result.stderr)
    except subprocess.TimeoutExpired:
        return dict(observation_timeout=True)


def complete_jsonl(path):
    """Read complete durable records; an in-progress final line is not evidence."""
    rows, errors = [], []
    if not path.is_file():
        return rows, errors
    try:
        with path.open('rb') as stream:
            for number, line in enumerate(stream, 1):
                if not line.endswith(b'\n'):
                    break
                try:
                    value = json.loads(line)
                    if not isinstance(value, dict):
                        raise ValueError('record is not an object')
                    rows.append(value)
                except (ValueError, UnicodeError) as error:
                    errors.append(dict(path=str(path), line=number, observation_error=str(error)))
    except OSError as error:
        errors.append(dict(path=str(path), observation_error=str(error)))
    return rows, errors


def completed_outer_metrics(base):
    """Current execution's measured completed updates, without a synthetic prefix."""
    path = base/'logs/actual-updates.jsonl'
    rows, errors = complete_jsonl(path)
    result = []
    metric_names = dict(reward='critic/rewards/mean', pg_loss='actor/pg_loss',
                        ppo_kl='actor/ppo_kl', grad_norm='actor/grad_norm')
    for row in rows:
        if row.get('update_status') != 'complete' or type(row.get('step')) is not int:
            continue
        value = dict(step=row['step'])
        if 'attempt_index' in row:
            value['attempt_index'] = row['attempt_index']
        metrics = row.get('metrics', {})
        for name, key in metric_names.items():
            if key in metrics:
                value[name] = metrics[key]
        for name, key in (('optimizer_writes', 'optimizer_writes_after'),
                          ('optimizer_writes_before', 'optimizer_writes_before'),
                          ('optimizer_writes_this_outer', 'actual_optimizer_writes')):
            if key in row:
                value[name] = row[key]
        result.append(value)
    return result, errors


def selection_em_summary(update, audit, path):
    """Measure the original complete strict-EM groups, independently of reward."""
    if not isinstance(audit, dict):
        raise ValueError('selection audit is not an object')
    step, aid = update['step'], update['attempt_id']
    expected_snapshot = update['run_id'] + ':' + str(step) + ':policy-' + aid
    if audit.get('step') != step or audit.get('policy_snapshot_id') != expected_snapshot or \
            audit.get('fixed_candidate_budget') is not True or \
            audit.get('attempted_candidates') != 2560 or audit.get('responses_per_group') != 5 or \
            audit.get('selected_groups') != 512:
        raise ValueError('completed selection identity or fixed2560/G5/selected512 budget differs')
    groups = audit.get('groups')
    if not isinstance(groups, list) or len(groups) != 2560 or not all(isinstance(group, dict) for group in groups) or \
            [group.get('ordinal') for group in groups] != list(range(2560)):
        raise ValueError('selection audit does not contain all2560 ordered completed groups')
    histogram = {str(k): 0 for k in range(6)}
    selected_histogram = {str(k): 0 for k in range(6)}
    mixed, all_correct, chosen = [], [], []
    for group in groups:
        labels = group.get('strict_correctness')
        if not isinstance(labels, list) or len(labels) != 5 or \
                any(type(value) is not int or value not in (0, 1) for value in labels):
            raise ValueError('selection group has incomplete/nonbinary strict-EM labels')
        k = sum(labels)
        category = 'mixed' if 0 < k < 5 else 'all_correct' if k == 5 else 'all_wrong'
        if group.get('category') != category or type(group.get('selected')) is not bool:
            raise ValueError('selection category/selected flag differs from actual strict EM')
        histogram[str(k)] += 1
        if category == 'mixed':
            mixed.append(group['ordinal'])
        elif category == 'all_correct':
            all_correct.append(group['ordinal'])
        if group['selected']:
            chosen.append(group['ordinal'])
            selected_histogram[str(k)] += 1
    expected_chosen = sorted(mixed[:512] if len(mixed) >= 512 else
                             mixed + all_correct[:512-len(mixed)])
    mixed_count = sum(histogram[str(k)] for k in range(1, 5))
    selected_mixed = sum(selected_histogram[str(k)] for k in range(1, 5))
    fallback = selected_histogram['5']
    if sum(histogram.values()) != 2560 or mixed_count != len(mixed) or \
            len(chosen) != 512 or chosen != expected_chosen or \
            audit.get('selected_mixed') != selected_mixed or \
            audit.get('selected_all_correct_fallback') != fallback or selected_mixed + fallback != 512:
        raise ValueError('candidate K histogram, mixed counts or actual ordinal mixed-first selection differs')
    # raw_generated_trajectories can include discarded insufficient pools. The
    # measured histogram describes only this completed successful fixed pool.
    retries, discarded = audit.get('pool_retry'), audit.get('discarded_insufficient_pools')
    raw_count = audit.get('raw_generated_trajectories')
    if type(retries) is not int or retries < 0 or not isinstance(discarded, list) or \
            len(discarded) != retries or type(raw_count) is not int or raw_count != 12800*(retries+1):
        raise ValueError('successful pool versus discarded/raw generation denominator differs')
    return dict(step=step, attempt_id=aid, policy_snapshot_id=expected_snapshot,
        status='MEASURED_COMPLETED_SELECTION', source_path=str(path),
        candidate_correct_count_histogram=histogram, candidate_questions=2560,
        candidate_trajectories=12800, raw_generated_trajectories=raw_count,
        candidate_trajectory_em=sum(k*histogram[str(k)] for k in range(6))/12800,
        candidate_mixed_questions=mixed_count, candidate_mixed_fraction=mixed_count/2560,
        selected_correct_count_histogram=selected_histogram, selected_questions=512,
        selected_trajectories=2560,
        selected_trajectory_em=sum(k*selected_histogram[str(k)] for k in range(6))/2560,
        selected_mixed=selected_mixed, selected_mixed_fraction=selected_mixed/512,
        selected_all_correct_fallback=fallback, selected_all_correct_fallback_fraction=fallback/512,
        pool_retry=retries, discarded_insufficient_pool_count=len(discarded),
        selection_basis='actual five strict terminal EM labels; first mixed ordinals, then K5 fallback',
        candidate_em_scope='successful fixed2560 pool only; discarded pools excluded; pre-update selection policy')


def completed_selection_summaries(base):
    """Bound large audit reads to the last ten actual completed execution rows."""
    rows, errors = complete_jsonl(base/'logs/actual-updates.jsonl')
    completed = [row for row in rows if row.get('update_status') == 'complete' and type(row.get('step')) is int]
    result = []
    for update in completed[-10:]:
        value = dict(step=update['step'], attempt_id=update.get('attempt_id'), status='UNKNOWN')
        try:
            aid = update.get('attempt_id')
            if not isinstance(aid, str) or re.fullmatch('[0-9a-f]{32}', aid) is None:
                raise ValueError('completed update has no valid selection attempt identity')
            path = base/'artifacts/mixed-selection'/(aid+'.json')
            value['source_path'] = str(path)
            audit = read_json(path)
            if 'observation_error' in audit:
                raise ValueError(audit['observation_error'])
            value = selection_em_summary(update, audit, path)
        except (KeyError, TypeError, ValueError) as error:
            value['observation_error'] = str(error)
        result.append(value)
    return result, errors


def validation_summary(record, path):
    """Derive small EM aggregates from actual persisted per-question results."""
    final = record.get('final', False)
    if type(final) is not bool or type(record.get('step')) is not int:
        raise ValueError('invalid validation step/final identity')
    if isinstance(record.get('em'), list):
        em, sources = record['em'], record.get('data_sources', [])
    else:
        rows = record.get('rows', [])
        em = [row['em'] for row in rows]
        sources = [row['data_source'] for row in rows]
    count = len(em)
    counts = dict(Counter(sources))
    if len(sources) != count or any(value not in (0, 1) for value in em):
        raise ValueError('incomplete or invalid per-question EM results')
    if final:
        if count != 51713 or len(counts) != 7:
            raise ValueError('final identity is not native51713 across seven sources')
    elif count != 512 or counts != dict(nq=256, hotpotqa=256):
        raise ValueError('fixed validation identity is not NQ256/Hotpot256')
    correct = {source: sum(score for label, score in zip(sources, em) if label == source)
               for source in counts}
    total = sum(em)
    if record.get('counts', record.get('source_counts', counts)) != counts:
        raise ValueError('persisted validation source counts differ from measured rows')
    if record.get('correct', record.get('correct_count', total)) != total or \
            record.get('weighted_em', total/count) != total/count:
        raise ValueError('persisted validation scalar EM differs from measured rows')
    return dict(step=record['step'], final=final, count=count, source_counts=counts,
                source_correct=correct, source_em={source: correct[source]/counts[source] for source in counts},
                weighted_em=total/count, correct_count=total, source_path=str(path))


def fixed_validation_summaries(base):
    path = base/'logs/validation-identities.jsonl'
    rows, errors = complete_jsonl(path)
    result = []
    identities = set()
    for record in rows:
        if record.get('split') != 'validation' and record.get('final') is not True:
            continue  # Stage64 is not fixed512 or final51713 evaluation.
        try:
            value = validation_summary(record, path)
            result.append(value)
            identities.add((value['step'], value['final']))
        except (KeyError, TypeError, ValueError) as error:
            errors.append(dict(path=str(path), step=record.get('step'), observation_error=str(error)))
    # QA5 is written just before the normal identity ledger. Summarize this
    # genuine completed result during that brief window, without duplicating it.
    after = base/'logs/formal-inline-development-after.json'
    if after.is_file():
        record = read_json(after)
        if (record.get('step'), False) not in identities:
            try:
                if record.get('role') != 'formal_inline_development_evaluation' or record.get('step') != 5:
                    raise ValueError('unexpected fixedQA5 artifact identity')
                result.append(validation_summary(record, after))
            except (KeyError, TypeError, ValueError) as error:
                errors.append(dict(path=str(after), observation_error=str(error)))
    return result, errors


def main():
    job = (BASE/'job.id').read_text().strip()
    result = dict(epoch=time.time(), base=str(BASE), job=job,
                  read_only=True, training_runtime_edited=False)
    result['queue'] = command(['squeue', '-j', job+',745198', '-h', '-o', '%i|%T|%N|%R|%S|%L'])
    result['accounting'] = command(['sacct', '-j', job+',745198', '-n', '-P', '-o', 'JobIDRaw,State,ExitCode,Elapsed,NodeList'])
    for name in ['logs/pipeline-status.json', 'logs/full-state-restored-workers.json',
                 'logs/full-state-restored-driver.json', 'logs/easy2560-restored-stream-before-next-outer.json',
                 'preparation/rolling-checkpoints/latest.json', 'logs/cleanup-status.json']:
        if (BASE/name).is_file():
            result[name] = read_json(BASE/name)
    result['ledgers'] = {}
    for name in ['actual-updates.jsonl', 'actual-attempts.jsonl'] + [f'optimizer-rank-{rank}.jsonl' for rank in range(4)]:
        path = BASE/'logs'/name
        if path.is_file():
            result['ledgers'][name] = tail(path, 18000)
    metrics, errors = completed_outer_metrics(BASE)
    result['completed_outer_metrics'] = metrics
    result['completed_outer_metrics_scope'] = 'Current execution actual-updates.jsonl only; inherited checkpoint prefix is not synthesized'
    selections, selection_errors = completed_selection_summaries(BASE)
    result['completed_selection_em'] = selections
    result['completed_selection_em_scope'] = 'Last at most10 completed current-execution updates; actual strict-EM audit groups, not in-progress pools or actor reward'
    validations, validation_errors = fixed_validation_summaries(BASE)
    if validations:
        result['fixed_validation_identities'] = validations
    if errors or selection_errors or validation_errors:
        result['compact_summary_observation_errors'] = errors + selection_errors + validation_errors
    result['candidate_pools'] = []
    pools = BASE/'artifacts/mixed-candidate-pools'
    if pools.is_dir():
        for path in sorted(pools.iterdir(), key=lambda p:p.stat().st_mtime)[-3:]:
            value = read_json(path/'progress.json')
            result['candidate_pools'].append(dict(name=path.name, step=value.get('step'),
                completed_questions=len(value.get('complete_ordinals', [])),
                reserved_questions=len(value.get('reserved', [])), source_consumed=value.get('source_consumed'),
                saved_chunk_files=len(list(path.glob('[0-9]*.pt'))),
                progress_error=value.get('observation_error')))
    # File metadata only: distinguish checkpoint writes from an idle process.
    result['checkpoint_files'] = []
    actor = BASE/'checkpoints/search-g1-common8/actor'
    for directory in sorted(actor.glob('global_step_*'), key=lambda p:p.stat().st_mtime)[-2:]:
        files = [path for path in directory.rglob('*') if path.is_file()]
        result['checkpoint_files'].append(dict(path=str(directory), files=len(files),
            bytes=sum(path.stat().st_size for path in files),
            last_mtime=max((path.stat().st_mtime for path in files), default=None)))
    result['scheduler_ledgers'] = {}
    for rank in range(4):
        path = BASE/f'logs/scheduler-rank-{rank}.jsonl'
        if path.is_file():
            rows = []
            for line in tail(path, 12000)['tail'].splitlines():
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
            result['scheduler_ledgers'][path.name] = rows[-2:]
    result['training_log'] = tail(BASE/'logs/search-g1-common8-training.log', 20000)
    result['trainer_log'] = tail(BASE/'logs/trainer.log', 5000)
    gpu = BASE/'logs/gpu-samples-232f20e112ac4a978c5aba52a0367195.jsonl'
    result['gpu_sample_tail'] = tail(gpu, 3000)
    try:
        import requests
        ready = read_json(SERVICE/'logs/service-ready.json')
        with requests.Session() as session:
            session.trust_env = False
            response = session.get(ready['endpoint'].replace('/retrieve','/health'), timeout=8)
            health = response.json()
        result['service_health'] = {key:health.get(key) for key in ['ok','service_job_id','hostname','physical_gpu_count','statistics']}
    except Exception as error:
        result['service_observation_error'] = str(error)
    print(json.dumps(result))


if __name__ == '__main__':
    main()
