"""Small-file CP34 recovery observation; no tensors, model or training operation."""
import json
from pathlib import Path
import re
import time

from reproduction.operations.easy7b_20261021 import observe_fa3_transition_hpc as existing
from reproduction.operations.easy7b_20261021.watch_fa3_transition_local import add_runtime_evidence

BASE = Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fa3-cp34-oom-r1-20261022')
RUN = '232f20e112ac4a978c5aba52a0367195'
SOURCE_JOB = '746125'
DOCUMENTS = ('logs/run-manifest.json', 'logs/pipeline-status.json',
    'logs/full-state-restored-workers.json', 'logs/full-state-restored-driver.json',
    'logs/easy2560-restored-stream-before-next-outer.json')


def arguments(job, runtime_commit, operator_commit):
    if not re.fullmatch(r'[0-9]+', job) or any(re.fullmatch(r'[0-9a-f]{40}', value) is None
                                               for value in (runtime_commit, operator_commit)):
        raise ValueError('Explicit numeric recovery job and full runtime/operator commits are required')


def probe_cache_log(reader):
    value = reader.tail(BASE/'logs/probe-server.log', 65536)
    rows, errors = [], []
    for line in value.get('tail', '').splitlines():
        marker = 'SEARCH_G1_PROBE_CACHE_RELEASE='
        if marker not in line:
            continue
        try: rows.append(json.loads(line.split(marker, 1)[1]))
        except ValueError as error: errors.append(str(error))
    return dict(path=str(BASE/'logs/probe-server.log'), bytes=value.get('bytes'),
        mtime=value.get('mtime'), complete_records_in_tail=len(rows), latest=rows[-1] if rows else None,
        observation_errors=errors, error_lines=[line[-1500:] for line in value.get('tail','').splitlines()
            if 'Traceback (most recent call last)' in line or 'CUDA out of memory' in line])


def optimizer_evidence(reader, value):
    value['post_restore_optimizer_records'] = {}
    for rank in range(4):
        rows, errors = reader.complete_jsonl(BASE/('logs/optimizer-rank-%d.jsonl' % rank))
        value['post_restore_ledger_observation_errors'].extend(errors)
        summaries = []
        for row in rows:
            outer = row.get('logical_update')
            if row.get('kind') != 'OUTER_BUDGET' or type(outer) is not int or outer < 35:
                continue
            writes = [r for r in rows if r.get('kind') == 'OPTIMIZER_WRITE' and
                      r.get('context', {}).get('logical_update') == outer]
            summary = {k:row[k] for k in ('run_id','rank','logical_update','optimizer_writes',
                'cumulative_writes','full_budget_complete','opportunities_processed','local_micro',
                'local_logical_micro','local_mini','gradient_accumulation') if k in row}
            summary['actual_write_counts'] = [r.get('write_count') for r in writes]
            summaries.append(summary)
        value['post_restore_optimizer_records'][str(rank)] = summaries


def observe(job, runtime_commit, operator_commit):
    arguments(job, runtime_commit, operator_commit)
    reader = existing.private_progress_reader(); reader.BASE = BASE
    try: actual_job = (BASE/'job.id').read_text().strip()
    except OSError as error: actual_job = None
    jobs = dict(source=SOURCE_JOB, retrieval_service='745198', new_training=job)
    identifiers = ','.join(jobs.values())
    value = dict(epoch=time.time(), base=str(BASE), requested_runtime_commit=runtime_commit,
        requested_operator_commit=operator_commit, jobs=jobs,
        new_job_record=dict(path=str(BASE/'job.id'), job=actual_job, requested_job=job,
                            matches_requested_job=actual_job == job),
        read_only=True, tensor_files_read=False, credentials_read=False,
        training_runtime_edited=False, jobs_submitted=False)
    value['queue'] = reader.command(['squeue','-j',identifiers,'-h','-o','%i|%T|%N|%R|%S|%L'])
    value['accounting'] = reader.command(['sacct','-j',identifiers,'-n','-P','-o',
                                        'JobIDRaw,State,ExitCode,Elapsed,NodeList'])
    value['documents'] = {name:existing.document(reader, BASE/name) for name in DOCUMENTS}
    value['actual_restore_ranks'] = {str(rank):existing.document(reader,
        BASE/('logs/full-state-restored-rank-%d.json' % rank)) for rank in range(4)}
    # Use the established audit reader through a private function namespace.
    import types
    audit = types.FunctionType(existing.execution_audits.__code__,
        dict(existing.execution_audits.__globals__, BASE=BASE))
    value['actual_flash3_execution'] = audit(reader)
    original_binding = value['documents']['logs/run-manifest.json'].get('value',{}).get('full_state_restore',{})
    add_runtime_evidence(value, reader, BASE)
    value['documents']['logs/run-manifest.json']['value']['full_state_restore']['source_checkpoint'] = original_binding.get('source_checkpoint')
    optimizer_evidence(reader, value)
    metrics, errors = reader.completed_outer_metrics(BASE)
    progress = dict(completed_outer_metrics=metrics, compact_summary_observation_errors=errors,
        candidate_pools=[], training_log=reader.tail(BASE/'logs/search-g1-common8-training.log',20000),
        trainer_log=reader.tail(BASE/'logs/trainer.log',8000))
    latest = BASE/'preparation/rolling-checkpoints/latest.json'
    if latest.is_file(): progress['preparation/rolling-checkpoints/latest.json'] = reader.read_json(latest)
    pools = BASE/'artifacts/mixed-candidate-pools'
    if pools.is_dir():
        for path in sorted(pools.iterdir(), key=lambda p:p.stat().st_mtime)[-3:]:
            item = reader.read_json(path/'progress.json')
            progress['candidate_pools'].append(dict(name=path.name,step=item.get('step'),
                completed_questions=len(item.get('complete_ordinals',[])),
                saved_chunk_files=len(list(path.glob('[0-9]*.pt'))),
                progress_error=item.get('observation_error')))
    value['new_training_progress'] = progress
    value['new_training_slurm_log'] = reader.tail(BASE/('logs/training-%s.log' % job),18000)
    value['grounding_cache_release'] = probe_cache_log(reader)
    resources = sorted((BASE/'logs').glob('native-ray-resources-train-ray-*.json'),key=lambda p:p.stat().st_mtime)
    value['allocator_resources'] = existing.document(reader, resources[-1]) if resources else dict(present=False)
    return value
