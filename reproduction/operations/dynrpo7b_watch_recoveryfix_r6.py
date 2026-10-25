"""Read-only live DynRPO monitor; return small summaries, never model/rollout tensors.

This observer does not change the running source or submit work. A saved receipt
is reported as evidence, not treated as a fresh full model hash/load validation.
"""
from reproduction.dynrpo7b.local_control import remote

SCRIPT = r'''
import json, pathlib, re, subprocess, time
base = pathlib.Path('/data/user/jxu957/experiments/dynrpo7b-formal-20261023-recoveryfix-r6')
result = {'time': time.time(), 'training_job': 751884, 'retrieval_job': 751011}

def read(relative):
    path = base / relative
    return json.loads(path.read_text()) if path.exists() else None

def fields(row, names):
    return {key: row.get(key) for key in names}

query = subprocess.run(['squeue', '-j', '751884,751011', '-h', '-o', '%i %T %M %N'],
    stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
result['queue_query_returncode'] = query.returncode
result['queue'] = query.stdout.strip()
if query.returncode:
    result['queue_query_error'] = query.stderr[-500:]
elif not any(row.startswith('751884 ') for row in query.stdout.splitlines()):
    query = subprocess.run(['sacct', '-X', '-j', '751884', '-o', 'State,ExitCode', '-n', '-P'],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True)
    result['terminal'] = query.stdout.strip()

pipeline = read('logs/pipeline-status.json')
if pipeline:
    result['pipeline'] = pipeline.get('stage')
errors = ['Traceback (most recent call last)', 'RayTaskError', 'CUDA out of memory']
for name, key in [('trainer.log', 'trainer'), ('train-751884.log', 'launcher')]:
    path = base / 'logs' / name
    if not path.exists():
        continue
    content = path.read_text()
    result[key] = {'bytes': path.stat().st_size, 'mtime': path.stat().st_mtime,
                   'error': any(value in content for value in errors)}
    if result[key]['error']:
        result[key]['error_tail'] = content[-2500:]
    if key != 'trainer':
        continue
    collections, metrics, last_collection = [], [], None
    for line in content.splitlines():
        if 'DYNRPO_COLLECTION=' in line:
            row = json.loads(line.split('DYNRPO_COLLECTION=', 1)[1])
            last_collection = row['step']
            collections.append(fields(row, ['step', 'candidate_fill_seconds', 'probe_top_group_counts',
                'selected_mixed', 'selected_mixed_fraction', 'attempt_id', 'question_probe_version']))
        if 'actor/completed_outer:' in line:
            items = {k: float(v) for k, v in re.findall(r'([A-Za-z0-9_/]+):(-?[0-9]+(?:\.[0-9]+)?)', line)}
            wanted = {k: v for k, v in items.items() if k.startswith(('actor/completed_outer',
                'infra/rollout/weight_syncs_per_rank', 'timing_s/')) or k.endswith('/em')}
            wanted['collection_outer'] = last_collection
            metrics.append(wanted)
    result['recent_collections'] = collections[-4:]
    result['recent_steps'] = metrics[-4:]

result['writes'] = []
for rank in range(4):
    path = base / 'logs' / ('optimizer-rank-%d.jsonl' % rank)
    rows = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
    writes = [row['write_count'] for row in rows if row.get('kind') == 'OPTIMIZER_WRITE']
    result['writes'].append(writes[-1] if writes else None)

latest = read('preparation/dynrpo7b-checkpoints/latest.json')
if latest:
    probe = latest['question_probe']
    result['checkpoint'] = fields(latest, ['completed_outer', 'status', 'source_commit'])
    result['checkpoint']['rank_saved_writes'] = [row['actual_optimizer_writes'] for row in latest['actor_ref']['worker_files']]
    result['checkpoint']['probe'] = fields(probe, ['head_version', 'head_sha256', 'fit_completed_epochs'])
    result['checkpoint']['probe']['reference'] = fields(probe['reference'],
        ['path', 'sha256', 'warmup_updates', 'online_updates', 'completed_update_outer', 'current_prefix'])
    result['checkpoint']['post_callback_rng_captured'] = latest['post_callback_rng_captured']

phase = read('preparation/dynrpo7b-checkpoints/formal-state.json')
if phase:
    result['phase'] = fields(phase, ['completed_actor_outer', 'phase', 'probe_pending',
        'probe_completed_epoch', 'probe_effective_from_outer', 'grounding_pending', 'next_outer'])

journals = sorted((base / 'preparation/dynrpo7b-collection').glob('outer-*.json'))
if journals:
    row = json.loads(journals[-1].read_text())
    result['latest_collection_journal'] = {'name': journals[-1].name, 'stage': row['stage'], 'parts': len(row['parts'])}

updates = sorted((base / 'artifacts/question-probe').glob('update-*'))
if updates:
    directory = updates[-1]
    epochs = []
    for path in sorted(directory.glob('epoch-*.json')):
        row = json.loads(path.read_text())
        epochs.append(fields(row, ['epoch', 'target4variance_MSE', 'actual_questions',
            'actual_optimizer_updates', 'online_updates', 'warmup_updates', 'seconds',
            'dataset_sha256', 'completed_actor_outer', 'rollout_actor_outer', 'prefix']))
    result['probe_update'] = {'directory': directory.name, 'epochs': epochs}
    path = directory / 'published.json'
    if path.exists():
        row = json.loads(path.read_text())
        result['probe_update']['publication'] = fields(row, ['status', 'online_updates',
            'dataset_sha256', 'completed_actor_outer', 'rollout_actor_outer', 'effective_state', 'prefix'])
        result['probe_update']['publication']['checkpoint_sha256'] = row['checkpoint']['sha256']
print(json.dumps(result))
'''

def observation_script(base=None, training_job=751884, retrieval_job=751011):
    from reproduction.dynrpo7b.infra_profile import own_base
    text = SCRIPT
    if base is not None:
        text = text.replace('/data/user/jxu957/experiments/dynrpo7b-formal-20261023-recoveryfix-r6', str(own_base(base)))
    if not str(training_job).isdecimal() or not str(retrieval_job).isdecimal():
        raise ValueError('Slurm job IDs must be decimal')
    return text.replace('751884',str(training_job)).replace('751011',str(retrieval_job))


if __name__ == '__main__':
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base')
    parser.add_argument('--training-job',type=int,default=751884)
    parser.add_argument('--retrieval-job',type=int,default=751011)
    args=parser.parse_args()
    remote(observation_script(args.base,args.training_job,args.retrieval_job))
