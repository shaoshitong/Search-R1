"""Read-only compact native, epoch, head-update and failure observation."""
import argparse
import json
from pathlib import Path
import subprocess
import time


def small(path, limit=262144):
    path = Path(path)
    if not path.is_file():
        return None
    if path.stat().st_size > limit:
        return dict(metadata_omitted=True, bytes=path.stat().st_size)
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError) as error:
        return dict(read_error=str(error))


def tail_json(path, limit=65536):
    path = Path(path)
    if not path.is_file():
        return None
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size-limit))
        raw = stream.read()
    for line in reversed(raw.splitlines()):
        try:
            return json.loads(line)
        except ValueError:
            continue
    return None


def text_tail(path, limit=1200):
    path = Path(path)
    if not path.is_file():
        return None
    with path.open('rb') as stream:
        stream.seek(max(0, path.stat().st_size-limit))
        return stream.read().decode('utf-8', errors='replace')


def observe(base):
    from reproduction.common8.native_allocation import fields
    from reproduction.prefix_only_correctness_ab.launch import birth_states
    base = Path(base).resolve()
    c = small(base/'preparation/launch-config.json') or {}
    receipt = small(base/'preparation/submit-receipt.json')
    stage = small(base/'logs/pipeline-status.json')
    epoch = int((stage or {}).get('epoch', 0))
    epoch_name = 'epoch-%04d' % epoch
    train = base/'train'
    result = dict(base=str(base), run_id=c.get('run_id'), observed_unix=time.time(),
        submit_receipt=receipt, stage=stage, latest=small(train/'latest.json'),
        latest_metrics=tail_json(train/'metrics.jsonl'),
        latest_train_progress=tail_json(train/'progress.jsonl'),
        current_train_complete=small(train/'epochs'/epoch_name/'complete.json'),
        current_rank_actual_updates={str(rank):tail_json(train/'epochs'/epoch_name/('rank-%d-updates.jsonl'%rank))
                                     for rank in range(8)},
        current_train_errors={str(rank):small(train/'epochs'/epoch_name/('rank-%d-error.json'%rank)) for rank in range(8)},
        current_feature_progress={str(rank):small(base/'epochs'/epoch_name/'features'/('progress-rank-%d.json'%rank)) for rank in range(8)},
        current_rollout_complete=None, current_rollout_ranks={},
        current_rollout_progress={str(rank):small(base/'epochs'/epoch_name/'rollout'/('rank-%d-progress.json'%rank)) for rank in range(8)},
        pipeline_complete=small(base/'logs/pipeline-complete.json'), cleanup=small(base/'logs/cleanup-status.json'))
    complete = small(base/'epochs'/epoch_name/'complete.json', limit=2*1024**2)
    if complete:
        result['current_rollout_complete'] = {key:complete.get(key) for key in
            ('epoch','questions','trajectories','model_revision','selection_sha256','actual_generated_trajectories','cached_epoch')}
    for rank in range(8):
        value = small(base/'epochs'/epoch_name/'rollout'/('rank-%d-complete.json'%rank))
        if value:
            result['current_rollout_ranks'][str(rank)] = {key:item for key,item in value.items()
                if isinstance(item,(str,int,float,bool)) or item is None}
    ledger = small(base/'logs/process-ledger.json', limit=64*1024**2) or {}
    births = [row for job in ledger.get('jobs', {}).values() for row in job.get('ledger', {}).values()]
    present = birth_states(births)
    result.update(owned_births_recorded=len(births), owned_births_present=len(present),
                  owned_active_births=sum(row['state']!='Z' for row in present),
                  owned_births_all_gone=not present)
    active = [(name, job) for name,job in ledger.get('jobs',{}).items() if job.get('state')!='STOPPED']
    result['active_processes'] = [dict(name=name, state=job.get('state'), leader=job.get('leader'),
        log_tail=text_tail(base/'logs'/(name+'.log'))) for name,job in active[-4:]]
    result['native_terminal_verified'] = False
    result['native_actual_exit0_verified'] = False
    if receipt and receipt.get('job_id'):
        job = str(receipt['job_id'])
        try:
            raw = subprocess.check_output(['scontrol','show','job',job,'-o'],text=True,stderr=subprocess.PIPE,timeout=30)
            native = fields(raw)
            if native.get('JobId') != job or native.get('Comment') != 'g1-prefix20-online:'+c['run_id']:
                raise ValueError('Actual native identity differs')
            result['native'] = {key:native.get(key) for key in ('JobId','JobState','ExitCode','NodeList','RunTime','TimeLimit')}
            terminal = native.get('JobState') in ('COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED')
            result['native_terminal_verified'] = terminal
            result['native_actual_exit0_verified'] = native.get('JobState')=='COMPLETED' and native.get('ExitCode')=='0:0'
            if terminal:
                result['native_terminal_raw'] = raw
                result['native_cgroup_exists'] = bool(list(Path('/sys/fs/cgroup').glob('**/job_'+job)) + list(Path('/sys/fs/cgroup').glob('**/job'+job)))
                gpu = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True,timeout=30).split()
                result['owned_gpu_pid_numbers_present'] = sorted(set(gpu) & {str(row['pid']) for row in births})
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
            result['native_observation_unavailable'] = str(error)
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', required=True)
    args = parser.parse_args()
    print(json.dumps(observe(args.base), allow_nan=False))


if __name__ == '__main__':
    main()
