"""Passive five-second whole-card/owned-process sampler; never controls training.

Only scalar telemetry is written. CPU is a lower bound over observed process
births, because this node's Slurm job has no CPU accounting controller.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time


def read(path):
    return json.loads(path.read_text())


def atomic(path, value):
    tmp = path.with_suffix('.tmp')
    with tmp.open('w') as stream:
        json.dump(value, stream, sort_keys=True, allow_nan=False)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(tmp, path)


def combine_resource_summaries(summaries):
    """Preserve disjoint telemetry segments without double-counting CPU births."""
    import math
    if not summaries:
        return None
    ordered = sorted(summaries, key=lambda row: row['started_unix'])
    latest = ordered[-1]
    identity = tuple(latest[key] for key in ('run_id', 'base', 'job'))
    peaks = {}; segments = []; gaps = []; cpu = 0.; previous = None
    for row in ordered:
        if tuple(row[key] for key in ('run_id', 'base', 'job')) != identity:
            raise ValueError('Resource segment belongs to another run/allocation')
        if row['cpu_accounting_complete'] is not False:
            raise ValueError('Observed CPU must retain its incomplete scope')
        lower = row['latest']['owned_cpu_core_seconds_lower_bound']
        if not math.isfinite(lower) or lower < 0:
            raise ValueError('Invalid observed CPU lower bound')
        cpu = max(cpu, lower)
        if previous is not None:
            if row['started_unix'] < previous['last_observed_unix']:
                raise ValueError('Telemetry segments overlap; reconcile before aggregation')
            gaps.append(dict(after_unix=previous['last_observed_unix'],
                             before_unix=row['started_unix'],
                             peak_and_cpu_samples_available=False))
        for uuid, card in row['whole_card_peaks'].items():
            peak = peaks.setdefault(uuid, dict(card))
            peak['max_used_mib'] = max(peak['max_used_mib'], card['max_used_mib'])
            peak['min_free_mib'] = min(peak['min_free_mib'], card['min_free_mib'])
            peak['max_utilization_percent'] = max(peak['max_utilization_percent'], card['max_utilization_percent'])
        segments.append(dict(source=row.get('segment_source'), started_unix=row['started_unix'],
            last_observed_unix=row['last_observed_unix'], samples=row['samples'],
            owned_cpu_core_seconds_lower_bound=lower))
        previous = row
    result = dict(latest)
    result.update(started_unix=ordered[0]['started_unix'], samples=sum(row['samples'] for row in ordered),
        whole_card_peaks=peaks, resource_segments=segments, resource_coverage_gaps=gaps,
        cpu_lower_bound_combination='Maximum across segments; overlapping process lifetimes are never summed',
        coverage='Observed telemetry segments only; gaps have no retrospective GPU peak or complete CPU claim')
    result['latest'] = dict(latest['latest'], owned_cpu_core_seconds_lower_bound=cpu)
    return result


def own_cpu(record):
    """Never attribute a reused PID or another session's CPU to this task."""
    path = Path('/proc') / str(record['pid'])
    try:
        fields = (path/'stat').read_text().rsplit(')', 1)[1].split()
        if (int(fields[19]) != record['start_ticks'] or int(fields[3]) != record['sid']
                or path.stat().st_uid != record['uid']):
            return None
        return int(fields[11]) + int(fields[12])
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None


def sample(base, job, output, interval=5., max_seconds=172800):
    base, output = Path(base), Path(output)
    if (base.parent != Path('/mnt/data/shaoshitong') or not base.name.startswith('search-g1-prefix-diversity200-')
            or base.resolve() != base or output not in (base/'preparation/resource5s-v1', base/'preparation/resource5s-v2')):
        raise ValueError('Require the canonical task-local telemetry directory')
    config = read(base/'preparation/launch-config.json')
    if read(base/'preparation/submit-receipt.json')['job_id'] != str(job):
        raise ValueError('Telemetry native identity differs')
    destination = output/'samples.jsonl'
    if destination.exists():
        raise FileExistsError('Existing telemetry is preserved; never duplicate/replay the sampler')
    started = time.time(); clock = os.sysconf('SC_CLK_TCK'); cpu_max = {}; known_births = set(); peaks = {}; count = 0
    native = {}; terminal = False; last_native_check = 0.; last_cpu_check = 0.
    cpu_interval = 60. if output.name == 'resource5s-v2' else interval
    with destination.open('x') as stream:
        while time.time()-started < max_seconds:
            began = time.monotonic(); now = time.time(); errors = []
            if now-last_native_check >= 60 or not native:
                result = subprocess.run(['scontrol','show','job',str(job),'-o'],capture_output=True,text=True,timeout=15)
                match = re.search(r'(?:^|\s)JobState=(\S+)', result.stdout)
                native = dict(returncode=result.returncode, state=match.group(1) if match else 'UNKNOWN',
                              stdout=result.stdout, stderr=result.stderr, observed_unix=now)
                last_native_check = now
                if result.returncode == 0 and ('JobId='+str(job) not in result.stdout or 'WorkDir='+str(base) not in result.stdout):
                    raise ValueError('Actual native allocation identity changed')
                terminal = native['state'] in ('COMPLETED','FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL')
            if now-last_cpu_check >= cpu_interval:
                ledger = read(base/'logs/process-ledger.json')
                if ledger['run_id'] != config['run_id']:
                    raise ValueError('Process ledger belongs to another run')
                for part in ledger['jobs'].values():
                    for record in part['ledger'].values():
                        key = '%d:%d:%d:%d' % (record['pid'],record['start_ticks'],record['sid'],record['uid'])
                        known_births.add(key)
                        if part.get('state') == 'STOPPED' and key not in cpu_max:
                            continue
                        ticks = own_cpu(record)
                        if ticks is not None:
                            cpu_max[key] = max(cpu_max.get(key, 0), ticks)
                del ledger
                last_cpu_check = now
            gpu = subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.total,memory.free,utilization.gpu',
                '--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=15)
            cards = []
            if gpu.returncode == 0:
                for line in gpu.stdout.splitlines():
                    values = [v.strip() for v in line.split(',')]
                    card = dict(index=int(values[0]),uuid=values[1],used_mib=int(values[2]),
                                total_mib=int(values[3]),free_mib=int(values[4]),utilization_percent=int(values[5]))
                    cards.append(card)
                if len(cards) != 8 or [v['uuid'] for v in sorted(cards,key=lambda x:x['index'])] != config['physical_gpu_uuids']:
                    raise ValueError('Whole-card UUID mapping differs')
                for card in cards:
                    prior = peaks.setdefault(card['uuid'],dict(max_used_mib=0,min_free_mib=card['total_mib'],max_utilization_percent=0))
                    prior['max_used_mib'] = max(prior['max_used_mib'],card['used_mib'])
                    prior['min_free_mib'] = min(prior['min_free_mib'],card['free_mib'])
                    prior['max_utilization_percent'] = max(prior['max_utilization_percent'],card['utilization_percent'])
            else:
                errors.append(dict(kind='nvidia_smi_read_failed',returncode=gpu.returncode,stderr=gpu.stderr))
            stage = read(base/'logs/pipeline-status.json')
            row = dict(observed_unix=now,job=str(job),stage=stage,cards=cards,
                owned_cpu_core_seconds_lower_bound=sum(cpu_max.values())/clock,
                observed_owned_process_births=len(cpu_max),known_owned_process_births=len(known_births),
                native_state=native['state'],errors=errors)
            stream.write(json.dumps(row,sort_keys=True,allow_nan=False)+'\n');stream.flush();os.fsync(stream.fileno());count += 1
            summary = dict(run_id=config['run_id'],base=str(base),job=str(job),started_unix=started,
                last_observed_unix=now,samples=count,requested_interval_seconds=interval,
                actual_last_sample_seconds=time.monotonic()-began,cpu_scan_interval_seconds=cpu_interval,latest=row,whole_card_peaks=peaks,native=native,
                cpu_scope='lower bound over actually observed owned birth utime+stime; stopped/unobserved processes and final intersample CPU may be missing',
                cpu_accounting_complete=False,gpu_scope='whole physical card including any other users; sampled peak is not an exact continuous peak',
                coverage='starts at this telemetry creation; no retrospective GPU peak claim',
                training_changed=False,model_tensors_read=False,new_model_forward=False)
            if native['returncode'] and 'Invalid job id specified' in native['stderr'] and (base/'logs/cleanup-status.json').exists():
                terminal = read(base/'logs/cleanup-status.json').get('owned_births_all_gone',False)
            summary['sampler_finished'] = terminal
            atomic(output/'summary.json',summary)
            if terminal:
                atomic(output/'complete.json',dict(summary,status='SOURCE_TERMINAL_OR_CLEANED_HANDLE_MISSING'))
                return
            time.sleep(max(.01, interval-(time.monotonic()-began)))
    atomic(output/'complete.json',dict(summary,status='OBSERVER_TIME_LIMIT_TRAINING_UNCHANGED'))


if __name__ == '__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--base',required=True);parser.add_argument('--job',required=True)
    parser.add_argument('--output',required=True);args=parser.parse_args();sample(args.base,args.job,args.output)
