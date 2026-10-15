"""Preserve a qualified probe after an authenticated zero-student-update failure."""
import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
import pwd
import contextlib
import fcntl
import hashlib
import shlex
import socket

from .common import BASE,DEFAULT_BASE,MODEL,own,read,sha,publish,verify_source

DEFERRED_KIND = 'ZERO_UPDATE_RETRY_WITH_DEFERRED_OWNED_CLEANUP_V1'
CONTROL_PYTHON = '/data/user/jxu957/experiments/topd-20261008/environment/topd-py310/bin/python'
SOURCE_LOCKS = ('preparation/formal.lock', 'preparation/preparation.lock', 'gpu-job.lock')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def pristine_student(source):
    """A missing claimed training boundary is mandatory, even for empty files."""
    for name in ('training-launch-claim.json', 'actual-updates.jsonl'):
        require(not (source/'logs'/name).exists(), 'Training may have started; require actual full-state recovery instead')
    require(not list((source/'logs').glob('optimizer-rank-*')) and
            not list((source/'logs').glob('policy-calibration-optimizer-rank-*')) and
            not any(p.is_file() for p in (source/'checkpoints').rglob('*')) and
            not any(p.is_file() for p in (source/'artifacts/question-probe/periodic-refits').rglob('*')),
            'Actual student or online-probe state exists; never reinitialize this prefix')


def validate_pretraining_ledger(ledger, source, run_id, uid, calibration_log):
    """Only this exact import-time failure may defer its retriever cleanup."""
    expected = {
        'gpu-preflight': [str(source/'env/bin/python'), '-m', 'torch.distributed.run', '--standalone',
                          '--nproc_per_node=4', 'reproduction/common8/gpu_preflight.py'],
        'model-initial': [str(source/'env/bin/python'), '-m', 'torch.distributed.run', '--standalone',
                          '--nproc_per_node=4', '-m', 'reproduction.common8.model_smoke', '--phase', 'initial'],
        'retrieval-stage': [str(source/'env/bin/python'), '-m', 'reproduction.common8.stage_retrieval',
                            '--base', str(source), '--port', '18016'],
        'retriever': [str(source/'env/bin/python'), '-m', 'reproduction.common8.retriever'],
        'initial-calibration': [str(source/'env/bin/python'), '-m', 'reproduction.common8.refit',
                                '--step', '0', '--model', str(MODEL)],
    }
    jobs = ledger.get('jobs', {})
    require(ledger.get('run_id') == run_id and set(jobs) == set(expected),
            'Deferred cleanup requires only the five known pretraining roles')
    require(calibration_log.rstrip().endswith("ModuleNotFoundError: No module named 'joblib'") and
            'reproduction/common8/refit.py' in calibration_log,
            'Deferred cleanup requires the actual initial-calibration joblib import failure')
    for name, arguments in expected.items():
        item = jobs[name]
        require(item.get('arguments') == arguments and item.get('ledger') and
                Path(item.get('log', '')) == source/'logs'/(name+'.log'),
                'Deferred cleanup role arguments or log changed: '+name)
        require(all(row.get('uid') == uid and row.get('sid') == item['leader'].get('sid') and
                    type(row.get('pid')) is int and type(row.get('start_ticks')) is int
                    for row in [item['leader'], *item['ledger'].values()]),
                'Deferred cleanup process identity changed: '+name)
        if name == 'retriever':
            require(item.get('state') == 'CLEANUP_FAILED' and item.get('exit_code') is None,
                    'Deferred cleanup is limited to the original uncleaned retriever')
        else:
            require(item.get('state') == 'STOPPED' and item.get('exit_code') == (1 if name=='initial-calibration' else 0),
                    'A different pretraining component remains unresolved: '+name)


def validate_cleanup_job(job, script, *, cleanup_job, failed_job, source, old_node, username, uid):
    require(job.get('JobId') == cleanup_job and job.get('JobState') in ('PENDING', 'RUNNING') and
            job.get('UserId') == username+'('+str(uid)+')' and
            job.get('JobName') == 'dynrpo-thread5-cleanup-'+failed_job and
            job.get('WorkDir') == str(source) and job.get('ReqNodeList') == old_node and
            job.get('NumNodes') == '1' and
            (job.get('JobState') != 'RUNNING' or job.get('NodeList') == old_node),
            'Deferred cleanup is not the authenticated owned original-node job')
    lines = [line.strip() for line in script.splitlines() if line.strip() and not line.lstrip().startswith('#')]
    require(len(lines) == 1, 'Unexpected deferred cleanup batch script')
    command = shlex.split(lines[0])
    require(len(command) == 8 and command[0] == 'env' and command[1] == 'SEARCH_G1_BASE='+str(source) and
            command[2].startswith('PYTHONPATH='+str(source/'source')+'/') and
            command[3:] == [CONTROL_PYTHON, '-m', 'reproduction.hpc3_dynrpo_periodic.reconcile_cleanup',
                            '--failed-job', failed_job],
            'Deferred cleanup batch command does not match the owned reconciler')
    # The exact source file is immutable and independently published; it need
    # not be the failed trainer's older commit.
    root = Path(command[2].split('=', 1)[1])
    require(re.fullmatch(r'[0-9a-f]{40}', root.name) is not None and root.resolve() == root and
            root.stat().st_uid == uid and
            (root/'reproduction/hpc3_dynrpo_periodic/reconcile_cleanup.py').is_file(),
            'Deferred cleanup source is not an owned immutable deployed commit')
    return dict(job=dict(job), batch_script_sha256=hashlib.sha256(script.encode()).hexdigest(),
                source=str(root), reconciler_sha256=sha(root/'reproduction/hpc3_dynrpo_periodic/reconcile_cleanup.py'))


def deferred_source(source, failed_job, cleanup_job, rows, run):
    from reproduction.common8.native_allocation import fields
    username = pwd.getpwuid(os.getuid()).pw_name
    parsed = [row.split('|') for row in rows]
    parents = [row for row in parsed if row[0] == failed_job]
    extern = [row for row in parsed if row[0] == failed_job+'.extern']
    require(len(parents) == len(extern) == 1 and parents[0][1:4] == ['FAILED', '1:0', username] and
            extern[0][1:3] == ['COMPLETED', '0:0'],
            'Deferred retry requires actual FAILED1:0 and released extern allocation')
    allocation = read(source/'logs/native-allocation-admission.json')
    node = parents[0][4]
    require(re.fullmatch(r'ACD1-[0-9]+', node) is not None and
            allocation.get('slurm_job_id') == failed_job and allocation.get('run_id') == run['run_id'] and
            allocation.get('node') == node and allocation.get('accepted') is True,
            'Deferred retry original allocation identity changed')
    status = read(source/'logs/pipeline-status.json')
    require(status.get('run_id') == run['run_id'] and status.get('stage') == 'OWNED_CLEANUP_FAILED',
            'Deferred retry requires the original owned cleanup failure')
    ledger = read(source/'logs/process-ledger.json')
    validate_pretraining_ledger(ledger, source, run['run_id'], os.getuid(),
                               (source/'logs/initial-calibration.log').read_text())
    job = fields(subprocess.check_output(['scontrol', 'show', 'job', cleanup_job, '-o'], text=True))
    script = subprocess.check_output(['scontrol', 'write', 'batch_script', cleanup_job, '-'], text=True)
    proof = validate_cleanup_job(job, script, cleanup_job=cleanup_job, failed_job=failed_job,
        source=source, old_node=node, username=username, uid=os.getuid())
    proof.update(kind=DEFERRED_KIND, source_base=str(source), source_job=failed_job,
        source_run_id=run['run_id'], source_node=node, source_terminal=parents[0],
        source_extern_terminal=extern[0], cleanup_job=cleanup_job, cleaned=False,
        source_run_manifest_sha256=sha(source/'logs/run-manifest.json'),
        final_acceptance_requires_actual_source_cleanup=True)
    return proof, ledger


def cpu_probe_readback(path):
    import torch
    from search_g1.qprobe_periodic_model import restore_engine
    from search_g1.full_state_restore import equal_tree
    saved = torch.load(path, map_location='cpu', weights_only=True)
    engine = restore_engine(saved['head'], device='cpu', epochs=20)
    equal_tree(engine.state_dict(), saved['head'])
    require(engine._adam_steps() == 400, 'Deferred retry actual CPU Adam readback is not400')
    return dict(actual_saved_model_adam_rng_readback=True, head_adam_steps=400, device='cpu')


def certificate(base, commit):
    """Pure-CPU immutable certificate binding, reused by submission and bind."""
    path = Path(base)/'preparation/zero-update-retry.json'
    if not path.exists():
        return None
    record = read(path)
    proof = record.get('deferred_cleanup')
    if proof is None:
        return None
    require(proof.get('kind') == DEFERRED_KIND and proof.get('target_base') == str(base) and
            proof.get('target_source_commit') == commit and proof.get('cleaned') is False and
            proof.get('final_acceptance_requires_actual_source_cleanup') is True and
            record.get('student_updates_before_failure') == 0 and record.get('inherited_probe_adam_steps') == 400,
            'Deferred retry certificate target or zero-update binding changed')
    publication = read(Path(base)/'preparation/source-publication.json')
    require(publication.get('commit') == commit and publication.get('pushed_and_fetched') is True and
            sha(Path(base)/'preparation/source-publication.json') == proof['target_publication_sha256'],
            'Deferred retry source publication changed')
    snapshot = Path(proof['source_ledger_snapshot'])
    require(snapshot == Path(base)/'preparation/deferred-source-process-ledger.json' and
            sha(snapshot) == proof['source_ledger_sha256'], 'Deferred retry saved source ledger changed')
    require(proof.get('source_base') == record.get('source_base') and
            proof.get('source_job') == record.get('source_job') and
            proof.get('source_run_id') == record.get('source_run_id') and
            re.fullmatch(r'ACD1-[0-9]+', proof.get('source_node', '')) is not None and
            record.get('inherited_probe_sha256') == proof.get('head_sha256') and
            proof.get('cpu_probe_readback', {}).get('actual_saved_model_adam_rng_readback') is True,
            'Deferred retry source or preserved probe binding changed')
    for name in ('continued-head.pt',):
        require(sha(Path(base)/'warmup'/name) == proof['head_sha256'], 'Deferred retry probe bytes changed')
    return proof


def excluded_nodes(base, commit):
    proof = certificate(base, commit)
    return sorted({'ACD1-21'} | ({proof['source_node']} if proof else set()))


def validate_retry_allocation(job, hostname, nodes, exclusions, *, job_id, uid, username, required):
    require(job.get('JobId') == job_id and job.get('JobState') == 'RUNNING' and
            job.get('UserId') == username+'('+str(uid)+')' and job.get('NumNodes') == '1' and
            len(nodes) == 1 and hostname.split('.')[0] == nodes[0] and
            not set(nodes).intersection(required) and set(required).issubset(exclusions),
            'Deferred retry allocation reached an excluded or unauthenticated node')


def guard_allocation(base, commit):
    proof = certificate(base, commit)
    if proof is None:
        return
    from reproduction.common8.native_allocation import fields
    job_id = os.environ.get('SLURM_JOB_ID', '')
    require(job_id.isdecimal(), 'Deferred retry requires an actual Slurm job')
    job = fields(subprocess.check_output(['scontrol', 'show', 'job', job_id, '-o'], text=True))
    def hosts(value):
        return subprocess.check_output(['scontrol', 'show', 'hostnames', value], text=True).split()
    validate_retry_allocation(job, socket.gethostname(), hosts(job.get('NodeList', '')),
        hosts(job.get('ExcNodeList', '')), job_id=job_id, uid=os.getuid(),
        username=pwd.getpwuid(os.getuid()).pw_name, required=excluded_nodes(base, commit))


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-base',required=True,type=Path)
    p.add_argument('--job',required=True);p.add_argument('--deferred-cleanup-job');args=p.parse_args();own(BASE)
    source=args.source_base
    pattern=re.escape(str(DEFAULT_BASE))+r'(?:-r[1-9][0-9]*)?'
    if (source==BASE or source.resolve()!=source or source.is_symlink() or
            re.fullmatch(pattern,str(source)) is None or source.stat().st_uid!=os.getuid()):
        raise ValueError('Pretraining retry must preserve an owned independent prior execution')
    record=read(source/'preparation/formal-submission.json')
    if record['job']!=args.job or not args.job.isdecimal():
        raise ValueError('Source job identity differs')
    rows=subprocess.check_output(['sacct','-nP','-X','-j',args.job,'--format=JobID,State,ExitCode,User'],text=True).strip().splitlines()
    if len(rows)!=1 or rows[0].split('|')[0]!=args.job or rows[0].split('|')[1].split()[0] not in ('FAILED','OUT_OF_MEMORY','TIMEOUT'):
        raise ValueError('Source is not an authenticated failed terminal execution')
    if rows[0].split('|')[3] != pwd.getpwuid(os.getuid()).pw_name:
        raise ValueError('Source failed job belongs to another account')
    active=subprocess.check_output(['squeue','-h','-u',str(os.getuid()),'-o','%i'],text=True).split()
    if args.job in active:
        raise ValueError('Source job is still active')
    with contextlib.ExitStack() as stack:
        # Keep the real source writers excluded through evidence and hardlink
        # adoption. Opening these existing own lockfiles does not touch assets.
        for name in SOURCE_LOCKS:
            lock=stack.enter_context((source/name).open('a'))
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        target_lock=stack.enter_context((BASE/'preparation/zero-update-retry.lock').open('a'))
        fcntl.flock(target_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        pristine_student(source)
        source_run=read(source/'logs/run-manifest.json')
        cleanup_path=source/'logs/cleanup-status.json'
        cleanup=read(cleanup_path) if cleanup_path.exists() else {}
        deferred=None
        if cleanup.get('cleaned') is not True:
            require(args.deferred_cleanup_job is not None and args.deferred_cleanup_job.isdecimal(),
                    'Prior owned processes have not been reconciled')
            publication=read(BASE/'preparation/source-publication.json')
            verify_source(publication['commit'])
            detailed=subprocess.check_output(['sacct','-nP','-j',args.job,
                '--format=JobID,State,ExitCode,User,NodeList'],text=True).strip().splitlines()
            deferred,ledger=deferred_source(source,args.job,args.deferred_cleanup_job,detailed,source_run)
            snapshot=BASE/'preparation/deferred-source-process-ledger.json'
            if snapshot.exists():
                require(read(snapshot)==ledger, 'A different deferred source ledger was already preserved')
            else:
                publish(snapshot,ledger,once=True)
            deferred.update(source_ledger_snapshot=str(snapshot),source_ledger_sha256=sha(snapshot),
                target_base=str(BASE),target_source_commit=read(BASE/'preparation/source-publication.json')['commit'],
                target_publication_sha256=sha(BASE/'preparation/source-publication.json'))
        else:
            require(cleanup.get('run_id')==source_run['run_id'], 'Prior cleanup belongs to a different execution')
        ready=read(source/'warmup/ready.json');original=source/'warmup/continued-head.pt'
        if ready['head_adam_steps']!=400 or sha(original)!=ready['head_sha256']:
            raise ValueError('The accepted existing probe was not preserved intact')
        if deferred is not None:
            deferred.update(cpu_probe_readback=cpu_probe_readback(original),head_sha256=ready['head_sha256'])
            require(sha(original)==ready['head_sha256'],'Source probe changed during CPU readback')
        for name in ('continued-head.pt','ready.json','warmup-metrics.json'):
            target=BASE/'warmup'/name;origin=source/'warmup'/name
            if target.exists():
                if sha(target)!=sha(origin):raise ValueError('Retry warmup reference differs')
            elif name.endswith('.pt'):
                os.link(origin,target)  # Same remote filesystem; no tensor copy or reset.
            else:
                shutil.copyfile(origin,target)
        publish(BASE/'preparation/zero-update-retry.json',dict(source_base=str(source),source_job=args.job,
            actual_source_terminal=rows[0],source_run_id=source_run['run_id'],
            student_updates_before_failure=0,inherited_probe_adam_steps=400,
            inherited_probe_sha256=ready['head_sha256'],probe_optimizer_reset=False,
            **(dict(deferred_cleanup=deferred) if deferred is not None else {}),
            scope='retry before the first student update; not a training checkpoint resume',checked_unix=time.time()),once=True)
    print(json.dumps(dict(stage='ZERO_STUDENT_UPDATE_RETRY_PROBE400_PRESERVED',base=str(BASE))))


if __name__=='__main__':main()
