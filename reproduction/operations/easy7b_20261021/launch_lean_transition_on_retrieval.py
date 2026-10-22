"""Once-only detached CPU source audit inside the existing retrieval allocation.

The launcher returns after spawning its small local supervisor. The supervisor
owns one srun step and saves its PID, log and exit status. It does not stop the
producer/retriever, allocate another job or GPU, or retry an unknown result.
The existing published transition submit enforces the source terminal state.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

SOURCE = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SOURCE))

from reproduction.easy2560_joint4.control import own_base, publish

TARGET = '/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fa3-lean-r1-20261022'
ALLOCATION_JOB = '745198'


def paths(base):
    base = own_base(base)
    return dict(base=base, claim=base/'preparation/lean-transition-retrieval-launch-claim.json',
                launched=base/'preparation/lean-transition-retrieval-launched.json',
                worker=base/'preparation/lean-transition-retrieval-supervisor.json',
                step=base/'preparation/lean-transition-retrieval-srun-process.json',
                result=base/'preparation/lean-transition-retrieval-exit.json',
                log=base/'logs/lean-transition-retrieval-cpu-audit.log')


def process_identity(pid):
    value = dict(pid=pid)
    try:
        raw = Path('/proc')/str(pid)
        value.update(start_ticks=int((raw/'stat').read_text().rsplit(')', 1)[1].split()[19]),
                     boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(), uid=os.getuid())
    except (FileNotFoundError, ProcessLookupError):
        value['exited_before_identity_read'] = True
    return value


def source_arguments(source_base=None, source_job=None, source_commit=None, minimum_outer=None):
    values = (('source-base', source_base), ('source-job', source_job),
              ('source-commit', source_commit), ('minimum-outer', minimum_outer))
    return [argument for name, value in values if value is not None
            for argument in ('--'+name, str(value))]


def command(base, commit, authorization_path, allocation_job=ALLOCATION_JOB, *,
            source_base=None, source_job=None, source_commit=None, minimum_outer=None):
    base = own_base(base)
    # Explicitly use the prepared target's immutable runtime, not this newer
    # operator's source for the CPU tensor reader or the training submission.
    return ['srun', '--jobid='+str(allocation_job), '--overlap', '--exact', '--ntasks=1',
            '--cpus-per-task=4', '--mem=128G', '--gres=none', '--chdir='+str(base/'repo'),
            '--export=ALL,CUDA_VISIBLE_DEVICES=,OMP_NUM_THREADS=1,OPENBLAS_NUM_THREADS=1',
            str(base/'env/bin/python'), '-B', '-m',
            'reproduction.operations.easy7b_20261021.fa3_lean_transition', 'submit',
            '--base', str(base), '--commit', commit,
            '--authorization-json', str(authorization_path),
            *source_arguments(source_base, source_job, source_commit, minimum_outer)]


def existing(p):
    result = dict(status='EXISTING_ONCE_ONLY_LAUNCH_INSPECT_ACTUAL_NO_RETRY',
                  claim=str(p['claim']), log=str(p['log']), automatic_retry=False)
    for name in ('launched', 'worker', 'step', 'result'):
        if p[name].is_file():
            result[name] = json.loads(p[name].read_text())
    return result


def launch(base, commit, authorization_path=None, allocation_job=ALLOCATION_JOB, *,
           source_base=None, source_job=None, source_commit=None, minimum_outer=None):
    p = paths(base)
    authorization_path = Path(authorization_path) if authorization_path else (
        p['base']/'preparation/fastpath-transition-qualified-authorization.json')
    if p['claim'].exists() or p['result'].exists() or p['worker'].exists():
        return existing(p)
    p['log'].parent.mkdir(parents=True, exist_ok=True)
    runner = [sys.executable, '-B', str(Path(__file__).resolve()), 'worker',
              '--base', str(p['base']), '--commit', commit,
              '--authorization-json', str(authorization_path), '--allocation-job', str(allocation_job)]
    runner.extend(source_arguments(source_base, source_job, source_commit, minimum_outer))
    argv = command(p['base'], commit, authorization_path, allocation_job,
                   source_base=source_base, source_job=source_job,
                   source_commit=source_commit, minimum_outer=minimum_outer)
    publish(p['claim'], dict(created_unix=time.time(), base=str(p['base']), target_runtime_commit=commit,
        authorization_json=str(authorization_path), runner=runner, srun_command=argv,
        existing_allocation_job=str(allocation_job), cpu_memory_bytes=128*1024**3,
        requested_step_gpus=0, login_tensor_loading=False, producer_stopped_here=False,
        retrieval_service_restarted=False, automatic_retry=False))
    try:
        with p['log'].open('xb') as log:
            child = subprocess.Popen(runner, stdin=subprocess.DEVNULL, stdout=log,
                                     stderr=subprocess.STDOUT, start_new_session=True)
        value = dict(status='DETACHED_CPU_AUDIT_SUPERVISOR_STARTED',
                     supervisor=process_identity(child.pid), log=str(p['log']),
                     target_runtime_commit=commit, created_unix=time.time(),
                     actual_full_state_audit_complete=False, training_submitted=False,
                     automatic_retry=False)
        publish(p['launched'], value)
        return value
    except Exception as error:
        # A launch claim without a known successful result is retained. Never
        # manufacture a second srun step on another invocation.
        failure = dict(status='SUPERVISOR_LAUNCH_FAILED_OR_UNKNOWN', error_type=type(error).__name__,
                       error=str(error), checked_unix=time.time(), automatic_retry=False)
        if not p['result'].exists():
            publish(p['result'], failure)
        raise


def worker(base, commit, authorization_path=None, allocation_job=ALLOCATION_JOB, *,
           source_base=None, source_job=None, source_commit=None, minimum_outer=None):
    p = paths(base)
    if p['worker'].exists() or p['result'].exists():
        return existing(p)
    claim = json.loads(p['claim'].read_text())
    authorization_path = Path(authorization_path) if authorization_path else (
        p['base']/'preparation/fastpath-transition-qualified-authorization.json')
    argv = command(p['base'], commit, authorization_path, allocation_job,
                   source_base=source_base, source_job=source_job,
                   source_commit=source_commit, minimum_outer=minimum_outer)
    if claim['srun_command'] != argv:
        raise ValueError('Supervisor arguments differ from the existing launch claim')
    publish(p['worker'], dict(supervisor=process_identity(os.getpid()), command=argv,
                             started_unix=time.time(), automatic_retry=False))
    try:
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
        step = subprocess.Popen(argv, stdin=subprocess.DEVNULL, env=environment)
        publish(p['step'], dict(process=process_identity(step.pid), command=argv,
                               started_unix=time.time(), existing_allocation_job=str(allocation_job)))
        returncode = step.wait()
        result = dict(status='CPU_AUDIT_AND_TRANSITION_COMMAND_EXITED', returncode=returncode,
                      finished_unix=time.time(), log=str(p['log']), automatic_retry=False,
                      actual_cuda_restore_accepted=False,
                      submission_result_path=str(p['base']/'preparation/fa3-transition-submitted.json'))
    except Exception as error:
        result = dict(status='CPU_AUDIT_COMMAND_FAILED_OR_UNKNOWN', error_type=type(error).__name__,
                      error=str(error), finished_unix=time.time(), automatic_retry=False)
    publish(p['result'], result)
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('launch', 'worker'))
    parser.add_argument('--base', default=TARGET)
    parser.add_argument('--commit', required=True, help='Prepared target runtime commit, e.g. full 48b87a7 SHA')
    parser.add_argument('--authorization-json')
    parser.add_argument('--allocation-job', default=ALLOCATION_JOB)
    parser.add_argument('--source-base')
    parser.add_argument('--source-job')
    parser.add_argument('--source-commit')
    parser.add_argument('--minimum-outer', type=int)
    args = parser.parse_args(argv)
    result = (launch if args.mode == 'launch' else worker)(
        args.base, args.commit, args.authorization_json, args.allocation_job,
        source_base=args.source_base, source_job=args.source_job,
        source_commit=args.source_commit, minimum_outer=args.minimum_outer)
    print(json.dumps(result, sort_keys=True), flush=True)
    if args.mode == 'worker':
        return result.get('returncode', 1 if result['status'] == 'CPU_AUDIT_COMMAND_FAILED_OR_UNKNOWN' else 0)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
