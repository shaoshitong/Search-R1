"""Verify a finished HPC3 Slurm cgroup without interpreting recycled PIDs."""
import os
import re
import subprocess

TERMINAL = frozenset(('COMPLETED', 'FAILED', 'CANCELLED', 'TIMEOUT', 'OUT_OF_MEMORY',
                      'NODE_FAIL', 'PREEMPTED', 'BOOT_FAIL', 'DEADLINE', 'REVOKED'))


def writers_present(terminal, config):
    job = str(config.get('slurm_job', ''))
    if (config.get('execution_backend') != 'slurm_hpc3' or not re.fullmatch(r'[0-9]+', job)
            or terminal.get('execution_backend') != 'slurm_hpc3'
            or terminal.get('slurm_job') != job or terminal.get('uid') != os.getuid()):
        raise ValueError('Require this account and the exact original HPC3 Slurm job')
    queue = subprocess.run(['squeue', '-h', '-j', job, '-o', '%i|%T'],
                           capture_output=True, text=True, timeout=20)
    if queue.stdout.strip():
        raise RuntimeError('Original Slurm job is still present in the live queue')
    if queue.returncode and 'Invalid job id specified' not in queue.stderr:
        raise RuntimeError('Slurm queue absence could not be verified')
    raw = subprocess.check_output(['sacct', '-nP', '-j', job,
        '--format=JobIDRaw,UID,State,ExitCode,NodeList'], text=True, timeout=20)
    rows = [line.split('|') for line in raw.splitlines() if line.strip()]
    primary = [row for row in rows if row[0] == job]
    if (len(primary) != 1 or len(primary[0]) != 5 or primary[0][1] != str(os.getuid())
            or primary[0][4] != config.get('slurm_node')):
        raise RuntimeError('Original job/account/node accounting is missing or differs')
    if not {job + '.batch', job + '.extern'} <= {row[0] for row in rows}:
        raise RuntimeError('Require terminal batch and extern/cgroup accounting')
    if any(len(row) != 5 or row[2].split()[0].rstrip('+') not in TERMINAL for row in rows):
        raise RuntimeError('Original Slurm allocation or a step has not reached a terminal state')
    return []
