"""Explicit direct H100 execution identity; never borrow another Slurm job."""
import os
import socket


def shared_h100(environ=None):
    env = os.environ if environ is None else environ
    value = env.get('RBA_H100_SHARED_EXECUTION_ID', '')
    if not value:
        return False
    if (not value.startswith('shared-h1005-ablation150-')
            or socket.gethostname().split('.')[0] != 'h100-5'
            or os.getuid() != 0 or env.get('SLURM_JOB_ID')):
        raise ValueError('Shared H100 identity must be this authorized h100-5 direct process')
    return True


def execution_id():
    if shared_h100():
        return os.environ['RBA_H100_SHARED_EXECUTION_ID']
    return os.environ['SLURM_JOB_ID']
