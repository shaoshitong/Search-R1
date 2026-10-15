"""Reconcile only a failed owned execution, on its actual original compute node.

The caller uses a new CPU-only Slurm allocation on that node. Login-node /proc
and /dev/shm observations cannot establish cleanup of a compute-node execution.
"""
import argparse
import json
import os
from pathlib import Path
import pwd
import socket
import subprocess
import time

from .common import BASE, own, read, publish


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--failed-job',required=True)
    args=parser.parse_args()
    own(BASE)
    from reproduction.common8.native_allocation import fields
    from reproduction.a800.owned_processes import OwnedProcesses, same, identity
    from reproduction.common8.run_pipeline import gpu_rows
    from reproduction.common8.temporary_cache import cleanup as clean_cache
    from reproduction.common8.stage_retrieval import cleanup as clean_stage
    run=read(BASE/'logs/run-manifest.json')
    allocation=read(BASE/'logs/native-allocation-admission.json')
    old=fields(allocation['actual_allocation_detail'])
    failed=args.failed_job
    assert failed.isdecimal() and old['JobId']==allocation['slurm_job_id']==failed
    assert allocation['run_id']==run['run_id']
    assert read(BASE/'preparation/formal-submission.json')['job']==failed
    username=pwd.getpwuid(os.getuid()).pw_name
    rows=subprocess.check_output(['sacct','-nP','-X','-j',failed,
        '--format=JobID,State,ExitCode,User,NodeList'],text=True).strip().splitlines()
    assert len(rows)==1
    row=rows[0].split('|')
    assert row[0]==failed and row[1].split()[0] in ('FAILED','OUT_OF_MEMORY','TIMEOUT','CANCELLED')
    assert row[3]==username and row[4]==old['NodeList']
    current_id=os.environ['SLURM_JOB_ID']
    assert current_id.isdecimal() and current_id!=failed
    current=fields(subprocess.check_output(['scontrol','show','job',current_id,'-o'],text=True))
    assert current['JobId']==current_id and current['JobState']=='RUNNING'
    assert current['UserId']==username+'('+str(os.getuid())+')'
    host=socket.gethostname().split('.')[0]
    before_nodes=subprocess.check_output(['scontrol','show','hostnames',old['NodeList']],text=True).split()
    current_nodes=subprocess.check_output(['scontrol','show','hostnames',current['NodeList']],text=True).split()
    assert len(before_nodes)==1 and host==before_nodes[0].split('.')[0]
    assert host in [node.split('.')[0] for node in current_nodes]
    assert BASE.stat().st_uid==os.getuid()
    ledger=OwnedProcesses(BASE/'logs/process-ledger.json',run['run_id'])
    assert ledger.recovery_only
    ledger.stop_all()
    # Re-read every stopped session too; a prior STOPPED marker alone is not proof.
    assert all(not ledger.observe(name) for name in ledger.jobs)
    records=[record for job in ledger.jobs.values() for record in job['ledger'].values()]
    assert all(not same(record) or identity(record['pid'])['state']=='Z' for record in records)
    if (BASE/'logs/corpus-cache-claim.json').exists():clean_cache(BASE)
    if (BASE/'logs/retrieval-staging-claim.json').exists():clean_stage(BASE)
    recorded={record['pid'] for record in records}
    deadline=time.monotonic()+30
    while True:
        remaining=[r for r in gpu_rows('gpu_uuid,pid','compute-apps') if int(r[1]) in recorded]
        if not remaining:break
        assert time.monotonic()<deadline,'Recorded CUDA contexts have not released'
        time.sleep(.5)
    proof=dict(run_id=run['run_id'],cleaned=True,checked_unix=time.time(),
        execution_hostname=host,failed_job=failed,actual_source_terminal=rows[0],
        cleanup_job_id=current_id,cleanup_actual_allocation=current,
        original_allocation_hostname_verified=True,all_owned_sessions_absent=True,
        recorded_cuda_contexts_absent=True,source_assets_deleted=False,
        scope='Original failed job resources only; no training or tensor restoration')
    publish(BASE/'logs/cleanup-status.json',proof)
    print(json.dumps(proof))


if __name__=='__main__':main()
