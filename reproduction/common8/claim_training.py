"""Exclusive fresh-run claim; a failed attempt must never silently reuse its run ID."""
import json
import os
import socket
import time
from pathlib import Path
from reproduction.common8.topology import execution_job_id, validate_base


def claim(base):
    from search_g1.prefix20_restore_only import deny_training
    deny_training()
    base=Path(base)
    run=json.loads((base/'logs/run-manifest.json').read_text())
    validate_base(base,run)
    assert run['profile']=='common8' and run['completed_updates']==0
    assert not (base/'logs/training-launch-claim.json').exists(), 'This run has already been claimed'
    assert not list((base/'logs').glob('optimizer-rank-*.jsonl')), 'Existing actual write evidence'
    assert not list((base/'logs').glob('scheduler-rank-*.jsonl')), 'Existing partial scheduler evidence'
    checkpoints=Path(run['checkpoint_dir'])
    assert not checkpoints.exists() or not any(checkpoints.iterdir()), 'Existing training checkpoint path'
    assert not Path(run['train_log']).exists(), 'Existing trainer log; inspect the attempt instead'
    parent=os.getppid()
    stat=Path(f'/proc/{parent}/stat').read_text().rsplit(')',1)[1].split()
    payload=dict(run_id=run['run_id'],shell_pid=parent,shell_start_ticks=int(stat[19]),
        slurm_job_id=execution_job_id(run),claimed_unix=time.time(),
        semantics='exclusive launch claim, not evidence of a completed update')
    if run.get('full_state_restore'):
        payload.update(full_state_restore=run['full_state_restore'],
            semantics='exclusive new allocation restoring the declared logical experiment; no replay or fresh optimizer')
    if run.get('execution_backend')=='ssh_local':
        allocation=json.loads((base/'logs/native-allocation-admission.json').read_text())
        assert allocation['node']==run['execution_host']==socket.gethostname()
        payload.update(execution_backend='ssh_local',execution_id=run['execution_id'],
                       hostname=socket.gethostname())
    with (base/'logs/training-launch-claim.json').open('x') as f:
        json.dump(payload,f);f.flush();os.fsync(f.fileno())
    return payload


if __name__=='__main__':
    print(json.dumps(claim(os.environ['SEARCH_G1_BASE'])))
