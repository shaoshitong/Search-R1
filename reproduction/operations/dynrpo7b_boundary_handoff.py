"""One controlled stop of an owned run at its latest fully saved Adam boundary.

Run on HPC3 after staging the new immutable runtime. This never submits jobs,
changes a running checkout, resets counters or touches the retrieval service.
"""
import argparse
import json
from pathlib import Path
import subprocess
import time


def command(*args):
    p = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       universal_newlines=True, timeout=30)
    if p.returncode:
        raise RuntimeError(p.stderr or p.stdout)
    return p.stdout


def writes(base):
    result = []
    for rank in range(4):
        records = [json.loads(line) for line in (base/'logs'/('optimizer-rank-%d.jsonl' % rank)).read_text().splitlines()]
        counts = [row['write_count'] for row in records if row.get('kind') == 'OPTIMIZER_WRITE']
        result.append(counts[-1] if counts else 0)
    return result


def handoff(old, new, job, minimum):
    old, new = Path(old), Path(new)
    info = command('scontrol','show','job',str(job),'-o')
    if ('UserId=mou176(' not in info or 'JobState=RUNNING ' not in info or
        'Command='+str(old/'repo/reproduction/dynrpo7b/train.sh')+' ' not in info):
        raise ValueError('The running job does not match this handoff')
    source = json.loads((new/'preparation/run-source.json').read_text())
    if (new/'job.id').exists():
        raise ValueError('The destination already has a submission')
    latest = json.loads((old/'preparation/dynrpo7b-checkpoints/latest.json').read_text())
    step = latest['completed_outer']
    expected = [row['actual_optimizer_writes'] for row in sorted(latest['actor_ref']['worker_files'], key=lambda x:x['rank'])]
    if step < minimum or latest['status'] != 'DYNRPO7B_ROLLING_READY' or writes(old) != expected:
        return dict(status='WAIT_COMPLETE_BOUNDARY',completed_outer=step,actual_writes=writes(old))
    receipt = old/'preparation/dynrpo7b-checkpoints'/('global_step_%d' % step)/'boundary.json'
    if json.loads(receipt.read_text()) != latest:
        raise ValueError('Latest is not the actual immutable full boundary')
    record = dict(status='HANDOFF_AUTHORIZED_AT_FULL_BOUNDARY',old_job=job,old_base=str(old),new_base=str(new),
        source_commit=source['source_commit'],restore_receipt=str(receipt),completed_outer=step,
        actual_optimizer_writes_per_rank=expected,checked_unix=time.time(),
        authorization='2026-10-23 human request to fix recovery and snapshot reuse; subsequent continue. Original exact message time unavailable.',
        scope='Replace only this owned training job from its full state; retain original retrieval, checkout and all artifacts',
        committed_actor_replayed=False)
    audit=new/'preparation/repair-handoff.json'
    audit.write_text(json.dumps(record,indent=2)+'\n')
    command('scancel','--signal=STOP','--full',str(job))
    try:
        time.sleep(2)
        actual=writes(old)
        if actual != expected:
            command('scancel','--signal=CONT','--full',str(job))
            record.update(status='CONTINUED_NEW_UNSAVED_WRITES',actual_optimizer_writes_per_rank=actual)
            audit.write_text(json.dumps(record,indent=2)+'\n')
            return record
        command('scancel',str(job))
    except BaseException:
        # If cancellation did not happen, never leave an owned live job stopped.
        subprocess.run(['scancel','--signal=CONT','--full',str(job)],stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        raise
    record.update(status='OLD_JOB_CANCEL_REQUESTED_FULL_STATE_RETAINED',checked_unix=time.time())
    audit.write_text(json.dumps(record,indent=2)+'\n')
    return record


if __name__ == '__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--old',required=True);p.add_argument('--new',required=True)
    p.add_argument('--job',type=int,required=True);p.add_argument('--minimum',type=int,required=True)
    a=p.parse_args()
    print(json.dumps(handoff(a.old,a.new,a.job,a.minimum),sort_keys=True))
