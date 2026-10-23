"""Retry a terminal allocation failure before any actual actor update."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time

from .deploy import launch_command, publish


def ensure_no_actor_writes(base):
    """Read the real four-rank journal before allowing a fresh retry.

    A first outer can write Adam before any driver update or checkpoint is
    published. Missing startup journals are valid; an unreadable journal or
    any persisted write evidence cannot establish a pre-actor boundary.
    """
    for rank in range(4):
        path=Path(base)/'logs'/('optimizer-rank-%d.jsonl'%rank)
        if not path.exists():
            continue
        try:
            with path.open() as stream:
                for line_number,line in enumerate(stream,1):
                    if not line.strip():
                        continue
                    row=json.loads(line)
                    if not isinstance(row,dict):
                        raise ValueError('Journal row is not a JSON object')
                    kind=row.get('kind')
                    wrote=kind=='OPTIMIZER_WRITE'
                    if kind=='OPPORTUNITY':
                        succeeded=row['optimizer_write_succeeded']
                        if type(succeeded) is not bool:
                            raise ValueError('Opportunity write result is not binary')
                        wrote=wrote or succeeded
                    if kind in ('OPPORTUNITY','OUTER_BUDGET'):
                        counters=('optimizer_writes','cumulative_writes','optimizer_writes_before','optimizer_writes_after')
                        for key in counters:
                            if key in row:
                                if type(row[key]) is not int or row[key]<0:
                                    raise ValueError('Invalid actual optimizer counter '+key)
                                wrote=wrote or row[key]>0
                        if kind=='OUTER_BUDGET' and not any(key in row for key in counters):
                            raise ValueError('Outer budget has no actual optimizer count')
                    if wrote:
                        raise RuntimeError('Use full-state resume after actual actor Adam write: %s:%d'%(path,line_number))
        except (OSError,UnicodeError,ValueError,KeyError) as error:
            raise RuntimeError('Cannot prove pre-actor retry from damaged journal '+str(path)) from error


def retry(base,commit,*,exclude):
    base=Path(base)
    previous=(base/'job.id').read_text().strip()
    state=subprocess.check_output(['sacct','-X','-n','-P','-j',previous,'--format=State'],text=True).strip()
    if not state.startswith(('FAILED','NODE_FAIL','CANCELLED','TIMEOUT','OUT_OF_MEMORY')):
        raise RuntimeError('Prior allocation is not confirmed terminal: '+state)
    if (base/'logs/actual-updates.jsonl').exists() or (base/'checkpoints').exists():
        raise RuntimeError('Use full-state resume after actor training starts')
    ensure_no_actor_writes(base)
    archive=base/'attempts'/previous
    archive.mkdir(parents=True,exist_ok=False)
    moves=['logs','job.id','preparation/train-submit-claim.json','preparation/train-submission.json',
           'preparation/formal-inline-method.json','preparation/dynrpo7b-configured.json',
           'data/calibration-300.parquet','data/audit.json',
           # Reward publication is once-only, including a pre-actor attempt.
           # Keep its complete pointer in the retired attempt before rebinding.
           'artifacts/live']
    for name in moves:
        source=base/name
        if source.exists():
            target=archive/name;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.move(str(source),str(target))
    (base/'logs').mkdir()
    command=launch_command(base,commit,'train',partition='acd_u',qos='normal',account='mou176',
        external_retriever_receipt=base/'preparation/external-retriever.json')
    command.insert(2,'--exclude='+exclude)
    publish(base/'preparation/train-submit-claim.json',dict(command=command,source_commit=commit,
        retry_of=previous,previous_terminal_state=state,excluded_node=exclude,
        actual_actor_updates_before_retry=0,created_unix=time.time()))
    result=subprocess.run(command,capture_output=True,text=True,timeout=30)
    value=dict(command=command,source_commit=commit,retry_of=previous,returncode=result.returncode,
        stdout=result.stdout,stderr=result.stderr,checked_unix=time.time())
    if result.returncode==0:
        job=result.stdout.strip().split(';')[0]
        if not job.isdecimal():raise RuntimeError('Ambiguous submission; inspect retained claim')
        (base/'job.id').write_text(job+'\n');value['job']=job
    publish(base/'preparation/train-submission.json',value)
    if result.returncode:raise RuntimeError(result.stderr)
    return value


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--commit',required=True)
    p.add_argument('--exclude',required=True);a=p.parse_args()
    print(json.dumps(retry(a.base,a.commit,exclude=a.exclude)),flush=True)
