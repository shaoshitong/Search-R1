"""Retry a terminal allocation failure before any actual actor update."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time

from .deploy import launch_command, publish


def retry(base,commit,*,exclude):
    base=Path(base)
    previous=(base/'job.id').read_text().strip()
    state=subprocess.check_output(['sacct','-X','-n','-P','-j',previous,'--format=State'],text=True).strip()
    if not state.startswith(('FAILED','NODE_FAIL','CANCELLED','TIMEOUT','OUT_OF_MEMORY')):
        raise RuntimeError('Prior allocation is not confirmed terminal: '+state)
    if (base/'logs/actual-updates.jsonl').exists() or (base/'checkpoints').exists():
        raise RuntimeError('Use full-state resume after actor training starts')
    archive=base/'attempts'/previous
    archive.mkdir(parents=True,exist_ok=False)
    moves=['logs','job.id','preparation/train-submit-claim.json','preparation/train-submission.json',
           'preparation/formal-inline-method.json','preparation/dynrpo7b-configured.json',
           'data/calibration-300.parquet','data/audit.json']
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
