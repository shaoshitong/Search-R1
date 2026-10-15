"""Scoped old-loss stop, terminal archive, and once-only fresh std-loss submit.

Run phases separately: request-stop, capture-terminal, submit. A timeout or an
existing intent never causes a second submission or an unrelated process kill.
"""
import argparse
import hashlib
import json
import os
import re
from pathlib import Path
import subprocess
import time

OLD = Path('/mnt/data/shaoshitong/search-g1-qprobe3-h1008-003a0b0-20261015-v3')
JOB = 3397


def read(p):return json.loads(p.read_text())
def once(p,v):
    with p.open('x') as f:json.dump(v,f,indent=2);f.flush();os.fsync(f.fileno())
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for chunk in iter(lambda:f.read(8388608),b''):h.update(chunk)
    return h.hexdigest()
def live(row):
    p=Path('/proc')/str(row['pid'])/'stat'
    try:return p.read_text().rsplit(')',1)[1].split()[19]==str(row['start_ticks'])
    except FileNotFoundError:return False

def validate(base):
    config=read(base/'preparation/h1008-launch-config.json')
    assert config['superseded_job_id']==JOB and config['superseded_training_base']==str(OLD)
    assert config['restore_receipt'] is None
    assert config['new_loss']==dict(loss_version='std_class_ce_v1',std_loss_weight=1.,count_aux_loss_weight=.1,mean_loss_weight=0.)
    assert read(OLD/'logs/run-manifest.json')['run_id']=='4e2fd299bf5e4f3e86d910661f6389b9'
    for name,digest in read(base/'preparation/source-files.json').items():
        assert sha(base/'repo'/name)==digest,name
    return config


def request_stop(base):
    validate(base);p=base/'preparation/std-loss-stop-intent.json'
    if p.exists():return dict(existing_intent=True,action='inspect same job; no repeated stop')
    assert live(dict(pid=1142827,start_ticks=2298896953))
    native=subprocess.check_output(['scontrol','show','job','-o',str(JOB)],text=True)
    assert 'JobState=RUNNING' in native and 'NodeList=h100-8' in native
    paths=[x for x in (OLD/'logs').rglob('*') if x.is_file() and not x.is_symlink()]
    before=[dict(path=str(x),inode=x.stat().st_ino,device=x.stat().st_dev) for x in paths]
    once(p,dict(job_id=JOB,scope='only old-loss job3397; goal remains active; preserve unrelated tasks and backup CPU worker',
        authority='Latest human goal: replace probe loss and fresh official actor/ref200; no inherited old training state',
        quoted_instruction='可能要重新跑，主要是损失进行更换',requested_unix=time.time(),driver=dict(pid=1142827,start_ticks=2298896953),original_log_inodes=before))
    r=subprocess.run(['scancel','--signal=TERM','--batch',str(JOB)],capture_output=True,text=True,timeout=30)
    once(base/'preparation/std-loss-stop-receipt.json',dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,checked_unix=time.time()))
    assert r.returncode==0,r.stderr
    return dict(stop_requested=True,job_id=JOB,goal_paused=False)


def capture_terminal(base):
    validate(base);p=base/'preparation/std-loss-old-terminal.json'
    if p.exists():return read(p)
    intent=read(base/'preparation/std-loss-stop-intent.json')
    ledger=read(OLD/'logs/process-ledger.json');births=[r for job in ledger['jobs'].values() for r in job['ledger'].values()]
    assert births and not [r for r in births if live(r)],'Original owned births still live; observe same job'
    assert not live(intent['driver'])
    accounting=subprocess.run(['sacct','-j',str(JOB),'-n','-P','-o','JobIDRaw,State,ExitCode'],capture_output=True,text=True)
    if accounting.returncode==0:
        rows=[r.split('|') for r in accounting.stdout.splitlines() if r.split('|')[0]==str(JOB)]
        assert len(rows)==1,accounting.stdout
    else:
        # This cluster disables Slurm accounting; controller state is authoritative.
        native=subprocess.check_output(['scontrol','show','job','-o',str(JOB)],text=True)
        fields=dict(re.findall(r'(\w+)=(\S+)',native))
        assert fields['JobId']==str(JOB) and fields['BatchHost']=='h100-8'
        rows=[[str(JOB),fields['JobState'],fields['ExitCode']]]
    assert rows[0][1].split()[0] in ('CANCELLED','FAILED','COMPLETED','TIMEOUT','NODE_FAIL','OUT_OF_MEMORY'),rows
    groups=list(Path('/sys/fs/cgroup').glob('**/job_'+str(JOB)));assert not groups,'Owned cgroup still exists'
    gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader'],text=True)
    owned={r['pid'] for r in births};assert not any(int(r.split(',')[0]) in owned for r in gpu.splitlines() if r.strip())
    original=[]
    for row in intent['original_log_inodes']:
        f=Path(row['path']);st=f.stat()
        same=st.st_ino==row['inode'] and st.st_dev==row['device']
        # Append-only raw logs must retain their original inode. JSON receipts
        # such as process-ledger are atomically replaced during normal cleanup.
        if f.suffix in ('.log','.out','.err'):assert same,str(f)
        original.append(dict(row,terminal_inode=st.st_ino,terminal_device=st.st_dev,
            same_original_inode=same,bytes=st.st_size,sha256=sha(f)))
    # Full raw log hashes and manifests stay remote; return a compact receipt.
    once(base/'preparation/std-loss-original-terminal-log-shas.json',original)
    result=dict(job_id=JOB,native_state=rows[0][1],native_exit_code=rows[0][2],original_owned_birth_count=len(births),original_births_live=0,
        cgroup_released=True,owned_gpu_contexts_released=True,original_log_files=len(original),original_log_hash_manifest_sha256=sha(base/'preparation/std-loss-original-terminal-log-shas.json'),
        actor_updates_inherited=False,checked_unix=time.time())
    once(p,result);return result


def submit_fresh(base):
    validate(base);terminal=capture_terminal(base)
    for name in ('submit-intent.json','submit-receipt.json'):
        if (base/'preparation'/name).exists():return dict(existing_submission=True,receipt=read(base/'preparation/submit-receipt.json') if (base/'preparation/submit-receipt.json').exists() else None)
    assert not (base/'logs/run-manifest.json').exists()
    from reproduction.h1008_qprobe.submit import submit
    return submit(base,read(base/'preparation/source-files.json'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--base',type=Path,required=True);p.add_argument('phase',choices=['request-stop','capture-terminal','submit']);a=p.parse_args()
    base=a.base.resolve();assert str(base).startswith('/mnt/data/shaoshitong/search-g1-qprobe3-stdloss-')
    print(json.dumps({'request-stop':request_stop,'capture-terminal':capture_terminal,'submit':submit_fresh}[a.phase](base)))


if __name__=='__main__':main()
