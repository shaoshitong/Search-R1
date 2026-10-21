"""Inventory only four source-bound files of two known terminal probe trials."""
import fcntl
import hashlib
import json
from pathlib import Path
import stat
import subprocess
import time

A=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021')
main=A/'joint-balanced-native-E4-124194-v1';snapshot=A/'joint124194-current-snapshot-eval-v1'
def read(p):return json.loads(Path(p).read_bytes())
def proof(p):
    p=Path(p);h=hashlib.sha256()
    with p.open('rb')as f:
        for block in iter(lambda:f.read(16*1024**2),b''):h.update(block)
    return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())
sacct=subprocess.check_output(['sacct','-n','-P','-j','745501,745981','--format=JobIDRaw,State,ExitCode'],universal_newlines=True)
for job in ('745501','745981'):assert job+'|COMPLETED|0:0'in sacct
locks=[]
for path in (main/'training.lock',snapshot/'evaluation.lock'):
    f=path.open('a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(f)
latest=read(main/'latest.json');assert latest['cursor']==read(main/'progress.json')['cursor']==read(main/'interrupted.json')['cursor']==1555
refs=[read(main/'evaluations/epoch-00/complete.json')['checkpoint'],read(main/'first-cp-independent-audit.json')['checkpoint'],latest,read(snapshot/'complete.json')['pinned_checkpoint']]
files=[]
for ref in refs:
    p=Path(ref['path']);assert p.parent in (main/'checkpoints',snapshot/'inputs')and p.resolve()==p and p.suffix=='.pt'
    s=p.stat();assert stat.S_ISREG(s.st_mode)and s.st_uid==21930
    actual=proof(p);assert actual=={k:ref[k]for k in ('path','bytes','sha256')}
    files.append(dict(actual,inode=s.st_ino,nlink=s.st_nlink,mtime_ns=s.st_mtime_ns))
assert len(files)==len({f['path']for f in files})==4
out=A/'retired-probe-cleanup-mixed-only-20261021';out.mkdir(exist_ok=True)
plan=dict(status='FOUR_KNOWN_TERMINAL_PROBE_BINARY_FILES_VERIFIED_FOR_HUMAN_AUTHORIZED_CLEANUP',files=files,
    logical_bytes=sum(f['bytes']for f in files),old_jobs=['745501','745981'],
    official_model_and_all_rollout_feature_banks_excluded=True,other_threads_and_old_E4_original_excluded=True,
    latest_human_source='official_Qwen2.5_7B_a09a_block27_fresh_optimizers',epoch=time.time())
p=out/'plan.json'
if p.exists():
    old=read(p);assert old['files']==files;plan=old
else:p.write_text(json.dumps(plan,sort_keys=True)+'\n')
print(json.dumps(dict(plan=proof(p),summary=plan)))
