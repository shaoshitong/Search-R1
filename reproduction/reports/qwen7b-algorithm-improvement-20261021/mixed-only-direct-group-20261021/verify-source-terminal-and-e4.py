"""Stream SHA only: verify authorized stop and preserve the prior E4 source."""
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import time

root=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019')
base=root/'algorithm-improvement-20261021/joint-balanced-native-E4-124194-v1'
def read(p):return json.loads(Path(p).read_bytes())
def proof(p):
    p=Path(p);h=hashlib.sha256()
    with p.open('rb')as f:
        for b in iter(lambda:f.read(16*1024**2),b''):h.update(b)
    return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())
row=subprocess.check_output(['sacct','-n','-P','-j','745501','--format=JobIDRaw,State,ExitCode'],universal_newlines=True)
assert '745501|COMPLETED|0:0' in row
latest,progress,intent,stop=[read(base/(n+'.json'))for n in ('latest','progress','update-intent','interrupted')]
assert latest['cursor']==progress['cursor']==intent['cursor']==stop['cursor']==1555
assert intent['phase']=='accepted_update_complete'and stop['checkpoint']==latest
assert proof(latest['path'])=={k:latest[k]for k in ('path','bytes','sha256')}
lock=(base/'training.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not(base/'failure.json').exists()
source=proof(root/'balanced-rank1-from-epoch1-prefix20-20261020/probe/epoch-0004.pt')
assert source['bytes']==3745810528 and source['sha256']=='e29168c36de152a57294f055e7d420cbba25cae31aeace32e793ba1098bf97a8'
print(json.dumps(dict(status='HUMAN_PIVOT_SOURCE_STOPPED_EXACT_CHECKPOINT_AND_PRIOR_E4_FRESH_SHA_PASSED',
    epoch=time.time(),old_job='745501',old_exit='COMPLETED_0_0',old_accepted_updates=1555,
    old_progress_equals_checkpoint_equals_intent_equals_interrupted=True,
    source_training_lock_free=True,old_checkpoint=latest,new_training_source=source,
    old_checkpoint_is_archive_only=True,new_native_block_Adam_source_steps=14418,
    new_group_head_random_initialization_only=True,model_or_tensor_loaded=False)))
