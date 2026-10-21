"""One human-authorized SIGUSR1 to the exact745501 Python writer only."""
import base64
import json
from pathlib import Path
import shlex
import subprocess
import time

base=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/joint-balanced-native-E4-124194-v1')
config=base/'config.json'
import hashlib
assert hashlib.sha256(config.read_bytes()).hexdigest()=='5aaa63c9348f410357be92ce491ba6d18286d618ab86dae6e63f193f2b4367b6'
c=json.loads((base/'controller.json').read_bytes())
assert c['pid']==300719 and c['job_id']=='745501'
job=subprocess.check_output(['scontrol','show','job','-o','745501'],universal_newlines=True)
for token in ['JobState=RUNNING','UserId=mou176(21930)','Account=mou176','WorkDir='+str(base),'Command='+str(base/'training.sbatch')]:
    assert token in job,token
body='''import os,json,signal,pathlib,time
p=pathlib.Path('/proc/300719');st=p.joinpath('stat').read_text().rsplit(')',1)[1].split();assert st[19]=='116543024'
assert p.stat().st_uid==21930
cmd=p.joinpath('cmdline').read_bytes();assert b'joint_balanced_native_main'in cmd and b'/joint-balanced-native-E4-124194-v1'in cmd
assert b'SLURM_JOB_ID=745501'in p.joinpath('environ').read_bytes().split(bytes([0]))
assert os.environ.get('SLURM_JOB_ID')=='745501'
os.kill(300719,signal.SIGUSR1)
print(json.dumps(dict(status='EXACT_OWNED_SOURCE_PYTHON_SIGUSR1_REQUESTED_NOT_YET_CHECKPOINTED',pid=300719,birth_ticks=st[19],job_id='745501',signal='SIGUSR1',epoch=time.time(),no_batch_or_foreign_signal=True)))
'''
args=['srun','--overlap','--jobid=745501','--account=mou176','--nodes=1','--ntasks=1','--cpus-per-task=1','--mem=1G','--gres=gpu:0','python3','-c',body]
x=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=25)
if x.returncode:raise RuntimeError(x.stdout[-1500:]+x.stderr[-1500:])
print(x.stdout.strip())
