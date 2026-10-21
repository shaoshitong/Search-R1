"""Single guarded pure-CPU metadata join; no GPU and no retired main reuse."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import time

b=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/joint169028-native-training-bank-v1')
jobname='q7b-full169028-bank-CPU'
def read(p):return json.loads(Path(p).read_bytes())
def proof(p):
    p=Path(p);raw=p.read_bytes()
    return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def once(p,x):
    p=Path(p);raw=x.encode()if isinstance(x,str)else(json.dumps(x,sort_keys=True)+'\n').encode()
    if p.exists():assert p.read_bytes()==raw,'Immutable metadata differs'
    else:
        with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
guard=(b/'cpu-submit.lock').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
if(b/'complete.json').exists():
    print(json.dumps(dict(status='FULL_BANK_ALREADY_COMPLETE',complete=proof(b/'complete.json'))));raise SystemExit(0)
if(b/'cpu-submission-result.json').exists():
    print(json.dumps(dict(status='ADOPT_PRIOR_CPU_JOIN',result=read(b/'cpu-submission-result.json'))));raise SystemExit(0)
active=subprocess.check_output(['squeue','-h','-u','mou176','-o','%i|%j|%T'],universal_newlines=True)
assert not any(v.split('|')[1]==jobname for v in active.splitlines())
assert not(b/'cpu-submission-intent.json').exists(),'Unknown prior submit; reconcile before any retry'
config=proof(b/'config.json');c=read(b/'config.json');pub=read(b/'source-publication.json')
assert c['questions']==169028 and c['target']==str(b)and len(c['banks'])==5
assert pub['commit']=='ba685344da51f9aaac0bd0d991195ce578fe1758'
code=Path(pub['code_root'])
for name,ref in pub['files'].items():
    p=proof(code/name);assert p['bytes']==ref['bytes']and p['sha256']==ref['sha256']
for item in c['banks']:assert proof(item['complete']['path'])==item['complete']
lock=(b/'join.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);lock.close()
script='''#!/usr/bin/env bash
#SBATCH --job-name=q7b-full169028-bank-CPU
#SBATCH --partition=debug
#SBATCH --account=mou176
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --gres=gpu:0
#SBATCH --time=00:15:00
#SBATCH --output=@@BASE@@/join-cpu-%j.log
#SBATCH --chdir=@@CODE@@
set -euo pipefail
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
exec python3 -u -m reproduction.operations.qwen7b_warmup.joint169028_bank --config @@CONFIG@@ --config-sha256 @@SHA@@
'''.replace('@@BASE@@',str(b)).replace('@@CODE@@',str(code)).replace('@@CONFIG@@',config['path']).replace('@@SHA@@',config['sha256'])
entry=b/'join-cpu.sbatch';once(entry,script);subprocess.check_call(['bash','-n',str(entry)])
partition=subprocess.check_output(['scontrol','show','partition','debug','-o'],universal_newlines=True)
assert 'State=UP'in partition
once(b/'cpu-submission-intent.json',dict(config=config,entry=proof(entry),epoch=time.time(),CPU=2,GPU=0))
r=subprocess.run(['sbatch','--hold','--parsable',str(entry)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=25)
if r.returncode or not re.fullmatch(r'\d+(?:;[^ ]+)?',r.stdout.strip()):
    raise RuntimeError('Submission unconfirmed; intent retained: '+r.stderr[-1200:])
job=r.stdout.strip().split(';')[0]
actual=subprocess.check_output(['scontrol','show','job','-o',job],universal_newlines=True)
for token in ['JobName='+jobname,'Account=mou176','Partition=debug','NumCPUs=2','MinMemoryNode=8G','TimeLimit=00:15:00','Dependency=(null)','Command='+str(entry),'WorkDir='+str(code)]:assert token in actual,token
assert not re.search(r'gres/gpu[=:][1-9]',actual),'Metadata join must allocate no GPU'
identity=dict(job_id=job,job=actual,config=config,entry=proof(entry),CPU=2,GPU=0,epoch=time.time())
once(b/'cpu-submission-identity.json',identity);subprocess.check_call(['scontrol','release',job])
result=dict(status='PURE_CPU_FULL169028_METADATA_JOIN_RELEASED',job_id=job,identity=proof(b/'cpu-submission-identity.json'),GPU=0,CPU=2)
once(b/'cpu-submission-result.json',result);print(json.dumps(result))
