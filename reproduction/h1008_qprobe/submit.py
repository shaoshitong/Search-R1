"""Once-only fresh H1008 submission after actual predecessor evaluation exit0."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
from reproduction.h1008_qprobe.adopt_reward_zero import digest


def validate_predecessor(result,cleanup,terminal):
    if result.get('qa_count')!=51713:raise ValueError('WAIT_FOR_ALL51713_FINAL_RESULTS')
    if cleanup.get('cleaned') is not True:raise ValueError('WAIT_FOR_OWNED_EVALUATION_RELEASE')
    if terminal.get('job_id')!=3395 or terminal.get('state')!='COMPLETED' or terminal.get('exit_code')!='0:0':raise ValueError('ACTUAL_PREDECESSOR_EXIT0_REQUIRED')
    return True


def once(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno())


def submit(base,source_files):
    base=Path(base).resolve();config=json.loads((base/'preparation/h1008-launch-config.json').read_text())
    if config['role']!='fresh_h1008_transformer3_question_probe200' or config.get('restore_receipt'):raise ValueError('FRESH_OFFICIAL_ONLY')
    if (base/'preparation/submit-intent.json').exists() or (base/'preparation/submit-receipt.json').exists():raise RuntimeError('Submission already attempted; query receipt/queue, never replay')
    predecessor=Path(config['predecessor_evaluation_base'])
    result=json.loads((predecessor/'journal/final-result.json').read_text());cleanup=json.loads((predecessor/'logs/cleanup-status.json').read_text())
    terminal=json.loads((base/'preparation/predecessor-exit0.json').read_text())
    validate_predecessor(result,cleanup,terminal)
    assert result['run_id']==cleanup['run_id']==config['predecessor_run_id']
    assert result['step']==200 and result['final'] is True
    assert result['checkpoint_manifest_sha256']==config['predecessor_checkpoint_manifest_sha256']
    assert result['validation_sha256']==config['predecessor_validation_sha256']
    for name,sha in source_files.items():
        if digest(base/'repo'/name)!=sha:raise ValueError('DEPLOYED_SOURCE_DIFFERS:'+name)
    if (base/'logs/run-manifest.json').exists():raise ValueError('FRESH_NAMESPACE_ALREADY_PREPARED_OR_STARTED')
    queue=subprocess.check_output(['squeue','-h','-w','h100-8','-t','RUNNING,PENDING,COMPLETING','-o','%i|%j'],text=True)
    if any('search-g1-' in row for row in queue.splitlines()):raise ValueError('ACTIVE_SEARCH_G1_JOB_EXISTS')
    launch=base/'preparation/launch-h1008.sh'
    lines=['#!/bin/bash','set -euo pipefail','export PATH='+shlex.quote(str(base/'env/bin')+':/usr/local/bin:/usr/bin:/bin'),
        'export PYTHONPATH='+shlex.quote(str(base/'repo')),'unset PYTHONHOME RAY_ADDRESS HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy',
        'cd '+shlex.quote(str(base/'repo')),'exec '+shlex.quote(str(base/'env/bin/python'))+' -m reproduction.h1008_qprobe.pipeline --base '+shlex.quote(str(base))]
    with launch.open('x') as f:f.write('\n'.join(lines)+'\n');f.flush();os.fsync(f.fileno())
    launch.chmod(0o700)
    command=['sbatch','--exclusive','--parsable','--partition=gpu','--job-name=search-g1-qprobe3-fresh200','--nodelist=h100-8','--nodes=1','--ntasks=1','--cpus-per-task=96','--mem=512G','--gres=gpu:8','--time=7-00:00:00','--signal=B:TERM@60','--export=NONE','--output='+str(base/'logs/native-%j.log'),'--error='+str(base/'logs/native-%j.log'),str(launch)]
    once(base/'preparation/submit-intent.json',dict(command=command,source_commit=config['source_commit'],verified_files=len(source_files),predecessor_terminal=terminal,created_unix=time.time()))
    r=subprocess.run(command,capture_output=True,text=True,timeout=30)
    receipt=dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,checked_unix=time.time())
    if r.returncode==0:receipt['job_id']=int(r.stdout.strip().split(';')[0])
    once(base/'preparation/submit-receipt.json',receipt)
    if r.returncode:raise RuntimeError('Submission failed; inspect immutable receipt')
    return receipt

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--source-manifest',required=True);a=p.parse_args()
    print(json.dumps(submit(a.base,json.loads(Path(a.source_manifest).read_text()))))
