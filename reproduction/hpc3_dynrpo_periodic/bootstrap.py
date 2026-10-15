"""Receive a code-only Git archive over stdin; build an independent remote shell.

The first input line is the fetched-Git SHA inventory/plan; remaining bytes are
the code archive. This program never receives a password, weight or dataset.
"""
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import tarfile
import time
import subprocess

EXPECTED='/data/user/jxu957/experiments/qprobe-teacher-warmup200-periodic25-thread5-20261016'
REUSED=Path('/data/user/jxu957/experiments/cosearch-continue300-x203-20261008')
DATA=Path('/data/user/jxu957/experiments/topd-20261008/runtime/assets/common-instruct-0b002b')


def main():
    os.umask(0o007)
    record=json.loads(sys.stdin.buffer.readline());publication,plan=record['publication'],record['plan']
    base=Path(plan['base'])
    if str(base)!=EXPECTED or base.resolve()!=base or base.is_symlink():
        raise ValueError('Refuse deployment outside the independent namespace')
    if publication['pinned_base']!='1af0d14937a3da6666eec63e8caa50c192d481e6' or not publication['pushed_and_fetched']:
        raise ValueError('Unpublished code cannot be deployed')
    archive=sys.stdin.buffer.read()
    if hashlib.sha256(archive).hexdigest()!=record['archive_sha256']:
        raise ValueError('Code transport SHA differs')
    base.mkdir(mode=0o770,exist_ok=True)
    if base.stat().st_uid!=os.getuid():raise ValueError('This namespace belongs to another owner')
    for name in ('preparation','logs','source','data','env/bin','cache/hf','cache/triton','cache/torch-extensions',
                 'cache/inductor','tmp','warmup','assets','env/lib/python3.10','bootstrap-tools/python-dev/usr/include'):
        (base/name).mkdir(parents=True,exist_ok=True)
    source=base/'source'/publication['commit']
    if source.exists():
        for name,digest in publication['files'].items():
            if hashlib.sha256((source/name).read_bytes()).hexdigest()!=digest:
                raise ValueError('An existing deployment differs; retain it for inspection')
    else:
        stage=source.with_name('.staging-'+source.name)
        stage.mkdir(exist_ok=False)
        with tarfile.open(fileobj=io.BytesIO(archive),mode='r:gz') as tar:
            for member in tar.getmembers():
                p=Path(member.name)
                if p.is_absolute() or '..' in p.parts or member.issym() or member.islnk():
                    raise ValueError('Unsafe code archive member')
            tar.extractall(stage)
        for name,digest in publication['files'].items():
            if hashlib.sha256((stage/name).read_bytes()).hexdigest()!=digest:
                raise ValueError('Remote source file differs: '+name)
        stage.rename(source)
    existing=base/'preparation/source-publication.json'
    if existing.exists() and json.loads(existing.read_text())['commit']!=publication['commit']:
        # A returned, rejected sbatch is not an unknown submission or a running
        # experiment. Preserve its receipts before correcting the resource limit.
        rejected=base/'preparation/preparation-submission.json'
        if rejected.exists():
            outcome=json.loads(rejected.read_text())
        else:
            if (base/'preparation/preparation-submit-claim.json').exists():
                raise ValueError('Prior submission outcome is unknown; reconcile it first')
            outcome=dict(returncode=1,stdout='')
        if outcome.get('job'):
            job=outcome['job']
            if not str(job).isdecimal():raise ValueError('Unknown prior job identity')
            accounting=subprocess.check_output(['sacct','-nP','-X','-j',job,'--format=JobID,State,ExitCode'],text=True)
            entries=[row.split('|') for row in accounting.splitlines() if row.strip()]
            completed_preparation=(len(entries)==1 and entries[0][0]==job and entries[0][1]=='COMPLETED' and
                entries[0][2]=='0:0' and (base/'warmup/ready.json').is_file())
            if len(entries)!=1 or entries[0][0]!=job or (entries[0][1].split()[0] not in ('FAILED','CANCELLED','OUT_OF_MEMORY','TIMEOUT') and not completed_preparation):
                raise ValueError('Existing producer is not an authenticated failed terminal job')
        elif outcome.get('returncode')==0 or outcome.get('stdout','').strip():
            raise ValueError('Cannot upgrade an ambiguous submission')
        accepted_warmup=(base/'warmup/ready.json').is_file()
        if accepted_warmup:
            ready=json.loads((base/'warmup/ready.json').read_text())
            if ready.get('head_adam_steps')!=400 or hashlib.sha256((base/'warmup/continued-head.pt').read_bytes()).hexdigest()!=ready['head_sha256']:
                raise ValueError('Accepted continuous warmup is not intact')
        for name in ('logs/run-manifest.json','preparation/warmup-feature-cursor.json',
                     'preparation/feature-cursor.json','warmup/latest-state.pt','warmup/ready.json'):
            if (base/name).exists() and not (accepted_warmup and name.startswith('warmup/')):
                raise ValueError('Real model/feature state exists; use a state-bound recovery instead')
        queue=subprocess.check_output(['squeue','-h','-u',str(os.getuid()),'--name=dynrpo-periodic5-preparation','-o','%i'],text=True)
        if queue.strip():raise ValueError('An existing queued producer must be reconciled first')
        queue=subprocess.check_output(['squeue','-h','-u',str(os.getuid()),'--name=dynrpo-periodic5-formal','-o','%i'],text=True)
        if queue.strip():raise ValueError('An existing formal job must not be hot modified')
        old=json.loads(existing.read_text())['commit']
        history=base/'preparation'/('rejected-submit-'+old);history.mkdir(exist_ok=False)
        for name in ('source-publication.json','deployment-plan.json','preparation-submit-claim.json','preparation-submission.json',
                     'preparation-controller.json','native-cpu-tests.json','progress.json'):
            path=base/'preparation'/name
            if path.exists():path.rename(history/name)
        alias=base/'repo'
        if not alias.is_symlink():raise ValueError('Source alias is not owned deployment metadata')
        temporary=base/'repo.next';temporary.symlink_to(source);temporary.replace(alias)
    def link(target,origin):
        origin=Path(origin)
        if not origin.exists():raise ValueError('Shared input unavailable: '+str(origin))
        if target.is_symlink():
            if target.resolve()!=origin.resolve():raise ValueError('Existing reference points elsewhere')
        elif target.exists():raise ValueError('Existing output must not be replaced by a shared input')
        else:target.symlink_to(origin.resolve())
    link(base/'repo',source)
    for name in ('manifest.json','train.parquet','validation.parquet','stage.parquet','final.parquet','exclusions-audit.json'):
        link(base/'data'/name,DATA/name)
    link(base/'retriever-gpu-lib',REUSED/'runtime/gpu-faiss-alignment-candidate')
    link(base/'assets/e5-base-v2',REUSED/'source/assets-root/assets/e5-base-v2')
    headers=Path('/data/user/jxu957/experiments/qprobe-teacher-warmup200-20261014/bootstrap-tools/python-dev/usr/include/python3.10')
    link(base/'bootstrap-tools/python-dev/usr/include/python3.10',headers)
    binary=REUSED/'runtime/retriever-python310-candidate/bin/python3.10'
    site=REUSED/'source/env/lib/python3.10/site-packages'
    link(base/'env/lib/python3.10/site-packages',site)
    wrapper='#!/usr/bin/env bash\nset -euo pipefail\n'
    wrapper+='export PYTHONPATH="${QPROBE_SOURCE:-'+str(base/'repo')+'}:'+str(base/'retriever-gpu-lib')+':'+str(site)+'${PYTHONPATH:+:$PYTHONPATH}"\n'
    wrapper+='export LD_LIBRARY_PATH="'+str(REUSED/'runtime/retriever-python310-candidate/lib')+'${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"\n'
    wrapper+='unset PYTHONHOME\nexec '+str(binary)+' "$@"\n'
    target=base/'env/bin/python'
    if target.exists() and target.read_text()!=wrapper:raise ValueError('Existing environment wrapper differs')
    if not target.exists():target.write_text(wrapper);target.chmod(0o755)
    def store(path,value):
        raw=json.dumps(value,sort_keys=True,indent=2)+'\n'
        if path.exists() and path.read_text()!=raw:raise ValueError('Existing deployment metadata differs')
        if not path.exists():
            with path.open('x') as f:f.write(raw);f.flush();os.fsync(f.fileno())
    store(base/'preparation/source-publication.json',publication)
    store(base/'preparation/deployment-plan.json',plan)
    print(json.dumps(dict(status='INDEPENDENT_HPC3_SOURCE_FULL_SHA_PASS',base=str(base),commit=publication['commit'],
        file_count=len(publication['files']),uid=os.getuid(),shared_assets_modified=False,checked_unix=time.time())))


if __name__=='__main__':main()
