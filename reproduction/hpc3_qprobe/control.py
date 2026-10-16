"""Once-only HPC3 stage submission and read-only reuse of existing assets."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time

REUSED = Path('/data/user/jxu957/experiments/cosearch-continue300-x203-20261008')
DATA = Path('/data/user/jxu957/experiments/topd-20261008/runtime/assets/common-instruct-0b002b')


def own_base(value):
    base=Path(value).resolve()
    if not str(base).startswith('/data/user/jxu957/experiments/qprobe-teacher-warmup200-'):
        raise ValueError('HPC3_QPROBE_REQUIRES_ITS_OWN_OUTPUT_NAMESPACE')
    return base


def publish(path,value):
    with Path(path).open('x') as f:
        json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno())


def link(target,source):
    if target.is_symlink():
        if target.resolve()!=source.resolve():raise ValueError('Existing asset link differs: '+str(target))
    elif target.exists():raise ValueError('Existing asset path is not a read-only link: '+str(target))
    else:target.symlink_to(source)


def headers(base,binary):
    source=Path('/share/anaconda3/envs/video_dit/include/python3.10')
    from search_g1.qprobe_features import sha_file
    if not (source/'Python.h').is_file() or not (source/'pyconfig.h').is_file():
        raise ValueError('Matching reusable Python3.10 headers unavailable')
    origin=source.parents[1]/'bin/python3.10'
    if sha_file(origin)!=sha_file(binary):raise ValueError('Python headers belong to another runtime binary')
    target=base/'bootstrap-tools/python-dev/usr/include';target.mkdir(parents=True,exist_ok=True)
    link(target/'python3.10',source)
    value=dict(source=str(source),runtime_binary_sha256=sha_file(binary),
        files={str(p.relative_to(source)):sha_file(p) for p in sorted(source.rglob('*')) if p.is_file()},
        status='MATCHING_PYTHON310_HEADER_FULL_SHA_PASS')
    receipt=base/'preparation/python310-headers.json'
    if receipt.exists():
        if json.loads(receipt.read_text())!=value:raise ValueError('Reusable Python headers changed after binding')
    else:publish(receipt,value)
    return value


def prepare(base):
    base=own_base(base)
    for name in ('logs','preparation','data','tmp','cache/hf','env/bin','env/lib/python3.10'):
        (base/name).mkdir(parents=True,exist_ok=True)
    for name in ('manifest.json','train.parquet','validation.parquet','stage.parquet',
                 'final.parquet','exclusions-audit.json'):
        source=DATA/name
        if not source.is_file():raise FileNotFoundError(str(source))
        link(base/'data'/name,source)
    site=REUSED/'source/env/lib/python3.10/site-packages'
    binary=REUSED/'runtime/retriever-python310-candidate/bin/python3.10'
    if not site.is_dir() or not binary.is_file():raise ValueError('Verified reusable Python runtime unavailable')
    link(base/'env/lib/python3.10/site-packages',site)
    (base/'assets').mkdir(exist_ok=True)
    link(base/'assets/e5-base-v2',REUSED/'source/assets-root/assets/e5-base-v2')
    link(base/'retriever-gpu-lib',REUSED/'runtime/gpu-faiss-alignment-candidate')
    headers(base,binary)
    from reproduction.hpc3_qprobe.probe_cpu_runtime import verify as verify_probe_cpu,fit_gate
    cpu_site,cpu_proof=verify_probe_cpu()
    actual_cpu_fit=fit_gate()
    cpu_binding=dict(root=str(cpu_site),status=cpu_proof['status'],
        wheels=cpu_proof['wheels'],file_count=len(cpu_proof['files']),actual_fit=actual_cpu_fit)
    cpu_receipt=base/'preparation/probe-cpu-runtime.json'
    if cpu_receipt.exists():
        if json.loads(cpu_receipt.read_text())!=cpu_binding:raise ValueError('CPU runtime binding changed')
    else:publish(cpu_receipt,cpu_binding)
    wrapper='#!/usr/bin/env bash\nset -euo pipefail\n'
    wrapper+='export PYTHONPATH="${QPROBE_SOURCE:-'+str(base/'repo')+'}:'+str(cpu_site)+':'+str(REUSED/'runtime/gpu-faiss-alignment-candidate')+':'+str(site)+'${PYTHONPATH:+:$PYTHONPATH}"\n'
    wrapper+='export LD_LIBRARY_PATH="'+str(REUSED/'runtime/retriever-python310-candidate/lib')+'${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"\n'
    wrapper+='unset PYTHONHOME\nexec '+shlex.quote(str(binary))+' "$@"\n'
    python=base/'env/bin/python'
    if python.exists():
        existing=python.read_text()
        if existing!=wrapper:
            legacy=wrapper.replace('export LD_LIBRARY_PATH="'+str(REUSED/'runtime/retriever-python310-candidate/lib')+'${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"',
                                  'export LD_LIBRARY_PATH='+shlex.quote(str(REUSED/'runtime/retriever-python310-candidate/lib')))
            if existing!=legacy or any((base/'preparation'/x).exists() for x in ('training-submit-claim.json','debug-submit-claim.json')):
                raise ValueError('Unexpected or already-used runtime wrapper differs')
            # Feature job has its own direct immutable binary entry; it never
            # references this wrapper. Upgrade only before debug/training claim.
            temporary=python.with_suffix('.prepared');temporary.write_text(wrapper);temporary.chmod(0o755)
            temporary.replace(python)
            publish(base/'preparation/env-wrapper-upgrade.json',dict(
                before_sha256=hashlib.sha256(existing.encode()).hexdigest(),
                after_sha256=hashlib.sha256(wrapper.encode()).hexdigest(),
                feature_entry_modified=False,training_claim_absent=True))
    else:
        with python.open('x') as f:f.write(wrapper);f.flush();os.fsync(f.fileno())
        python.chmod(0o755)
    return dict(base=str(base),data_source=str(DATA),python=str(python),
        model_source='/data/user/jxu957/experiments/topd-20261008/runtime/assets/qwen2.5-3b-instruct-aa8e725',
        output_resolves_to_remote=True,existing_experiments_modified=False)


def submit_features(base,commit):
    base=own_base(base);prepare(base)
    source=base/'source'/commit
    verification=json.loads((base/'preparation'/('source-'+commit+'-verification.json')).read_text())
    if verification['commit']!=commit or verification['status']!='PUBLISHED_SOURCE_FULL_SHA_PASS':
        raise ValueError('Feature source publication/deployment not accepted')
    receipt=base/'preparation/features-submission.json'
    intent=base/'preparation/features-submit-claim.json'
    if intent.exists() or receipt.exists():
        raise ValueError('Feature submission already attempted; inspect actual job instead of replaying')
    command=['sbatch','--parsable','--partition=acd_u','--nodes=1','--ntasks=1',
        '--gres=gpu:1','--cpus-per-task=8','--mem=64G','--time=06:00:00',
        '--job-name=qprobe-features200','--export=NONE',
        '--output='+str(base/'logs/features-%j.log'),'--error='+str(base/'logs/features-%j.log'),
        str(source/'reproduction/hpc3_qprobe/features.sh'),str(base),str(source)]
    publish(intent,dict(commit=commit,command=command,epoch=time.time()))
    result=subprocess.run(command,capture_output=True,text=True,timeout=30)
    value=dict(commit=commit,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,epoch=time.time())
    if result.returncode==0:
        job=result.stdout.strip().split(';')[0]
        if not job.isdecimal():raise ValueError('Slurm returned an invalid feature job identity')
        value['job']=job
    publish(receipt,value)
    if result.returncode:raise RuntimeError('Feature submission failed; inspect saved evidence')
    return value


def submit_training(base,commit):
    base=own_base(base);prepare(base)
    source=base/'source'/commit
    verification=json.loads((base/'preparation'/('source-'+commit+'-verification.json')).read_text())
    if verification['commit']!=commit or verification['status']!='PUBLISHED_SOURCE_FULL_SHA_PASS':
        raise ValueError('Training source publication/deployment not accepted')
    features=json.loads((base/'preparation/features-submission.json').read_text())
    if features['returncode']!=0 or not features['job'].isdecimal():raise ValueError('Missing own feature dependency')
    intent=base/'preparation/training-submit-claim.json';receipt=base/'preparation/training-submission.json'
    if intent.exists() or receipt.exists():raise ValueError('Training submission already attempted; never duplicate')
    if (base/'repo').exists() or (base/'repo').is_symlink():
        if (base/'repo').resolve()!=source:raise ValueError('Immutable run source already points elsewhere')
    else:(base/'repo').symlink_to(source)
    command=['sbatch','--parsable','--partition=acd_u','--nodes=1','--ntasks=1',
        '--gres=gpu:4','--cpus-per-task=48','--mem=512G','--time=7-00:00:00',
        '--job-name=qprobe-warmup200','--export=NONE','--dependency=afterok:'+features['job'],
        '--output='+str(base/'logs/training-%j.log'),'--error='+str(base/'logs/training-%j.log'),
        str(source/'reproduction/hpc3_qprobe/train.sh'),str(base),commit]
    publish(intent,dict(commit=commit,command=command,epoch=time.time(),feature_dependency=features['job']))
    result=subprocess.run(command,capture_output=True,text=True,timeout=30)
    value=dict(commit=commit,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,epoch=time.time(),feature_dependency=features['job'])
    if result.returncode==0:
        value['job']=result.stdout.strip().split(';')[0]
        if not value['job'].isdecimal():raise ValueError('Invalid actual training job ID')
    publish(receipt,value)
    if result.returncode:raise RuntimeError('Training submission failed; inspect saved evidence')
    return value


def submit_debug(base,commit,recover_failed_job=None):
    base=own_base(base);prepare(base);source=base/'source'/commit
    proof=json.loads((base/'preparation'/('source-'+commit+'-verification.json')).read_text())
    if proof['commit']!=commit or proof['status']!='PUBLISHED_SOURCE_FULL_SHA_PASS':
        raise ValueError('Debug source not published and verified')
    stem='debug'
    prior=None
    if recover_failed_job is not None:
        if not recover_failed_job.isdecimal():raise ValueError('Invalid failed debug job ID')
        receipts=[json.loads(p.read_text()) for p in (base/'preparation').glob('debug*-submission.json')]
        prior=next((v for v in receipts if v.get('job')==recover_failed_job),None)
        if prior is None:raise ValueError('Failed job is not this task\'s submitted debug job')
        from reproduction.common8.native_allocation import fields
        detail=subprocess.check_output(['scontrol','show','job','-d',recover_failed_job],text=True,timeout=30)
        actual=fields(detail)
        if actual['JobState'] not in ('FAILED','OUT_OF_MEMORY','TIMEOUT','NODE_FAIL') or actual['JobName']!='qprobe-runtime-debug' or actual['Command']!=str(base/'source'/prior['commit']/'reproduction/hpc3_qprobe/debug.sh'):
            raise ValueError('Only an actually failed own debug job may be recovered')
        evidence=base/'preparation'/('failed-debug-'+recover_failed_job);evidence.mkdir(exist_ok=True)
        old=base/'logs'/('debug-'+recover_failed_job+'.log')
        if old.is_file() and not (evidence/'debug.log').exists():
            with (evidence/'debug.log').open('xb') as f:f.write(old.read_bytes());f.flush();os.fsync(f.fileno())
        preserved=evidence/'terminal.json'
        if not preserved.exists():publish(preserved,dict(prior=prior,actual=actual,detail=detail,
            failure_log_sha256=hashlib.sha256(old.read_bytes()).hexdigest() if old.is_file() else None))
        stem='debug-recovery-'+recover_failed_job
    intent=base/'preparation'/(stem+'-submit-claim.json');receipt=base/'preparation'/(stem+'-submission.json')
    if intent.exists() or receipt.exists():raise ValueError('Debug submission already attempted; inspect actual job')
    command=['sbatch','--parsable','--partition=debug','--nodes=1','--ntasks=1','--gres=gpu:1',
        '--cpus-per-task=8','--mem=64G','--time=00:20:00','--job-name=qprobe-runtime-debug','--export=NONE',
        '--output='+str(base/'logs/debug-%j.log'),'--error='+str(base/'logs/debug-%j.log'),
        str(source/'reproduction/hpc3_qprobe/debug.sh'),str(base),str(source)]
    publish(intent,dict(commit=commit,command=command,epoch=time.time(),recovery_of_failed_job=recover_failed_job))
    r=subprocess.run(command,capture_output=True,text=True,timeout=30)
    value=dict(commit=commit,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,epoch=time.time(),recovery_of_failed_job=recover_failed_job)
    if r.returncode==0:
        value['job']=r.stdout.strip().split(';')[0]
        if not value['job'].isdecimal():raise ValueError('Invalid actual debug job ID')
    publish(receipt,value)
    if r.returncode:raise RuntimeError('Debug submission failed; inspect evidence')
    return value


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('prepare','submit-features','submit-training','submit-debug'))
    p.add_argument('--base',required=True);p.add_argument('--commit');p.add_argument('--recover-failed-job')
    a=p.parse_args()
    if a.recover_failed_job and a.mode!='submit-debug':raise ValueError('Only the debug recovery route is supported here')
    result=prepare(a.base) if a.mode=='prepare' else (submit_debug(a.base,a.commit,a.recover_failed_job) if a.mode=='submit-debug'
        else {'submit-features':submit_features,'submit-training':submit_training}[a.mode](a.base,a.commit))
    print(json.dumps(result),flush=True)


if __name__=='__main__':main()
