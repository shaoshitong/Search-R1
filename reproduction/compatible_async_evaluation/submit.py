"""Execution-only Slurm control for already prepared fe2 evaluation science.

No prepare, model loading, asset hashing, training stop, or automatic retry.
A separately published controller always executes the original prepared launch.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import socket
import subprocess
import time

from reproduction.compatible_async_evaluation.resume import TERMINAL, read, once
from reproduction.a800.owned_processes import same, identity
from reproduction.common8.native_allocation import fields

SCIENCE_COMMIT='fe2a29705dd105cfeb62f626e290fbe0605655ef'
SCIENCE_ROOT=Path('/mnt/data/shaoshitong/search-g1-compatible-eval-fe2a297-20261015/control')
RECIPE_SHA='a6924b53914477b7df5779486423be1b1d6b12f08905e3a95e9c2c07807a8be7'
TRAINER=Path('/mnt/data/shaoshitong/search-g1-qprobe3-binary10-c5fc096-20261015')
TRAINER_JOB='3402'
RUN='f77c3245dcfe49a88a5a99b1dec161e6'
ROLE='canonical_evaluation_execution_only_submission'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def query(args):return subprocess.check_output(args,text=True,timeout=30)
def node():
    if socket.gethostname().split('.')[0]!='h100-8':raise ValueError('EXECUTION_CONTROL_H1008_ONLY')


def prepared(base,expected_receipt_sha):
    """Only small prepared metadata; do not import the new controller's pipeline."""
    base=Path(base).resolve();root=base/'preparation'
    if base==TRAINER or TRAINER in base.parents:raise ValueError('EVALUATION_CONTROL_CANNOT_WRITE_TRAINER_NAMESPACE')
    if sha(root/'prepare-receipt.json')!=expected_receipt_sha:raise ValueError('PINNED_PREPARE_RECEIPT_CHANGED')
    receipt=read(root/'prepare-receipt.json');manifest=read(root/'compatible-evaluation-manifest.json');binding=read(root/'binding.json');intent=read(root/'prepare-intent.json')
    if sha(root/'binding.json')!=receipt['binding_sha256'] or sha(root/'compatible-evaluation-manifest.json')!=receipt['manifest_sha256']:raise ValueError('PREPARED_BINDING_OR_MANIFEST_CHANGED')
    if receipt.get('role')!='compatible_async_checkpoint_evaluation' or manifest.get('role')!=receipt['role'] or binding.get('role')!=receipt['role'] or intent.get('role')!=receipt['role']:raise ValueError('PREPARED_ROLE_CHANGED')
    if manifest['base']!=str(base) or manifest['run_id']!=RUN or manifest['source_base']!=str(TRAINER) or manifest['allocation_mode']!='shared' or manifest['world_size']!=8 or manifest['cpu_request']!=96 or manifest['memory_request']!='512G':raise ValueError('PREPARED_EXECUTION_OR_BUDGET_DIFFERS')
    if manifest['control_repo']!=str(SCIENCE_ROOT) or manifest['source_root']!=str(SCIENCE_ROOT) or binding['source_root']!=str(SCIENCE_ROOT) or binding['control_source_commit']!=SCIENCE_COMMIT or binding['recipe_sha256']!=RECIPE_SHA or manifest['recipe_sha256']!=RECIPE_SHA:raise ValueError('FROZEN_FE2_SCIENCE_CHANGED')
    for key in ('run_id','checkpoint_step','split'):
        if receipt[key]!=manifest[key] or binding[key]!=manifest[key] or intent[key]!=manifest[key]:raise ValueError('PREPARATION_IDENTITIES_DIVERGE')
    if intent['execution_id']!=manifest['execution_id'] or not re.fullmatch('[0-9a-f]{32}',manifest['execution_id']):raise ValueError('PREPARED_EXECUTION_ID_CHANGED')
    if TRAINER_JOB not in {str(v) for v in manifest.get('forbidden_training_job_ids',[])}:raise ValueError('TRAINER_OVERLAP_MUST_REMAIN_FORBIDDEN')
    if manifest['checkpoint_step'] not in (0,5,50,100,150,200) or manifest['split'] not in ('validation','final') or (manifest['split']=='final' and manifest['checkpoint_step']!=200):raise ValueError('PREPARED_STEP_SPLIT_INVALID')
    if manifest['binding']!=str(root/'binding.json') or manifest['config']!=str(root/'reference-config.json') or manifest['journal']!=str(base/'journal'):raise ValueError('PREPARED_METADATA_PATHS_CHANGED')
    if sha(root/'reference-config.json')!=binding['resolved_config_sha256']:raise ValueError('REFERENCE_CONFIG_CHANGED')
    launch=root/'launch.sh'
    expected='#!/bin/bash\nset -euo pipefail\nexport PYTHONPATH='+shlex.quote(str(SCIENCE_ROOT))+'\nexec '+shlex.join([str(base/'env/bin/python'),'-m','reproduction.compatible_async_evaluation.pipeline','--base',str(base)])+'\n'
    if launch.is_symlink() or launch.read_text()!=expected:raise ValueError('ORIGINAL_PREPARED_LAUNCH_CHANGED')
    return manifest,dict(prepare_receipt_sha256=expected_receipt_sha,manifest_sha256=receipt['manifest_sha256'],binding_sha256=receipt['binding_sha256'],launch_sha256=sha(launch))


def trainer_bindings():
    run=read(TRAINER/'logs/run-manifest.json');admission=read(TRAINER/'logs/native-allocation-admission.json');ledger=read(TRAINER/'logs/process-ledger.json')
    if run['run_id']!=RUN or run['source_commit']!='c5fc09603a10244655b98ec0ce8eada32668f83b' or str(admission['slurm_job_id'])!=TRAINER_JOB or admission['run_id']!=RUN or admission['node']!='h100-8' or admission['accepted'] is not True or ledger['run_id']!=RUN or not ledger['jobs']:raise ValueError('ACTUAL_TRAINER_IDENTITY_CHANGED')
    pair_path=TRAINER/'artifacts/question-probe/completed-pairs/global_step_200/source-receipt.json';pair=read(pair_path)
    if pair['run_id']!=RUN or pair['step']!=200 or pair['status']!='QPROBE_COMPLETED_PAIR_FULL_SHA_CPU_STATES_PASS' or pair['source_base']!=str(TRAINER):raise ValueError('ACTUAL_COMPLETE200_RECEIPT_REQUIRED')
    return ledger,dict(trainer_job_id=TRAINER_JOB,trainer_base=str(TRAINER),run_id=RUN,trainer_run_manifest_sha256=sha(TRAINER/'logs/run-manifest.json'),native_admission_sha256=sha(TRAINER/'logs/native-allocation-admission.json'),process_ledger_sha256=sha(TRAINER/'logs/process-ledger.json'),complete200_pair_sha256=sha(pair_path))


def queue_snapshot():
    raw=query(['squeue','-h','-o','%i|%j|%T'])
    rows=[]
    for line in raw.splitlines():
        parts=line.split('|')
        if len(parts)!=3:raise ValueError('QUEUE_RECORD_INVALID')
        rows.append(dict(job_id=parts[0].strip(),job_name=parts[1].strip(),state=parts[2].strip()))
    return rows


def released(ledger):
    cleanup=read(TRAINER/'logs/cleanup-status.json')
    if cleanup.get('run_id')!=RUN or cleanup.get('cleaned') is not True:raise ValueError('TRAINER_CLEANUP_NOT_COMPLETE')
    births=[r for job in ledger['jobs'].values() for r in job['ledger'].values()]
    sessions={(job['leader']['sid'],job['leader']['uid']) for job in ledger['jobs'].values()}
    if any(same(r) for r in births):raise ValueError('TRAINER_OWNED_BIRTH_STILL_LIVE')
    for path in Path('/proc').iterdir():
        if not path.name.isdecimal():continue
        try:r=identity(int(path.name))
        except (FileNotFoundError,ProcessLookupError,PermissionError):continue
        if r['state']!='Z' and (r['sid'],r['uid']) in sessions:raise ValueError('TRAINER_OWNED_SESSION_STILL_LIVE')
    # Same native-job cgroup check as the existing training terminal controller.
    if list(Path('/sys/fs/cgroup').glob('**/job_'+TRAINER_JOB)) or list(Path('/sys/fs/cgroup').glob('**/job'+TRAINER_JOB)):raise ValueError('TRAINER_NATIVE_CGROUP_NOT_RELEASED')
    gpu=query(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'])
    for pid in {str(r['pid']) for r in births}&{line.strip() for line in gpu.splitlines()}:
        try:current=identity(int(pid))
        except (FileNotFoundError,ProcessLookupError):raise ValueError('TRAINER_GPU_CONTEXT_STILL_PRESENT')
        if any(str(r['pid'])==pid and all(current[k]==r[k] for k in ('start_ticks','sid','uid')) for r in births):raise ValueError('TRAINER_GPU_CONTEXT_STILL_PRESENT')
    return dict(owned_births_released=True,owned_sessions_released=True,native_cgroup_released=True,owned_gpu_contexts_released=True)


def observe_terminal():
    node()
    if any(r['job_id']==TRAINER_JOB for r in queue_snapshot()):raise ValueError('TRAINER_STILL_ACTIVE_OR_QUEUED')
    # A successful same-job native terminal record is necessary, never absence alone.
    raw=query(['scontrol','show','job',TRAINER_JOB,'-o']);native=fields(raw)
    if native.get('JobId')!=TRAINER_JOB or native.get('NodeList')!='h100-8' or native.get('JobState') not in TERMINAL or not re.fullmatch(r'\d+:\d+',native.get('ExitCode','')):raise ValueError('ACTUAL_SAME_JOB_TERMINAL_REQUIRED')
    ledger,binding=trainer_bindings()
    return dict(binding,native_state=native['JobState'],native_exit_code=native['ExitCode'],actual_scontrol_stdout=raw,**released(ledger),checked_unix=time.time())


def capture_terminal(path):
    """Capture before native purge. If already captured, inspect; never overwrite."""
    path=Path(path)
    if path.exists():raise ValueError('TERMINAL_CAPTURE_EXISTS_INSPECT_DO_NOT_REPLAY')
    first=observe_terminal();second=observe_terminal()
    if any(first[k]!=second[k] for k in ('trainer_job_id','run_id','trainer_run_manifest_sha256','native_admission_sha256','process_ledger_sha256','complete200_pair_sha256','native_state','native_exit_code')):raise ValueError('TRAINER_TERMINAL_CHANGED_BETWEEN_OBSERVATIONS')
    once(path,dict(role=ROLE+'_trainer_terminal',observations=[first,second]));return dict(path=str(path),sha256=sha(path))


def capture_prepared_terminal(base,prepare_sha,output):
    base=Path(base).resolve()
    if not prepare_sha:raise ValueError('CAPTURE_REQUIRES_PINNED_EVALUATION_PREPARATION')
    prepared(base,prepare_sha)
    output=Path(output).resolve()
    if output!=base/'preparation/trainer-terminal-proof-once-v1.json':raise ValueError('TERMINAL_PROOF_REQUIRES_SEPARATE_EVALUATION_EVIDENCE_PATH')
    return capture_terminal(output)


def trainer_guard(path,expected_sha):
    node()
    if not expected_sha or sha(path)!=expected_sha:raise ValueError('PINNED_ACTUAL_TRAINER_TERMINAL_PROOF_REQUIRED')
    proof=read(path);rows=proof.get('observations',[])
    if proof.get('role')!=ROLE+'_trainer_terminal' or len(rows)!=2:raise ValueError('TWO_REAL_TRAINER_TERMINAL_OBSERVATIONS_REQUIRED')
    if any(r['job_id']==TRAINER_JOB for r in queue_snapshot()):raise ValueError('TRAINER_STILL_ACTIVE_OR_QUEUED')
    ledger,binding=trainer_bindings()
    for row in rows:
        if any(row.get(k)!=v for k,v in binding.items()) or row.get('native_state') not in TERMINAL or not re.fullmatch(r'\d+:\d+',row.get('native_exit_code','')):raise ValueError('TRAINER_TERMINAL_BINDING_DIFFERS')
        native=fields(row.get('actual_scontrol_stdout',''))
        if native.get('JobId')!=TRAINER_JOB or native.get('NodeList')!='h100-8' or native.get('JobState')!=row['native_state'] or native.get('ExitCode')!=row['native_exit_code'] or any(row.get(k) is not True for k in ('owned_births_released','owned_sessions_released','native_cgroup_released','owned_gpu_contexts_released')):raise ValueError('RETAINED_NATIVE_TERMINAL_OR_RELEASE_PROOF_INVALID')
    if rows[0]['native_state']!=rows[1]['native_state'] or rows[0]['native_exit_code']!=rows[1]['native_exit_code']:raise ValueError('RETAINED_TERMINAL_OBSERVATIONS_DIVERGE')
    return dict(binding,**released(ledger),terminal_proof_sha256=expected_sha)


def foreign_guard(manifest):
    node();active=query(['squeue','-h','-w','h100-8','-t','RUNNING,COMPLETING','-o','%i']).split()
    if TRAINER_JOB in active:raise ValueError('TRAINER_STILL_ACTIVE')
    permitted={str(r['job_id']):r for r in manifest.get('preserved_foreign_allocations',[])}
    if set(active)-set(permitted):raise ValueError('FOREIGN_ALLOCATION_BINDING_CHANGED_REPREPARE_FRESH_NAMESPACE')
    for job in active:
        native=fields(query(['scontrol','show','job',job,'-o']))
        if native.get('JobId')!=job or native.get('NodeList')!='h100-8' or native.get('UserId')!=permitted[job]['user_id'] or native.get('JobState') not in ('RUNNING','COMPLETING'):raise ValueError('FOREIGN_ALLOCATION_IDENTITY_CHANGED')
    inventory=[line.split(',') for line in query(['nvidia-smi','--query-gpu=index,uuid','--format=csv,noheader,nounits']).splitlines()]
    physical=[uuid.strip() for index,uuid in sorted(inventory,key=lambda r:int(r[0]))]
    if physical!=manifest['physical_gpu_uuids'] or len(set(physical))!=8:raise ValueError('PHYSICAL_EIGHT_UUID_MAPPING_CHANGED')
    permitted_apps={(r['gpu_uuid'],str(r['identity']['pid'])) for r in manifest.get('preserved_foreign_gpu_processes',[]) if same(r['identity'])}
    actual=[tuple(v.strip() for v in line.split(',')) for line in query(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits']).splitlines()]
    if set(actual)-permitted_apps:raise ValueError('FOREIGN_GPU_BIRTH_BINDING_CHANGED_REPREPARE_FRESH_NAMESPACE')
    return dict(active_preserved_jobs=active,actual_preserved_gpu_applications=actual)


def wrapper(base):
    return '#!/bin/bash\nset -euo pipefail\nexport PATH=/usr/local/bin:/usr/bin:/bin\nexport CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7\nunset PYTHONHOME RAY_ADDRESS HTTP_PROXY HTTPS_PROXY ALL_PROXY http_proxy https_proxy all_proxy\nexec /bin/bash '+shlex.quote(str(base/'preparation/launch.sh'))+'\n'


def command(base,execution):
    return ['sbatch','--parsable','--partition=gpu','--oversubscribe','--nodelist=h100-8','--nodes=1','--ntasks=1','--cpus-per-task=96','--mem=512G','--time=7-00:00:00','--signal=B:TERM@60','--export=NONE','--job-name=g1ce-'+execution[:12],'--comment=g1-canonical-eval:'+execution,'--chdir='+str(base),'--output='+str(base/'logs/native-%j.log'),'--error='+str(base/'logs/native-%j.log'),str(base/'preparation/submit-launch.sh')]


def query_existing(base):
    """Never sbatch. Preserve UNKNOWN when the scheduler cannot resolve identity."""
    root=base/'preparation';intent=read(root/'submit-intent.json')
    execution=intent.get('execution_id','')
    if intent.get('role')!=ROLE or intent.get('base')!=str(base) or intent.get('run_id')!=RUN or not re.fullmatch('[0-9a-f]{32}',execution) or intent.get('job_name')!='g1ce-'+execution[:12] or intent.get('comment')!='g1-canonical-eval:'+execution or sha(root/'submit-launch.sh')!=intent['wrapper_sha256']:raise ValueError('EXISTING_SUBMIT_IDENTITY_OR_WRAPPER_CHANGED')
    matches=[]
    for row in queue_snapshot():
        if row['job_name']==intent['job_name']:
            native=fields(query(['scontrol','show','job',row['job_id'],'-o']))
            if native.get('JobId')==row['job_id'] and native.get('NodeList')=='h100-8' and native.get('Comment')==intent['comment'] and native.get('WorkDir')==str(base) and native.get('Command')==str(root/'submit-launch.sh') and native.get('JobName')==intent['job_name']:matches.append(dict(job_id=row['job_id'],native=native))
    receipt=root/'submit-receipt.json'
    saved=read(receipt) if receipt.exists() else None
    if saved is not None and saved.get('intent_sha256')!=sha(root/'submit-intent.json'):raise ValueError('EXISTING_RECEIPT_INTENT_SHA_CHANGED')
    if saved and saved.get('job_id') and not matches:
        try:
            native=fields(query(['scontrol','show','job',saved['job_id'],'-o']))
            if native.get('JobId')==saved['job_id'] and native.get('NodeList')=='h100-8' and native.get('Comment')==intent['comment'] and native.get('WorkDir')==str(base) and native.get('Command')==str(root/'submit-launch.sh') and native.get('JobName')==intent['job_name']:matches.append(dict(job_id=saved['job_id'],native=native))
        except subprocess.CalledProcessError:pass
    if len(matches)>1:raise ValueError('MULTIPLE_MATCHING_SUBMISSIONS_UNKNOWN_NEVER_REPLAY')
    result=dict(status='RECONCILED_EXISTING_JOB' if len(matches)==1 else 'UNKNOWN_NO_REPLAY',matches=matches,receipt=saved,sbatch_replayed=False)
    if matches:
        path=root/'submit-reconciliation.json'
        if not path.exists():once(path,result)
    return result


def submit(base,prepare_sha,terminal_path,terminal_sha,controller_commit):
    node();base=Path(base).resolve();root=base/'preparation'
    if base==TRAINER or TRAINER in base.parents:raise ValueError('EVALUATION_CONTROL_CANNOT_WRITE_TRAINER_NAMESPACE')
    with (root/'submit-control.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (root/'submit-intent.json').exists():return query_existing(base)
        if (root/'submit-receipt.json').exists() or (root/'submit-launch.sh').exists():raise ValueError('PARTIAL_SUBMISSION_ARTIFACTS_INSPECT_NO_REPLAY')
        manifest,binding=prepared(base,prepare_sha)
        if (base/'logs/process-ledger.json').exists() or Path(manifest['ray_tmp_dir']).exists():raise ValueError('EVALUATION_ALREADY_STARTED_OR_PARTIAL_INSPECT')
        terminal=trainer_guard(terminal_path,terminal_sha);foreign=foreign_guard(manifest)
        text=wrapper(base);launch=root/'submit-launch.sh'
        with launch.open('x') as f:f.write(text);f.flush();os.fsync(f.fileno())
        launch.chmod(0o700);cmd=command(base,manifest['execution_id'])
        intent=dict(role=ROLE,base=str(base),run_id=RUN,execution_id=manifest['execution_id'],science_control_commit=SCIENCE_COMMIT,execution_control_commit=controller_commit,prepared=binding,trainer_release=terminal,foreign_bindings=foreign,wrapper_sha256=sha(launch),command=cmd,job_name='g1ce-'+manifest['execution_id'][:12],comment='g1-canonical-eval:'+manifest['execution_id'],created_unix=time.time())
        once(root/'submit-intent.json',intent)
        try:
            result=subprocess.run(cmd,capture_output=True,text=True,timeout=30)
            receipt=dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,status='REJECTED_NO_REPLAY',intent_sha256=sha(root/'submit-intent.json'))
            if result.returncode==0:
                parsed=result.stdout.strip().split(';')[0]
                if not parsed.isdecimal():receipt['status']='UNKNOWN_NO_REPLAY'
                else:receipt.update(status='SUBMITTED',job_id=parsed)
        except (subprocess.TimeoutExpired,OSError) as error:
            receipt=dict(status='UNKNOWN_NO_REPLAY',error=repr(error),intent_sha256=sha(root/'submit-intent.json'))
        once(root/'submit-receipt.json',receipt)
        return receipt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('phase',choices=('submit','query','capture-trainer-terminal'));p.add_argument('--base',required=True);p.add_argument('--prepare-receipt-sha');p.add_argument('--trainer-terminal-proof');p.add_argument('--trainer-terminal-proof-sha');p.add_argument('--controller-proof',required=True);p.add_argument('--controller-commit',required=True);p.add_argument('--output');a=p.parse_args()
    proof=read(a.controller_proof)
    if not re.fullmatch('[0-9a-f]{40}',a.controller_commit):p.error('full controller commit required')
    if proof.get('published_control_commit')!=a.controller_commit or proof.get('all_blobs_full_sha_match') is not True or proof['files'].get('reproduction/compatible_async_evaluation/submit.py')!=sha(__file__):raise ValueError('INDEPENDENTLY_PUBLISHED_EXECUTION_CONTROL_REQUIRED')
    if a.phase=='capture-trainer-terminal':
        if not a.output or not a.prepare_receipt_sha:p.error('capture needs pinned prepare-receipt-sha and an unused evaluation evidence output path')
        result=capture_prepared_terminal(a.base,a.prepare_receipt_sha,a.output)
    elif a.phase=='query':result=query_existing(Path(a.base).resolve())
    else:
        if not all((a.prepare_receipt_sha,a.trainer_terminal_proof,a.trainer_terminal_proof_sha)):p.error('submit requires pinned prepare receipt and trainer terminal proof')
        result=submit(a.base,a.prepare_receipt_sha,a.trainer_terminal_proof,a.trainer_terminal_proof_sha,a.controller_commit)
    print(json.dumps(result))

if __name__=='__main__':main()
