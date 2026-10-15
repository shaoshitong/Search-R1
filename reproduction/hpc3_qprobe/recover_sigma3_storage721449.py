"""Recover only the observed pre-model storage-permission failure721449."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from reproduction.hpc3_qprobe.sigma3_experiment import BASE, ROOT, _sha
from reproduction.hpc3_qprobe.control import publish,link

JOB='721449'
COMMIT='aca0e779e94ddc3fbd50e49c3d59e2e889fbcf12'
BLOCKED=Path('/data/user/jxu957/experiments/cosearch-continue300-x203-20261008/runtime/cosearch-final200-four-gpu-20261015/cache/tmp/tmp90_ovm6e')


def recover():
    if os.getuid()!=14090:raise ValueError('SIGMA3_RECOVERY_OWNER_DIFFERS')
    accounting=subprocess.check_output(['sacct','-nP','-X','-j',JOB,'--format=JobID,State,ExitCode,UID,NodeList'],text=True,timeout=20).strip()
    if accounting!='721449|FAILED|1:0|14090|ACD1-56':raise ValueError('SIGMA3_NOT_ACTUAL_FAILED_OWN_JOB')
    prior=json.loads((BASE/'preparation/training-submission.json').read_text())
    claim=json.loads((BASE/'preparation/training-submit-claim.json').read_text())
    if prior.get('job')!=JOB or prior.get('returncode')!=0 or prior['commit']!=COMMIT or claim['commit']!=COMMIT:
        raise ValueError('SIGMA3_ORIGINAL_SUBMISSION_DIFFERS')
    source=ROOT/'source'/COMMIT
    expected=[str(source/'reproduction/hpc3_qprobe/train.sh'),str(BASE),COMMIT]
    if claim['command'][-3:]!=expected:raise ValueError('SIGMA3_ORIGINAL_ENTRY_DIFFERS')
    original=BASE/'logs'/('training-'+JOB+'.log');raw=original.read_bytes()
    signature=('PERSONAL_STORAGE_USAGE_NOT_VERIFIED: du: cannot read directory '+repr(str(BLOCKED))+': Permission denied').encode()
    if signature not in raw:raise ValueError('SIGMA3_NOT_OBSERVED_STORAGE_PERMISSION_FAILURE')
    forbidden=['logs/run-manifest.json','logs/native-allocation-admission.json','logs/training-launch-claim.json',
        'logs/actual-updates.jsonl','artifacts/question-probe/teacher-warmup-ready.json']
    if any((BASE/n).exists() for n in forbidden) or list((BASE/'logs').glob('optimizer-rank-*')):
        raise ValueError('SIGMA3_ACTOR_WORK_ALREADY_EXISTS_REQUIRE_FULL_STATE_RECOVERY')
    preserved=BASE/'failed-job-721449';preserved.mkdir(exist_ok=True)
    log=preserved/'training.log'
    if log.exists():
        if log.read_bytes()!=raw:raise ValueError('SIGMA3_PRESERVED_FAILURE_LOG_CHANGED')
    else:shutil.copy2(original,log)
    proof=ROOT/'preparation'/('source-'+COMMIT+'-verification.json');v=json.loads(proof.read_text())
    if v['status']!='PUBLISHED_SOURCE_FULL_SHA_PASS' or v['commit']!=COMMIT:raise ValueError('SIGMA3_SOURCE_PUBLICATION_DIFFERS')
    for name,sha in v['files'].items():
        if _sha(source/name)!=sha:raise ValueError('SIGMA3_IMMUTABLE_RECOVERY_SOURCE_DIFFERS')
    check=subprocess.run(['du','-s','-B1',str(BLOCKED)],capture_output=True,text=True,timeout=30)
    if check.returncode or check.stderr.strip():raise ValueError('SIGMA3_EXACT_BLOCKED_DIRECTORY_STILL_UNREADABLE: '+check.stderr)
    failure=dict(accounting=accounting,original_submission=prior,original_claim=claim,
        original_log_sha256=hashlib.sha256(raw).hexdigest(),actor_started=False,formal_outers=0,
        exact_directory_readback=check.stdout,teacher_copy_preserved=True,checked_unix=time.time())
    preserve=preserved/'failure-and-recovery.json'
    if not preserve.exists():publish(preserve,failure)
    stem='recovery721449';intent=BASE/'preparation'/(stem+'-submit-claim.json');receipt=BASE/'preparation'/(stem+'-submission.json')
    if intent.exists() or receipt.exists():raise ValueError('SIGMA3_RECOVERY_ALREADY_ATTEMPTED_INSPECT_NO_REPLAY')
    queued=subprocess.check_output(['squeue','-h','-u','jxu957','-o','%i|%T|%Z'],text=True,timeout=20)
    if any(line.split('|')[-1]==str(BASE) for line in queued.splitlines()):raise ValueError('SIGMA3_EXISTING_OWN_BASE_JOB')
    command=['sbatch','--parsable','--account=jxu957','--qos=formal-user','--partition=emergency_acd',
        '--nodes=1','--ntasks=1','--gres=gpu:4','--cpus-per-task=48','--mem=512G','--time=7-00:00:00',
        '--exclude=ACD1-31,ACD1-67','--job-name=qprobe-sigma3-200','--export=NONE','--chdir='+str(BASE),
        '--output='+str(BASE/'logs/training-%j.log'),'--error='+str(BASE/'logs/training-%j.log')]+expected
    test=subprocess.run(command[:1]+['--test-only']+command[1:],capture_output=True,text=True,timeout=20)
    publish(BASE/'preparation'/(stem+'-test-only.json'),dict(exit=test.returncode,stdout=test.stdout,stderr=test.stderr,epoch=time.time()))
    if test.returncode:raise ValueError('SIGMA3_RECOVERY_SCHEDULER_TEST_FAILED')
    publish(intent,dict(commit=COMMIT,command=command,recovery_of=JOB,failure=failure,epoch=time.time()))
    result=subprocess.run(command,capture_output=True,text=True,timeout=20)
    value=dict(commit=COMMIT,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,recovery_of=JOB,epoch=time.time())
    if result.returncode==0:
        value['job']=result.stdout.strip().split(';')[0]
        if not value['job'].isdecimal():raise ValueError('SIGMA3_BAD_RECOVERY_SUBMISSION')
    publish(receipt,value)
    if result.returncode:raise ValueError('SIGMA3_RECOVERY_SUBMISSION_FAILED_SAVED')
    return value


if __name__=='__main__':print(json.dumps(recover()),flush=True)
