"""Preserve only the terminal 730961 pre-model bootstrap failure and small logs."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

BASE = Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-recovery5-after730912-20261018')
JOB = '730961'
COMMIT = '8033adb58beb85bbf9ec5815486ca710a77d3177'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def publish(path, value):
    with Path(path).open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True); stream.write('\n')
        stream.flush(); os.fsync(stream.fileno())


def preserve():
    target = BASE/('failed-job-'+JOB); receipt = target/'failure-evidence.json'
    if receipt.is_file():
        value = json.loads(receipt.read_text())
        if value['source_job'] != JOB or value['source_commit'] != COMMIT:
            raise ValueError('Preserved730961 identity differs')
        for row in value['files']:
            if sha(target/row['relative_path']) != row['sha256']:
                raise ValueError('Preserved730961 SHA differs')
        return value
    if target.exists(): raise ValueError('Partial730961 preservation exists; inspect without overwriting')
    result = subprocess.run(['sacct','-j',JOB,'--format=JobIDRaw,State,ExitCode,NodeList,Start,End',
        '--noheader','--parsable2'], capture_output=True, text=True, timeout=30)
    rows = {row[0]:row[1:] for row in (line.split('|') for line in result.stdout.splitlines()) if len(row)>=6}
    expected = {JOB:('FAILED','1:0'),JOB+'.batch':('FAILED','1:0'),JOB+'.extern':('COMPLETED','0:0')}
    if result.returncode or any(tuple(rows.get(job,[])[:2])!=state or rows[job][2]!='ACD1-13'
            for job,state in expected.items()):
        raise ValueError('730961 terminal accounting differs')
    source = json.loads((BASE/'preparation/run-source.json').read_text())
    run = json.loads((BASE/'logs/run-manifest.json').read_text())
    binding = run.get('full_state_restore',{})
    if ((BASE/'job.id').read_text().strip()!=JOB or source['source_commit']!=COMMIT or
            run['source_commit']!=COMMIT or run['completed_updates']!=0 or
            binding.get('kind')!='easy2560_native4' or binding.get('completed_outer')!=5 or
            run.get('initial_probe_version')!=binding.get('reward_probe_version') or
            not run.get('initial_probe_version')):
        raise ValueError('730961 actual saved probe bootstrap identity differs')
    log = BASE/('logs/training-'+JOB+'.log')
    if "run['initial_probe_version'] is None" not in log.read_text() or 'AssertionError' not in log.read_text():
        raise ValueError('730961 actual entry assertion differs')
    names = ['job.id','preparation/training-submit-claim.json','preparation/training-submission.json',
             'preparation/run-source.json','logs/run-manifest.json','logs/easy2560-configuration-prepared.json',
             'logs/training-'+JOB+'.log']
    names += [str(path.relative_to(BASE)) for path in sorted((BASE/'logs').glob('native-ray-resources-*.json'))]
    if ((BASE/'logs/process-ledger.json').exists() or
            (BASE/'logs/native-allocation-admission.json').exists() or
            any((BASE/'logs').glob('full-state-restored-*.json')) or
            any((BASE/'logs').glob('optimizer-rank-*.jsonl'))):
        raise ValueError('730961 has model/runtime evidence beyond the observed bootstrap failure')
    target.mkdir(mode=0o770)
    publish(target/'terminal-observation.json',dict(checked_unix=time.time(),source_job=JOB,
        source_commit=COMMIT,accounting=dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)))
    records = []
    for name in names:
        original = BASE/name; copied = target/name
        if original.is_symlink() or not original.is_file() or original.stat().st_size>2*1024**2:
            raise ValueError('Only actual small730961 files are permitted')
        before = sha(original); copied.parent.mkdir(parents=True,exist_ok=True)
        with original.open('rb') as src, copied.open('xb') as dst:
            shutil.copyfileobj(src,dst); dst.flush(); os.fsync(dst.fileno())
        if sha(original)!=before or sha(copied)!=before: raise ValueError('730961 copy SHA differs')
        records.append(dict(relative_path=name,bytes=copied.stat().st_size,sha256=before))
    value = dict(status='FAILED730961_SAVED_PROBE_FRESH_ENTRY_ASSERTION_PRESERVED',source_job=JOB,
        source_commit=COMMIT,source_base=str(BASE),files=records,checked_unix=time.time(),
        operator_sha256=sha(__file__),actual_new_training_updates=0,actual_cuda_restore_accepted=False,
        actual_completed_updates_field=run['completed_updates'],restored_boundary=binding['completed_outer'],
        rejected_initial_probe_version=run['initial_probe_version'],tensors_copied=False,
        original_evidence_modified=False,env_or_repo_modified=False,jobs_submitted=0)
    publish(receipt,value)
    fd=os.open(target,os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)
    return value


if __name__=='__main__':
    result=preserve()
    print(json.dumps(dict(status=result['status'],files=len(result['files']),
        bytes=sum(row['bytes'] for row in result['files']),receipt_sha256=sha(BASE/('failed-job-'+JOB)/'failure-evidence.json'),
        actual_cuda_restore_accepted=False,jobs_submitted=0)))
