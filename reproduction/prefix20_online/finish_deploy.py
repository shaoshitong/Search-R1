"""Finish one copied deployment using pinned native evidence after Slurm purge.

This is a separate published control helper. It never patches frozen runtime
files, repeats sbatch, or transfers models/rollouts through the local machine.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile
import time

REPO = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', required=True)
    parser.add_argument('--runtime-commit', required=True)
    parser.add_argument('--config', required=True)
    parser.add_argument('--terminal-evidence', required=True)
    parser.add_argument('--terminal-evidence-sha256', required=True)
    parser.add_argument('--management-receipt', required=True)
    parser.add_argument('--receipts', required=True)
    args = parser.parse_args()
    archive = subprocess.check_output(['git','-C',str(REPO),'archive','--format=tar',args.runtime_commit])
    with tarfile.open(fileobj=io.BytesIO(archive)) as source:
        files = {entry.name:hashlib.sha256(source.extractfile(entry).read()).hexdigest()
                 for entry in source if entry.isfile()}
    config = json.loads(Path(args.config).read_text())
    config.update(base=args.base, source_commit=args.runtime_commit)
    control_commit = subprocess.check_output(['git','-C',str(REPO),'rev-parse','HEAD'],text=True).strip()
    payload = dict(base=args.base, runtime_commit=args.runtime_commit, control_commit=control_commit,
                   config=config, files=files, terminal_path=args.terminal_evidence,
                   terminal_sha256=args.terminal_evidence_sha256)
    code = r'''
import fcntl,hashlib,json,os,pathlib,sys,time
p=PAYLOAD;b=pathlib.Path(p['base']);prep=b/'preparation'
def read(path):return json.loads(path.read_text())
def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(8*1024**2),b''):h.update(block)
 return h.hexdigest()
def once(path,value):
 with path.open('x') as f:json.dump(value,f,sort_keys=True,allow_nan=False);f.flush();os.fsync(f.fileno())
assert str(b).startswith('/mnt/data/shaoshitong/search-g1-prefix20-online1000-')
with (prep/'finish-deploy-control.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (prep/'deploy-receipt.json').exists():
  print(json.dumps(dict(status='EXISTING_DEPLOYMENT_INSPECT_NO_REPLAY',receipt=read(prep/'deploy-receipt.json'))));raise SystemExit
 if (prep/'finish-deploy-intent.json').exists() or ((prep/'restore-intent.json').exists() and not (prep/'restore-receipt.json').exists()):
  print(json.dumps(dict(status='PARTIAL_FINISH_OR_RESTORE_INSPECT_NO_REPLAY')));raise SystemExit
 intent=read(prep/'deploy-intent.json');c=p['config']
 assert intent['commit']==p['runtime_commit'] and intent['run_id']==c['run_id'] and intent['file_count']==len(p['files'])
 assert not (prep/'launch-config.json').exists() and not (prep/'source-files.json').exists()
 for name,digest in p['files'].items():assert sha(b/'repo'/name)==digest,name
 source=pathlib.Path(c['restore']['source_base'])
 evidence=pathlib.Path(p['terminal_path'])
 assert evidence.parent==source/'preparation' and sha(evidence)==p['terminal_sha256']
 saved=read(evidence);native=saved.get('native') or saved.get('facts',{}).get('native')
 assert isinstance(native,str),'Pinned terminal evidence needs original native record'
 sys.path.insert(0,str(b/'repo'))
 from reproduction.common8.native_allocation import fields
 terminal=fields(native);job=str(read(source/'preparation/submit-receipt.json')['job_id'])
 assert terminal['JobId']==job and terminal['JobState'] in ('FAILED','CANCELLED','TIMEOUT','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED','COMPLETED')
 if c['restore'].get('source_job_id') is not None:assert str(c['restore']['source_job_id'])==job
 if c['restore'].get('expected_source_exit_code') is not None:assert terminal['ExitCode']==c['restore']['expected_source_exit_code']
 assert terminal['Comment']=='g1-prefix20-online:'+c['run_id']
 assert terminal['Command']==str(source/'preparation/native-launch.sh') and terminal['WorkDir']==str(source)
 from reproduction.prefix_only_correctness_ab.launch import birth_states
 ledger=read(source/'logs/process-ledger.json');births=[row for entry in ledger['jobs'].values() for row in entry['ledger'].values()]
 assert not birth_states(births),'Actual source birth still present'
 assert sha(pathlib.Path(c['cached_features'])/'manifest.json')==c['cached_features_manifest_sha256']
 once(prep/'finish-deploy-intent.json',dict(runtime_commit=p['runtime_commit'],control_commit=p['control_commit'],run_id=c['run_id'],terminal_evidence=str(evidence),terminal_evidence_sha256=p['terminal_sha256'],existing_deploy_intent_sha256=sha(prep/'deploy-intent.json'),created_unix=time.time()))
 from reproduction.prefix20_online.recovery import stage
 restored=stage(c,terminal=terminal)
 once(prep/'launch-config.json',c);once(prep/'source-files.json',p['files'])
 result=dict(status='CODE_DEPLOYED_RECOVERY_STAGED_NOT_SUBMITTED',base=str(b),commit=p['runtime_commit'],control_commit=p['control_commit'],run_id=c['run_id'],target_code_files=len(p['files']),all_target_code_full_sha_match=True,cached_features_manifest_sha256=c['cached_features_manifest_sha256'],config_sha256=sha(prep/'launch-config.json'),models_or_rollouts_copied_from_Mac=False,remote_checkpoint_staged=True,completed_epoch=restored['completed_epoch'],start_epoch=restored['start_epoch'],head_sha256=restored['head_sha256'],reserved_pending_selection_reused=restored['reserved_pending_selection_reused'],source_job_id=job,source_native_state=terminal['JobState'],source_native_exit_code=terminal['ExitCode'],target_learning_rate=c['learning_rate'],terminal_source='pinned_original_native_record_after_Slurm_purge',terminal_evidence_sha256=p['terminal_sha256'],native_submission_attempted=False,checked_unix=time.time())
 once(prep/'deploy-receipt.json',result);once(prep/'finish-deploy-receipt.json',result)
 print(json.dumps(result))
'''.replace('PAYLOAD', repr(payload))
    from reproduction.h1008_qprobe.management_transport import remote
    socket = json.loads(Path(args.management_receipt).read_text())['socket']
    receipt = Path(args.receipts)/('prefix20-finish-deploy-'+str(time.time_ns())+'.json')
    result = remote(code, receipt, control_path=socket)
    print(json.dumps(dict(receipt=str(receipt), result=json.loads(result)), allow_nan=False))


if __name__ == '__main__':
    main()
