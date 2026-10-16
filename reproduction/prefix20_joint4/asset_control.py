"""Mac control only: publish public asset code; send HF credential over stdin."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

ROOT='/data/shaoshitong/qprobe-prefix20-hpc3-4gpu-20261016/assets'

REMOTE = r'''
import base64,fcntl,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path
p=json.load(sys.stdin);token=p.pop('token');root=Path(p['root'])
root.mkdir(parents=True,exist_ok=True)
control_lock=(root/'control.lock').open('a');fcntl.flock(control_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
if (root/'launch-claim.json').exists():
 if not p['resume_failed']:raise ValueError('EXISTING_ASSET_CLAIM_INSPECT_BEFORE_START')
 if (root/'complete.json').exists():raise ValueError('ASSETS_ALREADY_COMPLETE')
 previous=json.loads((root/'launch.json').read_text());process=Path('/proc/'+str(previous['pid'])+'/stat')
 boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
 if process.exists() and boot==previous['boot_id']:
  fields=process.read_text().rsplit(')',1)[1].split()
  if int(fields[19])==previous['start_ticks'] and fields[0]!='Z':raise ValueError('OWNED_ASSET_PROCESS_STILL_ACTIVE')
 lines=(root/'download.log').read_text().splitlines()
 failure=json.loads(lines[-1]) if lines else {}
 if failure.get('status') not in ('ASSET_COMPONENT_FAILED','CANONICAL_ASSET_COMPONENT_FAILED'):
  raise ValueError('NO_ACTUAL_TERMINAL_ASSET_FAILURE_EVIDENCE')
 with (root/'assets.lock').open('a') as asset_lock:
  fcntl.flock(asset_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  failed=root/('failed-attempt-'+str(time.time_ns()));failed.mkdir()
  preserved=[]
  for name in ('launch-claim.json','launch.json','source-publication.json','download.log','assets.py',
               'canonical_assets.py','canonical-asset-plan.json','probe-asset-plan.json','inventory-error.json'):
   target=root/name
   if target.is_file():
    raw=target.read_bytes();preserved.append(dict(path=name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
    shutil.move(str(target),str(failed/name))
  with (failed/'failure-preservation.json').open('x') as f:
   json.dump(dict(previous_process=previous,terminal_failure=failure,files=preserved,checked_unix=time.time()),f,indent=2)
elif p['resume_failed']:raise ValueError('NO_PREVIOUS_FAILED_ASSET_CLAIM')
files=[]
for name,value in p['files'].items():
 raw=base64.b64decode(value);target=root/name;digest=hashlib.sha256(raw).hexdigest()
 if target.exists():
  if hashlib.sha256(target.read_bytes()).hexdigest()!=digest:raise ValueError('DEPLOYED_SOURCE_DIFFERS')
 else:
  with target.open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 files.append(dict(path=str(target),sha256=digest,bytes=len(raw)))
publication=dict(source_commit=p['source_commit'],remote_source_files=files,checked_unix=time.time())
with (root/'source-publication.json').open('x') as f:json.dump(publication,f,indent=2)
with (root/'launch-claim.json').open('x') as f:json.dump(dict(source_commit=p['source_commit'],created_unix=time.time()),f)
argv=['/usr/bin/python3',str(root/p['entry']),'--root',str(root),'--plan',str(root/p['plan_name']),'--proxy','http://127.0.0.1:17898']
with (root/'download.log').open('ab') as out:
 child=subprocess.Popen(argv,stdin=subprocess.PIPE,stdout=out,stderr=out,start_new_session=True)
 child.stdin.write((token+'\n').encode());child.stdin.close()
start_ticks=int(Path('/proc/'+str(child.pid)+'/stat').read_text().rsplit(')',1)[1].split()[19])
receipt=dict(publication,pid=child.pid,start_ticks=start_ticks,boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),root=str(root),token_persisted=False,training_job_submitted=False)
with (root/'launch.json').open('x') as f:json.dump(receipt,f,indent=2)
print(json.dumps(receipt))
'''


def main():
    p=argparse.ArgumentParser();p.add_argument('--source-commit',required=True)
    p.add_argument('--kind',choices=('probe','canonical'),default='probe')
    p.add_argument('--resume-failed',action='store_true')
    p.add_argument('--receipt',required=True);a=p.parse_args()
    here=Path(__file__).resolve().parent
    head=subprocess.check_output(['git','-C',str(here),'rev-parse','HEAD'],text=True).strip()
    remote=subprocess.check_output(['git','-C',str(here),'ls-remote',
        'git@github.com:shaoshitong/Search-R1.git',
        'refs/heads/codex/qprobe-prefix20-hpc3-4gpu-20261016'],text=True).split()[0]
    if head!=a.source_commit or remote!=head:raise ValueError('LATEST_SOURCE_NOT_PUBLISHED')
    token=Path.home().joinpath('.cache/huggingface/token').read_text().strip()
    entry='assets.py' if a.kind=='probe' else 'canonical_assets.py'
    plan_name='probe-asset-plan.json' if a.kind=='probe' else 'canonical-asset-plan.json'
    names=('assets.py',plan_name) if a.kind=='probe' else ('assets.py','canonical_assets.py',plan_name)
    payload=dict(token=token,root=ROOT if a.kind=='probe' else ROOT.rsplit('/',1)[0]+'/canonical-data',source_commit=head,
                 resume_failed=a.resume_failed,
                 entry=entry,plan_name=plan_name,
                 files={name:base64.b64encode((here/name).read_bytes()).decode()
                        for name in names})
    result=subprocess.run(['ssh','-T','-o','BatchMode=yes','-o','ConnectTimeout=10',
        'hkust2','python3 -c '+shlex.quote(REMOTE)],input=json.dumps(payload),
        text=True,capture_output=True,timeout=40)
    if result.returncode:
        raise RuntimeError('REMOTE_ASSET_LAUNCH_FAILED; inspect remote owned claim before retry')
    receipt=json.loads(result.stdout)
    Path(a.receipt).write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))


if __name__=='__main__':main()
