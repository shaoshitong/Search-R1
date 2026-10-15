"""Authenticated backup admission and one exact78 CPU launch, RAM SSH credentials.

Timeout is unknown and never authorizes another launch. Use inspect thereafter.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import time

from reproduction.hpc3_qprobe import local_account_bridge as bridge
from reproduction.hpc3_qprobe import local_old_original_reserve as remote
from reproduction.hpc3_qprobe import retire_exact78_old_originals as r

BACKUP_COMMIT='afda9f708ad0d93312c162d3aa8452816c420f35'


def backup_admission():
    code="""import sys,json,time
from pathlib import Path
sys.path.insert(0,'/backup/shaoshitong/qprobe-old-original-reserve-20261016/source/afda9f708ad0d93312c162d3aa8452816c420f35')
from reproduction.hpc3_qprobe import reserve_old_probe_originals as r
v=r.inspect('afda9f708ad0d93312c162d3aa8452816c420f35')
p=r.TARGET.with_name('qprobe-old-original-reserve-durability-20261016')/'final/complete.json'
assert r.digest(p)=='1410139a59035dd45a228fce4ac6b8bc4f67591420dd07ef4de1a25bb1b202d1'
d=r.read_owned(p,1002)
assert v['actual_terminal_accepted'] and v['current_target_metadata_full_set_verified']
assert v['complete_sha256']==d['backup_complete_sha256']=='ba51bc0dc2511117bfef6072832d3ee28f475b0828fb6ff2037d1d52b2f98739'
assert d['status']=='ACTUAL_COMPLETED_FULL_SHA_BACKUP_OWNED_FILES_AND_DIRECTORIES_FSYNC_PASS'
assert d['verified_files']==118 and d['verified_bytes']==121296682925 and d['synced_directories']==63
print(json.dumps(dict(status='AUTHENTICATED_HKUST3_TERMINAL_RESERVE_AND_DURABILITY_PASS',
 observed_unix=time.time(),target_host=r.identity(),backup_complete_sha256=v['complete_sha256'],
 durability_sha256=r.digest(p),backup_files=118,backup_bytes=121296682925,
 target_metadata_full_set_verified=True,actual_terminal_accepted=True,original_inspection=v)))
"""
    actual=remote.ssh('env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 python3 -c '+shlex.quote(code),'')
    r.need(actual.returncode==0 and not actual.stderr,'actual HKUST3 backup admission SSH failed')
    value=json.loads(actual.stdout);value['actual_ssh_exit_code']=actual.returncode
    return value


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('start','inspect'))
    parser.add_argument('--commit',required=True);parser.add_argument('--control-work',type=Path,required=True)
    parser.add_argument('--audit-dir',type=Path,required=True);args=parser.parse_args()
    a=backup_admission() if args.mode=='start' else None
    payload=json.dumps(a,sort_keys=True)+'\n' if a is not None else ''
    repo=r.ROOT/'source'/args.commit
    code="""import sys,json,hashlib
from reproduction.hpc3_qprobe import retire_exact78_old_originals as r
from reproduction.hpc3_qprobe import reserve_old_probe_originals as s
commit=sys.argv[1];mode=sys.argv[2];r.code_guard(commit)
if mode=='start':
 raw=sys.stdin.buffer.read();value=json.loads(raw);sha=hashlib.sha256(raw).hexdigest()
 if r.DIRECTORY.exists():out=r.inspect(commit)
 else:
  if r.ADMISSION.exists():assert r.ADMISSION.read_bytes()==raw,'Unknown prior admission; inspect only'
  else:
   with r.ADMISSION.open('xb') as f:f.write(raw);f.flush();__import__('os').fsync(f.fileno())
  out=r.start(commit,sha,True)
else:out=r.inspect(commit)
print(json.dumps(out))
"""
    env=['env','QPROBE_SOURCE='+str(repo),'PYTHONPATH='+str(repo),'CUDA_VISIBLE_DEVICES=','PYTHONDONTWRITEBYTECODE=1']
    env += [name+'=1' for name in remote.r.CAPS]
    command=' '.join(shlex.quote(s) for s in env+[str(r.reserve.BASE/'env/bin/python'),'-c',code,args.commit,args.mode])
    actual=bridge.run('jxu957',command,payload,str(args.control_work),login_host='10.120.48.31',
        expected_host_identity=remote.r.plan()['source_host'])
    audit=dict(mode=args.mode,source_commit=args.commit,returncode=actual.returncode,
        stdout=actual.stdout,stderr=actual.stderr,admission=a,observed_unix=time.time())
    args.audit_dir.mkdir(parents=True,exist_ok=True)
    path=args.audit_dir/('exact78-retirement-'+args.mode+'-'+str(time.time_ns())+'.json')
    path.write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    r.need(actual.returncode==0,'actual HPC3 request failed or unknown; inspect '+str(path))
    print(json.dumps(dict(audit=str(path),observed=json.loads(actual.stdout))))


if __name__=='__main__':main()
