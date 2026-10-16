"""Small local once-launch/status entry; only code and receipts on the Mac.

Integrate all four modules into the repository, commit/push, then run. This
entry verifies the exact remote Git SHA and the modern existing bridge source.
It never touches auth.json/SSH keys/network settings or training runtime code.
"""
import argparse
import base64
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time

try:
    from . import checkpoint_archive as core
    from . import archive_supervisor as held
except ImportError:
    import checkpoint_archive as core
    import archive_supervisor as held

BRANCH='refs/heads/codex/qprobe-prefix20-hpc3-4gpu-20261016'
REMOTE='git@github.com:shaoshitong/Search-R1.git'
MODERN_BRIDGE_SHA='a75a6a302f4ffed002be36371dc193048cdfcc1a016d99ae98acdbfb101a915f'

BOOTSTRAP_HK3=r'''
import base64,hashlib,json,os,socket,sys,time
from pathlib import Path
sys.dont_write_bytecode=True
p=json.load(sys.stdin);password=p.pop('password',None);config=p['config']
assert socket.gethostname()=='xielab3'
anchor=Path('/backup/shaoshitong/qprobe-prefix20-hpc3-4gpu-20261016/archive-controller/code')
assert len(config['code_commit'])==40 and all(c in '0123456789abcdef' for c in config['code_commit'])
root=anchor/config['code_commit']
assert not any(x.is_symlink() for x in (root,*root.parents))
if p['mode']=='launch' and not root.exists():
 root.parent.mkdir(parents=True,exist_ok=True);root.mkdir(exist_ok=False)
 assert set(p['contents'])==set(config['code_manifest'])=={'checkpoint_archive.py','archive_supervisor.py','archive_controller.py'}
 for name in sorted(p['contents']):
  raw=base64.b64decode(p['contents'][name],validate=True);record=config['code_manifest'][name]
  assert len(raw)==record['bytes'] and hashlib.sha256(raw).hexdigest()==record['sha256']
  with (root/name).open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 with (root/'code-publication.json').open('x') as f:
  json.dump(dict(code_commit=config['code_commit'],files=config['code_manifest']),f,sort_keys=True,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
 fd=os.open(root,os.O_RDONLY);os.fsync(fd);os.close(fd)
if not root.exists():
 print(json.dumps(dict(status='UNSTARTED_OR_UNKNOWN_CODE_ABSENT',training_job_changed=False)),flush=True);sys.exit(0)
sys.path.insert(0,str(root));import archive_supervisor as s
s.validate_config(config);s.verify_bundle(root,config)
value=s.launch_controller(config,password) if p['mode']=='launch' else s.controller_status(config)
password=None;p=None
print(json.dumps(value,allow_nan=False),flush=True)
'''


def published_bundle(repo,expected_commit):
    repo=Path(repo).resolve()
    core.need(re.fullmatch('[0-9a-f]{40}',expected_commit) is not None,'LAUNCH_FULL_CODE_COMMIT_REQUIRED')
    commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    core.need(commit==expected_commit,'LAUNCH_CHECKOUT_COMMIT_DIFFERS')
    core.need(not subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True),'LAUNCH_SOURCE_UNPUBLISHED_CHANGES')
    published=subprocess.check_output(['git','-C',str(repo),'ls-remote',REMOTE,BRANCH],text=True).strip().split()
    core.need(len(published)==2 and published[0]==commit and published[1]==BRANCH,'LAUNCH_LATEST_SOURCE_NOT_PUSHED')
    contents={};manifest={}
    for name in held.BUNDLE_NAMES:
        relative='reproduction/prefix20_joint4/'+name
        path=repo/relative
        core.regular(path)
        raw=subprocess.check_output(['git','-C',str(repo),'show',commit+':'+relative])
        core.need(path.read_bytes()==raw,'LAUNCH_WORKTREE_BYTES_DIFFER_FROM_COMMIT')
        contents[name]=base64.b64encode(raw).decode();manifest[name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    launcher=repo/'reproduction/prefix20_joint4/archive_launcher.py'
    core.need(Path(__file__).resolve()==launcher,'LAUNCH_MUST_USE_COMMITTED_ENTRY')
    core.need(subprocess.check_output(['git','-C',str(repo),'show',commit+':reproduction/prefix20_joint4/archive_launcher.py'])==launcher.read_bytes(),
        'LAUNCH_ENTRY_DIFFERS_FROM_COMMIT')
    return contents,manifest


def password_from_private_memory(path):
    # Existing protected credentials are read only at execution, never copied.
    content=Path(path).read_text();line=next(row for row in content.splitlines() if row.startswith('账号 jxu957'))
    password=re.split(r'密码\s*[:：]?\s*',line,1)[1].strip().strip('`');content=line=None
    core.need(password and '\n' not in password,'LAUNCH_RAM_PASSWORD_REQUIRED')
    return password


def run(args):
    output=Path(args.output).resolve();core.safe_components(output.parent)
    claim=output.with_name(output.name+'.launch-claim.json');mode=args.mode
    new_claim=not claim.exists()
    if not new_claim:
        old=core._read_json(claim);config=held.validate_config(old['config'])
        expected=dict(base=str(core.own_base(args.base)),training_commit=args.training_commit,
            source_job=args.source_job,code_commit=args.code_commit)
        core.need(old['config_sha256']==core.document_sha(config) and all(config[k]==v for k,v in expected.items()),
            'LOCAL_ONCE_LAUNCH_BINDING_DIFFERS')
        contents={};mode='status'  # Read existing work even when current checkout later advances.
    else:
        core.need(mode=='launch','LOCAL_LAUNCH_CLAIM_REQUIRED_FOR_STATUS')
        contents,manifest=published_bundle(args.repo,args.code_commit)
        config=dict(schema_version=1,base=str(core.own_base(args.base)),training_commit=args.training_commit,
            source_job=args.source_job,code_commit=args.code_commit,code_manifest=manifest,
            controller_root=str(held.HK_ROOT/'archive-controller'/('job-'+args.source_job)),hpc_identity=held.HPC_IDENTITY,
            known_hosts=held.KNOWN_HOSTS,ssh_peer=held.PEER,poll_seconds=60,source_wait_deadline_seconds=10*86400,transport_stall_seconds=300)
        held.validate_config(config)
    bridge_path=Path(args.bridge_source);core.checked_file(bridge_path,MODERN_BRIDGE_SHA)
    spec=importlib.util.spec_from_file_location('existing_modern_account_bridge',bridge_path)
    bridge=importlib.util.module_from_spec(spec);spec.loader.exec_module(bridge)
    password=password_from_private_memory(args.credentials_file) if mode=='launch' else None
    if new_claim:
        core.publish(claim,dict(config=config,config_sha256=core.document_sha(config),code_commit=args.code_commit,
            remote_pushed_sha=args.code_commit,credentials_persisted=False,created_unix=time.time()))
    packet=dict(mode=mode,config=config,contents=contents if mode=='launch' else {},password=password)
    try:
        result=bridge._campus_long('python3 -B -c '+shlex.quote(BOOTSTRAP_HK3),json.dumps(packet),60)
    except Exception as error:
        result=subprocess.CompletedProcess([],125,'','Observation exception '+type(error).__name__+'; query existing claim, never replay launch')
    finally:password=packet=None
    value=dict(mode_requested=args.mode,mode_sent=mode,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr,
        config_sha256=core.document_sha(config),code_commit=args.code_commit,training_commit=args.training_commit,
        source_job=args.source_job,credentials_persisted=False,training_job_changed=False,checked_unix=time.time())
    if result.returncode==124:value['status']='LAUNCH_OBSERVATION_UNKNOWN_QUERY_EXISTING_CLAIM_NO_REPLAY'
    elif result.returncode:value['status']='LAUNCH_OR_STATUS_FAILED_INSPECT_DURABLE_CLAIM'
    else:
        try:value['remote']=json.loads(result.stdout)
        except ValueError:value['status']='REMOTE_OUTPUT_UNKNOWN_NO_REPLAY'
    core.publish(output.with_name(output.name+'.observation-'+str(time.time_ns())+'.json'),value)
    held.atomic(output,value)
    print(json.dumps(value,allow_nan=False));return result.returncode


def main():
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('launch','status'))
    parser.add_argument('--repo',required=True);parser.add_argument('--bridge-source',required=True)
    parser.add_argument('--code-commit',required=True);parser.add_argument('--training-commit',required=True)
    parser.add_argument('--base',required=True);parser.add_argument('--source-job',required=True);parser.add_argument('--output',required=True)
    parser.add_argument('--credentials-file',default='/Users/shaoshitong/Documents/remote_codex_control/memories/hpc3-credentials.private.md')
    args=parser.parse_args();raise SystemExit(run(args))


if __name__=='__main__':main()
