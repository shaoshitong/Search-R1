"""Selected epoch68 inputs HKUST3→HPC3; credentials use RAM/stdin only."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time

from reproduction.prefix20_joint4.control import ASSETS, BASE, assets_ready, own_base, publish, storage_admission
from reproduction.prefix20_joint4.assets import sha
from reproduction.prefix20_joint4.deploy_source import IDENTITY

SOURCE='/backup/shaoshitong/qprobe-prefix20-hpc3-4gpu-20261016'
SOURCE_FILES={
    'head.safetensors':'assets/probe/head.safetensors',
    'probe-config.json':'assets/probe/probe-config.json',
    'eligible-question-pool.json':'canonical-data/selected/canonical/eligible-question-pool.json',
}

HK3_PROGRAM=r'''
import fcntl,hashlib,json,os,shlex,socket,subprocess,sys,time
from pathlib import Path
p=json.load(sys.stdin);password=p.pop('password');root=Path(p['source'])
assert socket.gethostname()=='xielab3'
def digest(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def save(path,value):
 with path.open('x') as out:json.dump(value,out,indent=2);out.flush();os.fsync(out.fileno())
def remote(command):
 a=subprocess.run(p['ssh']+[command],env=env,text=True,capture_output=True,timeout=90)
 if a.returncode:raise RuntimeError('FINAL_HOP_REMOTE_FAILED_'+str(a.returncode)+': '+a.stderr[-1500:])
 return json.loads(a.stdout)
with (root/'final-hop.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 files=[]
 for name,rel in p['source_files'].items():
  path=root/rel;size,expected=p['assets'][name]
  assert path.is_file() and not any(x.is_symlink() for x in (path,*path.parents))
  assert path.stat().st_size==size and digest(path)==expected
  files.append(dict(name=name,bytes=size,sha256=expected))
 stage=root/('final-hop-selected-'+p['commit']);stage.mkdir(exist_ok=False)
 for name,rel in p['source_files'].items():os.link(root/rel,stage/name)
 save(stage/'claim.json',dict(source_commit=p['commit'],files=files,pid=os.getpid(),started_unix=time.time()))
 env=os.environ.copy();env['SSHPASS']=password;password=None
 admitted=remote(p['preflight_command']);save(stage/'target-admitted.json',admitted)
 transport=shlex.join(p['ssh'])
 a=subprocess.run(['rsync','-a','--partial','--append-verify','--no-owner','--no-group','-e',transport,
  *[str(stage/x['name']) for x in files],p['peer']+':'+p['target']+'/'],env=env,capture_output=True,text=True,timeout=900)
 save(stage/'transport-result.json',dict(returncode=a.returncode,stdout=a.stdout,stderr=a.stderr,checked_unix=time.time()))
 if a.returncode:raise RuntimeError('FINAL_HOP_RSYNC_FAILED_PRESERVE_CLAIM_AND_PARTIALS')
 accepted=remote(p['accept_command']);save(stage/'complete.json',accepted)
 env.pop('SSHPASS',None)
 print(json.dumps(accepted),flush=True)
'''


def receive(base, commit, mode):
    base=own_base(base)
    if os.getuid()!=14090:raise ValueError('FINAL_HOP_JXU_TARGET_REQUIRED')
    source=base/'source'/commit
    manifest=json.loads((source/'source-publication.json').read_text())
    name='reproduction/prefix20_joint4/final_hop.py'
    if manifest['source_commit']!=commit or sha(Path(__file__))!=manifest['files'][name]['sha256']:
        raise ValueError('FINAL_HOP_RECEIVER_SOURCE_DIFFERS')
    root=base/'assets/prefix20';claim=base/'preparation/assets-final-hop-claim.json'
    if (base/'job.id').exists() or (base/'preparation/training-submit-claim.json').exists():
        raise ValueError('FINAL_HOP_CANNOT_MODIFY_SUBMITTED_RUN')
    if mode=='preflight':
        storage=storage_admission(base)
        if claim.exists() or (root.exists() and any(root.iterdir())):
            raise ValueError('FINAL_HOP_ALREADY_ATTEMPTED_INSPECT_EXISTING_CLAIM')
        root.mkdir(parents=True,exist_ok=True)
        value=dict(source_commit=commit,files=ASSETS,target=str(root),storage=storage,
            checked_unix=time.time(),route='HF→HKUST2→HKUST3→HPC3',training_job_submitted=False)
        publish(claim,value);return value
    d=json.loads(claim.read_text())
    if d['source_commit']!=commit or d['files']!={k:list(v) for k,v in ASSETS.items()}:
        raise ValueError('FINAL_HOP_TARGET_CLAIM_DIFFERS')
    value=dict(status='PREFIX20_SELECTED_INPUTS_FRESH_ALL3_SHA_PASS',source_commit=commit,
        assets=assets_ready(base),route='HF→HKUST2→HKUST3→HPC3',
        no_mac_asset_staging=True,source_staging_retained=True,checked_unix=time.time())
    publish(base/'preparation/assets-final-hop-complete.json',value);return value


def transfer(args):
    repo=Path(__file__).resolve().parents[2]
    commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],text=True).strip()
    if subprocess.check_output(['git','-C',str(repo),'status','--porcelain'],text=True):
        raise ValueError('FINAL_HOP_SOURCE_UNPUBLISHED')
    remote=subprocess.check_output(['git','-C',str(repo),'ls-remote','git@github.com:shaoshitong/Search-R1.git',
        'refs/heads/codex/qprobe-prefix20-hpc3-4gpu-20261016'],text=True).split()[0]
    if remote!=commit:raise ValueError('FINAL_HOP_SOURCE_NOT_PUSHED')
    spec=importlib.util.spec_from_file_location('authorized_bridge',args.bridge_source)
    bridge=importlib.util.module_from_spec(spec);spec.loader.exec_module(bridge)
    secret=Path('/Users/shaoshitong/Documents/remote_codex_control/memories/hpc3-credentials.private.md').read_text()
    line=next(x for x in secret.splitlines() if x.startswith('账号 jxu957'))
    password=re.split(r'密码\s*[:：]?\s*',line,1)[1].strip().strip('`');secret=line=None
    ssh=['sshpass','-e','ssh','-T','-oStrictHostKeyChecking=yes','-oControlMaster=no','-S','none',
        '-oHostKeyAlias=hpc3login.hpc.hkust-gz.edu.cn',
        '-oUserKnownHostsFile=/backup/shaoshitong/topd-hf-migration-20261009/hpc3-known-hosts',
        '-oConnectTimeout=8','jxu957@10.120.48.31']
    # rsync adds its peer itself; do not put a second destination in -e.
    peer=ssh[-1];transport=ssh[:-1]
    def command(mode):
        raw=shlex.join(['env','QPROBE_SOURCE='+BASE+'/source/'+commit,'PYTHONPATH='+BASE+'/source/'+commit,BASE+'/env/bin/python','-m',
            'reproduction.prefix20_joint4.final_hop','receive','--base',BASE,'--commit',commit,'--receive-mode',mode])
        return bridge.bound_command(raw,'10.120.48.31',IDENTITY)
    payload=dict(password=password,source=SOURCE,commit=commit,source_files=SOURCE_FILES,assets=ASSETS,
        ssh=transport+[peer],peer=peer,target=BASE+'/assets/prefix20',
        preflight_command=command('preflight'),accept_command=command('accept'))
    # Separate the complete ssh argv for remote commands from rsync's transport.
    program=HK3_PROGRAM.replace("transport=shlex.join(p['ssh'])","transport=shlex.join(p['ssh'][:-1])")
    a=bridge._campus_long('python3 -c '+shlex.quote(program),json.dumps(payload),1000)
    password=payload=None
    value=dict(returncode=a.returncode,stdout=a.stdout,stderr=a.stderr,source_commit=commit,checked_unix=time.time())
    Path(args.output).write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value));raise SystemExit(a.returncode)


def main():
    p=argparse.ArgumentParser();p.add_argument('mode',choices=('transfer','receive'))
    p.add_argument('--bridge-source');p.add_argument('--control-work');p.add_argument('--output')
    p.add_argument('--base');p.add_argument('--commit');p.add_argument('--receive-mode',choices=('preflight','accept'))
    a=p.parse_args()
    if a.mode=='transfer':transfer(a)
    else:print(json.dumps(receive(a.base,a.commit,a.receive_mode)))


if __name__=='__main__':main()
