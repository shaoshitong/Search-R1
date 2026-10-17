"""Publish-first durable deployment of the append-only CPU100 joint component supplement."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import time

from .checkpoint_cpu_control import FILES


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('deploy','observe'));parser.add_argument('--receipts',required=True)
    args=parser.parse_args();source=Path(__file__).resolve().parents[2]
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=source,text=True).strip()
    remote_sha=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+branch],cwd=source,text=True).split()[0]
    assert remote_sha==commit,'Normal push is required before helper execution'
    names={name:name for name in (*FILES,'reproduction/prefix_diversity200/checkpoint100_joint_supplement.py')}
    for name in ('ORIGINAL_READER.log','READINESS.json','PARTIAL_ACTOR_RECEIPTS.json'):
        names['evidence/'+name]='reproduction/prefix_diversity200/control_evidence/acceptance100/'+name
    files={}
    for target,name in names.items():
        raw=subprocess.check_output(['git','show',commit+':'+name],cwd=source)
        files[target]=dict(data=base64.b64encode(raw).decode(),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
    payload=dict(commit=commit,files=files)
    script='''import pathlib,json,hashlib,base64,subprocess,time,os
p=PAYLOAD;b=pathlib.Path('/mnt/data/shaoshitong/search-g1-prefix-diversity200-20261016-f3ee1435e501-resume4');d=b/'acceptance100-joint-supplement-v1';unit='prefix-diversity-f3ee1435-checkpoint100-joint-supplement'
def read(path):return json.loads(path.read_text())
def once(path,value):
 with path.open('x')as f:json.dump(value,f,sort_keys=True);f.flush();os.fsync(f.fileno())
assert hashlib.sha256((b/'preparation/launch-config.json').read_bytes()).hexdigest()=='384cf0f516e7bf9ce3623bdf48e87a078f8eceaa5af5e03f238e04e86ae59ad8'
if PHASE=='deploy':
 assert hashlib.sha256((b/'preparation/joint-milestone-0100-independent-full-sha-cpu-once-v1.json').read_bytes()).hexdigest()=='f917d16cf75a658e1fa066775d60f9053929bd7d099cc7a5c40d18e2579cbdcf'
 assert not d.exists(),'Existing supplemental control must not be replayed';d.mkdir(mode=0o700)
 for name,row in p['files'].items():
  rel=pathlib.PurePosixPath(name);assert not rel.is_absolute()and'..'not in rel.parts
  raw=base64.b64decode(row['data']);assert hashlib.sha256(raw).hexdigest()==row['sha256'];target=d/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
 once(d/'deploy-receipt.json',dict(helper_commit=p['commit'],files={k:{'sha256':v['sha256'],'bytes':v['bytes']}for k,v in p['files'].items()},created_unix=time.time(),training_modified=False,original_failure_overwritten=False))
 cmd=['systemd-run','--user','--unit='+unit,'--property=MemoryMax=12G','--property=CPUQuota=100%','--property=Nice=19','--property=IOWeight=10','--property=RuntimeMaxSec=3600','--property=StandardOutput=append:'+str(d/'reader.log'),'--property=StandardError=append:'+str(d/'reader.log'),'--setenv=CUDA_VISIBLE_DEVICES=','--setenv=OMP_NUM_THREADS=1','--setenv=MKL_NUM_THREADS=1','--setenv=PYTHONPATH='+str(d),'--working-directory='+str(d),'/mnt/data/shaoshitong/search-r1-rl/tree-grpo-retained18-20261008/env/bin/python','-u','-m','reproduction.prefix_diversity200.checkpoint100_joint_supplement','--base',str(b),'--control',str(d)]
 r=subprocess.run(cmd,capture_output=True,text=True);once(d/'systemd-submit.json',dict(command=cmd,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,observed_unix=time.time()));assert r.returncode==0,r.stderr
r={'observed_unix':time.time(),'control_root':str(d),'helper_commit':p['commit']}
for name in ('deploy-receipt.json','systemd-submit.json','intent.json'):
 if(d/name).exists():r[name]=read(d/name)
if(d/'reader.log').exists():r['log_bytes']=(d/'reader.log').stat().st_size;r['log_tail']=(d/'reader.log').read_text()[-6000:]
r['systemd']=subprocess.run(['systemctl','--user','show',unit]+['--property='+key for key in ('ActiveState','SubState','Result','ExecMainPID','ExecMainStatus','MemoryCurrent','CPUUsageNSec')],capture_output=True,text=True).stdout
if(d/'proof.json').exists():r['proof']=read(d/'proof.json');r['proof_sha256']=hashlib.sha256((d/'proof.json').read_bytes()).hexdigest()
r['original_failure_sha256']=hashlib.sha256((b/'preparation/joint-milestone-0100-independent-full-sha-cpu-once-v1.json').read_bytes()).hexdigest()
print(json.dumps(r))
'''.replace('PAYLOAD',repr(payload)).replace('PHASE',repr(args.phase))
    from reproduction.h1008_qprobe.management_transport import remote
    out=Path(args.receipts);out.mkdir(parents=True,exist_ok=True)
    evidence=out/('joint100-supplement-'+args.phase+'-'+str(time.time_ns())+'.json')
    result=json.loads(remote(script,evidence,target='h100-10'))
    path=evidence.with_name(evidence.stem+'-result.json');path.write_text(json.dumps(result,indent=2)+'\n')
    small={k:v for k,v in result.items()if k not in ('proof','deploy-receipt.json')}
    if 'proof'in result:small['proof_status']=result['proof']['status'];small['proof_error']=result['proof'].get('error')
    print(json.dumps(dict(receipt=str(path),result=small)))


if __name__=='__main__':main()
