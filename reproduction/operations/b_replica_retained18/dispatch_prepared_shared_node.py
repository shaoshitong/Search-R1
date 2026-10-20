"""Once-only shared native dispatch on an allowed H100 using private retained18 paths."""
import argparse
import base64
import hashlib
import json
import subprocess
import time
from .control import REPO, OUT, remote


INNER = r'''
import pathlib,json,time,socket,subprocess
assert socket.gethostname()==NODE and NODE in('h100-5','h100-8','h100-10')
c='/mnt/data/sihe-share5/workspace/training_data_node_10/shaoshitong/'+NAME;p=pathlib.Path(c)
ready=json.loads((p/'preparation/native-recovery-complete.json').read_text());assert ready['outer']==OUTER and not ready['training_started']
assert not(p/'preparation/submit-intent.json').exists()
assert not(p/'preparation/target-node-dispatch-intent.json').exists()
q=subprocess.check_output(['squeue','-h','-n','sst-b-diverse-cp64-eight','-o','%T'],text=True).strip();assert not q,q
gpu=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=20);assert gpu.returncode==0,gpu.stdout+gpu.stderr
cards=[line.split(',')for line in gpu.stdout.splitlines()];assert sorted(int(r[0])for r in cards)==list(range(8))
footprint=[38546,36690,37428,41628,52502,32836,39296,44202]
assert all(int(r[3])-int(r[2])>=footprint[int(r[0])]for r in cards),'Actual shared GPU footprint currently does not fit'
stash=pathlib.Path('/run')/UNIT
env=['/usr/bin/env','CUDA_DEVICE_ORDER=PCI_BUS_ID','SEARCH_G1_PHYSICAL_WORLD_SIZE=8','SEARCH_G1_B_PREFIX_TOKENS=80','OMP_NUM_THREADS=2','MKL_NUM_THREADS=2','PYTHONDONTWRITEBYTECODE=1','PYTHONPATH='+c+'/repo']
common=['/usr/bin/unshare','--mount','--propagation','private','/bin/bash',str(stash/'private_shared_paths.sh')]+env+['/dataset/shaoshitong/Search-G1-common8/env/bin/python','-u']
command=['systemd-run','--unit='+UNIT+'.service','--service-type=exec','--property=Restart=no','--property=CPUQuota=3200%','--property=MemoryMax=256G']+common+[str(stash/'wait_prepared_shared.py'),'--base',c,'--activation',str(stash/'activate_prepared_shared.py'),'--activation-sha256',ACTIVATION_SHA]
intent=dict(time=time.time(),node=NODE,command=command,bootstrap_commit=COMMIT,native_commit=ready['source_commit'],files=FILES,inventory=gpu.stdout,own_queue=q,slurm_reserved_gpus=0)
with(p/'preparation/target-node-dispatch-intent.json').open('x')as f:json.dump(intent,f)
r=subprocess.run(command,capture_output=True,text=True,timeout=25);result=dict(time=time.time(),returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,unit=UNIT+'.service')
with(p/'preparation/target-node-dispatch-result.json').open('x')as f:json.dump(result,f)
assert r.returncode==0,result
resources=['systemd-run','--unit='+UNIT+'-resources.service','--service-type=exec','--property=Restart=no','--property=BindsTo='+UNIT+'.service','--property=After='+UNIT+'.service','--property=CPUQuota=100%','--property=MemoryMax=1G']+common+[str(stash/'wait_shared_resources.py'),'--base',c]
with(p/'preparation/target-node-resources-intent.json').open('x')as f:json.dump(dict(time=time.time(),command=resources),f)
r=subprocess.run(resources,capture_output=True,text=True,timeout=25);resource=dict(time=time.time(),returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,unit=UNIT+'-resources.service')
with(p/'preparation/target-node-resources-result.json').open('x')as f:json.dump(resource,f)
print(json.dumps(dict(node=NODE,main=result,resources=resource,files=FILES,native_commit=ready['source_commit'],bootstrap_commit=COMMIT,slurm_reserved_gpus=0)))
'''


def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--outer',required=True,type=int)
    p.add_argument('--node',required=True,choices=['h100-5','h100-8','h100-10']);p.add_argument('--unit',required=True);a=p.parse_args()
    if not a.name.startswith('search-g1-prefix-selection200-')or '/'in a.name or '..'in a.name or not a.unit.startswith('sst-b-diverse-')or any(x not in 'abcdefghijklmnopqrstuvwxyz0123456789-'for x in a.unit):raise ValueError('Task namespace/unit required')
    if subprocess.check_output(['git','status','--porcelain'],cwd=REPO):raise ValueError('Publish code before dispatch')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip()
    if subprocess.check_output(['git','ls-remote','origin','refs/heads/'+branch],cwd=REPO,text=True).split()[0]!=commit:raise ValueError('Source not pushed')
    files={};manifest=[]
    for name in('private_shared_paths.sh','activate_prepared_shared.py','wait_prepared_shared.py','wait_shared_resources.py'):
        raw=(REPO/'reproduction/operations/b_replica_retained18'/name).read_bytes();files[name]=base64.b64encode(raw).decode()
        manifest.append(dict(path='/run/'+a.unit+'/'+name,sha256=hashlib.sha256(raw).hexdigest()))
    constants=dict(NAME=a.name,OUTER=a.outer,NODE=a.node,UNIT=a.unit,COMMIT=commit,FILES=manifest,
        ACTIVATION_SHA=next(r['sha256']for r in manifest if r['path'].endswith('/activate_prepared_shared.py')))
    code='\n'.join(k+'='+repr(v)for k,v in constants.items())+'\n'+INNER
    expected={r['path'].rsplit('/',1)[1]:r['sha256']for r in manifest}
    script='BUNDLE='+repr(files)+'\nEXPECTED='+repr(expected)+'\nUNIT='+repr(a.unit)+'\nCODE='+repr(code)+'\n'+r'''
import pathlib,base64,subprocess,json,hashlib
p=pathlib.Path('/run')/UNIT;p.mkdir(exist_ok=False)
for name,data in BUNDLE.items():
 f=p/name;f.write_bytes(base64.b64decode(data));assert hashlib.sha256(f.read_bytes()).hexdigest()==EXPECTED[name]
command=['unshare','--mount','--propagation','private','bash',str(p/'private_shared_paths.sh'),'/usr/bin/python3','-c',CODE]
r=subprocess.run(command,capture_output=True,text=True,timeout=160)
if r.returncode:raise RuntimeError('DISPATCH_RECONCILE_EXISTING_INTENT '+r.stdout+r.stderr)
print(r.stdout)
'''
    f=OUT/('shared-node-dispatch-%d-transport.json'%time.time_ns())
    value=json.loads(remote(script,f,target=a.node,established_controller=True))
    (OUT/('CP%d_SHARED_NODE_DISPATCH_RESULT.json'%a.outer)).write_text(json.dumps(value,indent=2));print(json.dumps(value),flush=True)


if __name__=='__main__':main()
