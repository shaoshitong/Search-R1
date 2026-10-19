"""Dispatch one prepared recovery on retained18; never retry unknown submission."""
import argparse
import hashlib
import json
import subprocess
import time
from .control import REPO, OUT, remote


SCRIPT = r'''
import pathlib,json,subprocess,hashlib,time,socket
assert socket.gethostname()=='h100-10'
p=pathlib.Path('/mnt/data/shaoshitong')/NAME;c='/mnt/data/sihe-share5/workspace/training_data_node_10/shaoshitong/'+NAME
ready=json.loads((p/'preparation/native-recovery-complete.json').read_text());assert ready['outer']==OUTER and not ready['training_started']
assert not(p/'preparation/submit-intent.json').exists()
assert not(p/'preparation/retained18-dispatch-intent.json').exists()
queue=subprocess.check_output(['squeue','-h','-n','sst-b-diverse-cp64-eight','-o','%T'],text=True).strip();assert not queue,queue
gpu=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.total','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=20);assert gpu.returncode==0,gpu.stdout+gpu.stderr
cards=[line.split(',')for line in gpu.stdout.splitlines()];assert len(cards)==8
footprint=[38546,36690,37428,41628,52502,32836,39296,44202]
assert all(int(row[3])-int(row[2])>=footprint[int(row[0])]for row in cards),'Observed actor memory footprint does not currently fit'
stash=pathlib.Path('/run')/UNIT;stash.mkdir(exist_ok=False)
names=['private_shared_paths.sh','activate_prepared_shared.py','wait_prepared_shared.py','wait_shared_resources.py']
files=[]
for name in names:
 source=p/'repo/reproduction/operations/b_replica_retained18'/name;target=stash/name;target.write_bytes(source.read_bytes());files.append(dict(path=str(target),sha256=hashlib.sha256(target.read_bytes()).hexdigest()))
activation=next(x['sha256']for x in files if x['path'].endswith('/activate_prepared_shared.py'))
env=['/usr/bin/env','CUDA_DEVICE_ORDER=PCI_BUS_ID','SEARCH_G1_PHYSICAL_WORLD_SIZE=8','SEARCH_G1_B_PREFIX_TOKENS=80','OMP_NUM_THREADS=2','MKL_NUM_THREADS=2','PYTHONDONTWRITEBYTECODE=1','PYTHONPATH='+c+'/repo']
common=['/usr/bin/unshare','--mount','--propagation','private','/bin/bash',str(stash/'private_shared_paths.sh')]+env+['/dataset/shaoshitong/Search-G1-common8/env/bin/python','-u']
command=['systemd-run','--unit='+UNIT+'.service','--service-type=exec','--property=Restart=no','--property=CPUQuota=3200%','--property=MemoryMax=256G']+common+[str(stash/'wait_prepared_shared.py'),'--base',c,'--activation',str(stash/'activate_prepared_shared.py'),'--activation-sha256',activation]
intent=dict(time=time.time(),command=command,bootstrap_commit=COMMIT,native_commit=ready['source_commit'],files=files,inventory=gpu.stdout,own_queue=queue)
with(p/'preparation/retained18-dispatch-intent.json').open('x')as h:json.dump(intent,h)
r=subprocess.run(command,capture_output=True,text=True,timeout=25);result=dict(time=time.time(),returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,unit=UNIT+'.service')
with(p/'preparation/retained18-dispatch-result.json').open('x')as h:json.dump(result,h)
assert r.returncode==0,result
resources=['systemd-run','--unit='+UNIT+'-resources.service','--service-type=exec','--property=Restart=no','--property=BindsTo='+UNIT+'.service','--property=After='+UNIT+'.service','--property=CPUQuota=100%','--property=MemoryMax=1G']+common+[str(stash/'wait_shared_resources.py'),'--base',c]
with(p/'preparation/retained18-resources-intent.json').open('x')as h:json.dump(dict(command=resources,time=time.time()),h)
r=subprocess.run(resources,capture_output=True,text=True,timeout=25);resource=dict(time=time.time(),returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,unit=UNIT+'-resources.service')
with(p/'preparation/retained18-resources-result.json').open('x')as h:json.dump(resource,h)
print(json.dumps(dict(main=result,resources=resource,files=files,native_commit=ready['source_commit'],bootstrap_commit=COMMIT)))
'''


def main():
    p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--outer',required=True,type=int);p.add_argument('--unit',required=True);a=p.parse_args()
    if not a.name.startswith('search-g1-prefix-selection200-') or '/'in a.name or '..'in a.name or not a.unit.startswith('sst-b-diverse-') or '/'in a.unit or '.'in a.unit:
        raise ValueError('Current experiment namespace and unique unit required')
    if subprocess.check_output(['git','status','--porcelain'],cwd=REPO):raise ValueError('Publish code before dispatch')
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip()
    if subprocess.check_output(['git','ls-remote','origin','refs/heads/'+branch],cwd=REPO,text=True).split()[0]!=commit:raise ValueError('Latest code is not pushed')
    header='\n'.join(k+'='+repr(v)for k,v in dict(NAME=a.name,OUTER=a.outer,UNIT=a.unit,COMMIT=commit).items())
    evidence=OUT/('retained18-dispatch-%d-transport.json'%time.time_ns())
    value=json.loads(remote(header+'\n'+SCRIPT,evidence,target='h100-10',established_controller=True))
    (OUT/('CP%d_RETAINED18_DISPATCH_RESULT.json'%a.outer)).write_text(json.dumps(value,indent=2));print(json.dumps(value),flush=True)


if __name__=='__main__':main()
