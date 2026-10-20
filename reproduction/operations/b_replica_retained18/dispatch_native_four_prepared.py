"""One submission of a qualified same-four native joint on an allowed H100."""
import argparse,base64,hashlib,json,subprocess,time
from .control import REPO,OUT,remote

SCRIPT=r'''
import pathlib,json,time,socket,subprocess,base64,hashlib
assert socket.gethostname()==TARGET and TARGET in ('h100-5','h100-10','h100-11')
p=pathlib.Path('/run')/UNIT;p.mkdir(exist_ok=False)
for name,data in BUNDLE.items():
 f=p/name;f.write_bytes(base64.b64decode(data));assert hashlib.sha256(f.read_bytes()).hexdigest()==HASHES[name]
c='/mnt/data/sihe-share5/workspace/training_data_node_10/shaoshitong/'+NAME
code=r"""
import pathlib,json,time,subprocess,socket
b=pathlib.Path(BASE);ready=json.loads((b/'preparation/native-recovery-complete.json').read_text())
assert ready['outer']==OUTER and ready['world_size']==4 and not ready['training_started']
assert 181<OUTER<200 and ready['optimizer_reset'] is False
cfg=json.loads((b/'preparation/launch-config.json').read_text())
assert cfg['source_commit']==ready['source_commit'] and cfg['resume_joint']['recovery_mode']=='diverse_same_four_latest_joint_v1'
assert not(b/'logs/process-ledger.json').exists()and not(b/'preparation/shared-activation.json').exists()
assert not(b/'preparation/submit-intent.json').exists()and not(b/'preparation/target-node-dispatch-intent.json').exists()
q=subprocess.check_output(['squeue','-h','-n','sst-b-diverse-cp64-eight','-o','%T'],text=True).strip();assert not q
inventory=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.total','--format=csv,noheader,nounits'],text=True)
# No foreign signal or empty-node admission rule; these four assigned cards are
# isolated from reference B4-7. Check actual free bytes and ports at dispatch.
cards=[[x.strip()for x in line.split(',')]for line in inventory.splitlines()if int(line.split(',')[0])in range(4)]
assert len(cards)==4
for port in [19710,19720,19721,19722,19723]:
 with socket.socket()as sock:sock.bind(('127.0.0.1',port))
common=['unshare','--mount','--propagation','private','bash',WRAPPER,'env','CUDA_DEVICE_ORDER=PCI_BUS_ID','SEARCH_G1_PHYSICAL_WORLD_SIZE=4','SEARCH_G1_B_PREFIX_TOKENS=80','OMP_NUM_THREADS=2','MKL_NUM_THREADS=2','PYTHONDONTWRITEBYTECODE=1','PYTHONPATH='+BASE+'/repo','/dataset/shaoshitong/Search-G1-common8/env/bin/python','-u']
command=['systemd-run','--unit='+UNIT+'.service','--service-type=exec','--property=Restart=no','--property=CPUQuota=3200%','--property=MemoryMax=256G']+common+[ACTIVATION,'--base',BASE]
with(b/'preparation/target-node-dispatch-intent.json').open('x')as f:json.dump(dict(time=time.time(),command=command,inventory=inventory,commit=COMMIT,world_size=4,cards=[0,1,2,3],foreign_processes_stopped=False),f)
r=subprocess.run(command,capture_output=True,text=True,timeout=25);result=dict(time=time.time(),returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,unit=UNIT+'.service')
with(b/'preparation/target-node-dispatch-result.json').open('x')as f:json.dump(result,f)
assert r.returncode==0,result
resources=['systemd-run','--unit='+UNIT+'-resources.service','--service-type=exec','--property=Restart=no','--property=BindsTo='+UNIT+'.service','--property=After='+UNIT+'.service','--property=CPUQuota=100%','--property=MemoryMax=1G']+common+[RESOURCES,'--base',BASE]
r=subprocess.run(resources,capture_output=True,text=True,timeout=25)
print(json.dumps(dict(main=result,resources=dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr),world_size=4,inventory=cards)))
"""
header='\n'.join(k+'='+repr(v)for k,v in dict(BASE=c,UNIT=UNIT,COMMIT=COMMIT,OUTER=OUTER,WRAPPER=str(p/'private_shared_paths.sh'),ACTIVATION=str(p/'activate_prepared_shared.py'),RESOURCES=str(p/'wait_shared_resources.py')).items())
r=subprocess.run(['unshare','--mount','--propagation','private','bash',str(p/'private_shared_paths.sh'),'/usr/bin/python3','-c',header+'\n'+code],capture_output=True,text=True,timeout=160)
if r.returncode:raise RuntimeError('RECONCILE_ONCE_ONLY_SUBMISSION '+r.stdout+r.stderr)
print(r.stdout)
'''

def main():
 p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--unit',required=True);p.add_argument('--outer',required=True,type=int);p.add_argument('--target',required=True,choices=['h100-5','h100-10','h100-11']);a=p.parse_args()
 assert a.name.startswith('search-g1-prefix-selection200-reference-replica-b-diverse10to5-cp%d-four-'%a.outer)and '/'not in a.name and '..'not in a.name
 assert a.unit.startswith('sst-b-diverse-cp%d-four-'%a.outer)and all(x in'abcdefghijklmnopqrstuvwxyz0123456789-'for x in a.unit)
 assert not subprocess.check_output(['git','status','--porcelain'],cwd=REPO)
 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip();branch=subprocess.check_output(['git','branch','--show-current'],cwd=REPO,text=True).strip()
 assert subprocess.check_output(['git','ls-remote','origin','refs/heads/'+branch],cwd=REPO,text=True).split()[0]==commit
 files={name:(REPO/'reproduction/operations/b_replica_retained18'/name).read_bytes()for name in('private_shared_paths.sh','activate_prepared_shared.py','wait_shared_resources.py')}
 assert 181<a.outer<200
 constants=dict(NAME=a.name,UNIT=a.unit,COMMIT=commit,OUTER=a.outer,TARGET=a.target,BUNDLE={k:base64.b64encode(v).decode()for k,v in files.items()},HASHES={k:hashlib.sha256(v).hexdigest()for k,v in files.items()})
 code='\n'.join(k+'='+repr(v)for k,v in constants.items())+'\n'+SCRIPT
 v=json.loads(remote(code,OUT/('four-dispatch-'+str(time.time_ns())+'.json'),target=a.target,established_controller=True))
 (OUT/('CP%d_SAME_FOUR_DISPATCH.json'%a.outer)).write_text(json.dumps(v,indent=2));print(json.dumps(v),flush=True)

if __name__=='__main__':main()
