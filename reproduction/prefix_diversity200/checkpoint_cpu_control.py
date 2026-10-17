"""Publish-first, code-only owned systemd deployment for an actual milestone reader."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import time


FILES = ('reproduction/prefix_diversity200/__init__.py',
         'reproduction/prefix_diversity200/actor.py', 'reproduction/prefix_diversity200/diversity.py',
         'reproduction/prefix_diversity200/checkpoint_cpu_once.py',
         'reproduction/prefix_diversity200/checkpoint_i1i6_cpu.py',
         'reproduction/prefix_diversity200/checkpoint_i1i6_runtime_files.json',
         'reproduction/prefix_diversity200/interaction.py',
         'reproduction/prefix_diversity200/checkpoint_cpu_runner.py',
         'reproduction/prefix20_online/pool.py')


def script(phase, config, commit, files):
    payload = dict(config=config, helper_commit=commit, files=files)
    return '''import base64,hashlib,json,os,pathlib,subprocess,time
p=PAYLOAD;c=p['config'];b=pathlib.Path(c['base']);root=b/c['control_name']
def read(path):return json.loads(path.read_text())
def once(path,value):
 with path.open('x')as f:json.dump(value,f,sort_keys=True);f.flush();os.fsync(f.fileno())
actual=read(b/'preparation/launch-config.json')
assert actual['run_id']==c['run_id'] and actual['source_commit']==c['runtime_source_commit']
assert b.parent==pathlib.Path('/mnt/data/shaoshitong') and c['outer'] in (5,50,100,150,200)
if c.get('runtime_config_sha256'):assert hashlib.sha256((b/'preparation/launch-config.json').read_bytes()).hexdigest()==c['runtime_config_sha256']
proof_name='joint-milestone-%04d-independent-full-sha-cpu-once-v1'%c['outer']
joint_path=b/'outers'/('outer-%06d'%c['outer'])/'complete.json'
candidate=None
if joint_path.is_file():
 j=read(joint_path);a=j['actor'];n=c['outer']
 assert j['run_id']==c['run_id']and j['source_commit']==c['runtime_source_commit']and j['outer']==n
 assert a['completed_outer']==j['probe']['completed_outer']==n and a['actual_adam_steps_per_rank']==j['probe']['actual_adam_steps_per_rank']==[20*n]*8 and a['scheduler_completed_outer']==n
 jobs=read(b/'logs/process-ledger.json')['jobs'];attempt=pathlib.Path(a['state_path']).name.split('-attempt-',1)[1]
 writers={name:jobs.get(name)for name in ('outer-%06d-actor-%s'%(n,attempt),'outer-%06d-probe-update'%n)}
 candidate={'joint_sha256':hashlib.sha256(joint_path.read_bytes()).hexdigest(),'writers':writers,'writers_stopped':all(v and v.get('state')=='STOPPED'and v.get('exit_code')==0 for v in writers.values())}
if PHASE=='check':
 latest=read(b/'latest-joint.json')if(b/'latest-joint.json').exists()else None
 print(json.dumps(dict(status='PENDING_ACTUAL_JOINT_NO_INTENT'if candidate is None else('ACTUAL_JOINT_WRITERS_STOPPED_READY'if candidate['writers_stopped']else'PENDING_WRITERS_NO_INTENT'),candidate=candidate,latest_complete=latest['outer']if latest else 0,config_sha256=hashlib.sha256((b/'preparation/launch-config.json').read_bytes()).hexdigest(),audit_intent_created=False,training_modified=False,observed_unix=time.time())))
 raise SystemExit
if PHASE=='deploy':
 assert candidate and candidate['writers_stopped'],'Actual immutable joint and stopped successful writers required'
 assert not root.exists(),'Existing owned control namespace requires observation, not replay'
 assert not list((b/'preparation').glob(proof_name+'*.json')),'Prior audit intent/proof must be reconciled'
 root.mkdir(mode=0o700)
 for name,row in p['files'].items():
  rel=pathlib.PurePosixPath(name);assert not rel.is_absolute()and'..'not in rel.parts
  raw=base64.b64decode(row['data']);assert hashlib.sha256(raw).hexdigest()==row['sha256']
  dest=root/name;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(raw)
  assert hashlib.sha256(dest.read_bytes()).hexdigest()==row['sha256']
 once(root/'deploy-receipt.json',dict(config=c,helper_commit=p['helper_commit'],files={k:{'sha256':v['sha256'],'bytes':v['bytes']}for k,v in p['files'].items()},created_unix=time.time(),runtime_source_modified=False))
 cmd=['systemd-run','--user','--unit='+c['unit'],'--property=MemoryMax='+c['memory_max'],'--property=CPUQuota='+c['cpu_quota'],'--property=Nice=19','--property=IOWeight=10','--property=RuntimeMaxSec='+str(c['runtime_max_seconds']),'--setenv=CUDA_VISIBLE_DEVICES=','--setenv=OMP_NUM_THREADS=1','--setenv=MKL_NUM_THREADS=1','--setenv=PYTHONPATH='+str(root),'--working-directory='+str(root),c['python'],'-u','-m','reproduction.prefix_diversity200.checkpoint_cpu_runner']
 r=subprocess.run(cmd,capture_output=True,text=True)
 once(root/'systemd-submit.json',dict(command=cmd,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,observed_unix=time.time()))
 assert r.returncode==0,r.stderr
r={'time':time.time(),'control_root':str(root),'helper_commit':p['helper_commit']}
for name in ('reader-start.json','reader-exit.json','systemd-submit.json'):
 if(root/name).is_file():r[name]=read(root/name)
log=root/'reader.log'
if log.exists():
 r['log_bytes']=log.stat().st_size;r['log_tail']=log.read_bytes()[-6000:].decode(errors='replace')
fields=['ActiveState','SubState','Result','ExecMainPID','ExecMainStatus','MemoryCurrent','MemoryPeak','CPUUsageNSec','IOReadBytes','MemoryMax','CPUQuotaPerSecUSec']
r['systemd']=subprocess.run(['systemctl','--user','show',c['unit']]+['--property='+x for x in fields],capture_output=True,text=True).stdout
proof=b/'preparation'/(proof_name+'.json')
if proof.exists():
 v=read(proof);r['proof_sha256']=hashlib.sha256(proof.read_bytes()).hexdigest();r['proof']=v
print(json.dumps(r))
'''.replace('PAYLOAD', repr(payload)).replace('PHASE', repr(phase))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('check','deploy', 'observe')); parser.add_argument('--receipts', required=True)
    parser.add_argument('--config', default='reproduction/prefix_diversity200/checkpoint5_cpu_control.json')
    parser.add_argument('--master-receipt',required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[2]
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip()
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=source, text=True).strip()
    remote_sha = subprocess.check_output(['git', 'ls-remote', 'origin', 'refs/heads/'+branch], cwd=source, text=True).split()[0]
    assert remote_sha == commit, 'Helper source must be normally pushed before execution'
    files = {}
    for name in (*FILES, args.config):
        raw = subprocess.check_output(['git', 'show', commit+':'+name], cwd=source)
        files[name] = dict(data=base64.b64encode(raw).decode(), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
    config = json.loads(base64.b64decode(files[args.config]['data']))
    from reproduction.h1008_qprobe.management_transport import remote
    receipts = Path(args.receipts); receipts.mkdir(parents=True, exist_ok=True)
    path = receipts/('checkpoint%d-cpu-'%config['outer']+args.phase+'-'+str(time.time_ns())+'.json')
    master=json.loads(Path(args.master_receipt).read_text())
    assert master['status']=='AUTHENTICATED_MASTER_OCHECK'
    result = json.loads(remote(script(args.phase, config, commit, files), path, control_path=master['control_path'], target=config['target']))
    output = path.with_name(path.stem+'-result.json'); output.write_text(json.dumps(result, indent=2)+'\n')
    small = {k:v for k,v in result.items() if k != 'proof'}
    if 'proof' in result: small['proof_status'] = result['proof']['status']
    print(json.dumps(dict(receipt=str(output), result=small)))


if __name__ == '__main__': main()
