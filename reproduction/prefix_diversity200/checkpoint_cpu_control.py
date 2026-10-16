"""Publish-first, code-only owned systemd deployment for the actual milestone5 reader."""
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
         'reproduction/prefix_diversity200/checkpoint_cpu_runner.py',
         'reproduction/prefix_diversity200/checkpoint5_cpu_control.json',
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
assert b.parent==pathlib.Path('/mnt/data/shaoshitong') and c['outer']==5
if PHASE=='deploy':
 assert (b/'outers/outer-000005/complete.json').is_file()
 assert not root.exists(),'Existing owned control namespace requires observation, not replay'
 assert not list((b/'preparation').glob('joint-milestone-0005-independent-full-sha-cpu-once-v1*.json')),'Prior audit intent/proof must be reconciled'
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
proof=b/'preparation/joint-milestone-0005-independent-full-sha-cpu-once-v1.json'
if proof.exists():
 v=read(proof);r['proof_sha256']=hashlib.sha256(proof.read_bytes()).hexdigest();r['proof']=v
print(json.dumps(r))
'''.replace('PAYLOAD', repr(payload)).replace('PHASE', repr(phase))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=('deploy', 'observe')); parser.add_argument('--receipts', required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[2]
    commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=source, text=True).strip()
    branch = subprocess.check_output(['git', 'branch', '--show-current'], cwd=source, text=True).strip()
    remote_sha = subprocess.check_output(['git', 'ls-remote', 'origin', 'refs/heads/'+branch], cwd=source, text=True).split()[0]
    assert remote_sha == commit, 'Helper source must be normally pushed before execution'
    files = {}
    for name in FILES:
        raw = subprocess.check_output(['git', 'show', commit+':'+name], cwd=source)
        files[name] = dict(data=base64.b64encode(raw).decode(), sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
    config = json.loads(base64.b64decode(files['reproduction/prefix_diversity200/checkpoint5_cpu_control.json']['data']))
    from reproduction.h1008_qprobe.management_transport import remote
    receipts = Path(args.receipts); receipts.mkdir(parents=True, exist_ok=True)
    path = receipts/('checkpoint5-cpu-'+args.phase+'-'+str(time.time_ns())+'.json')
    result = json.loads(remote(script(args.phase, config, commit, files), path, target=config['target']))
    output = path.with_name(path.stem+'-result.json'); output.write_text(json.dumps(result, indent=2)+'\n')
    small = {k:v for k,v in result.items() if k != 'proof'}
    if 'proof' in result: small['proof_status'] = result['proof']['status']
    print(json.dumps(dict(receipt=str(output), result=small)))


if __name__ == '__main__': main()
