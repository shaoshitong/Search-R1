"""Once stage passive scalar telemetry outside the frozen training checkout."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
from reproduction.h1008_qprobe.management_transport import remote

REMOTE = r'''
import pathlib,json,hashlib,subprocess,os,time
p=PAYLOAD;b=pathlib.Path(p['base']);segment=p.get('segment','resource5s-v1');assert segment in ('resource5s-v1','resource5s-v2')
d=b/'preparation'/segment;unit='g1-resource5s-'+p['run_id'][:12]+'-'+p['job']+('-v2' if segment=='resource5s-v2' else '')
def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def once(path,value):
 with path.open('x') as f:json.dump(value,f,sort_keys=True);f.flush();os.fsync(f.fileno())
def unit_state():
 v=subprocess.run(['systemctl','show',unit,'--property=LoadState,ActiveState,SubState,Result,MainPID,ExecMainStatus,ControlGroup'],capture_output=True,text=True,timeout=15)
 return dict(returncode=v.returncode,stdout=v.stdout,stderr=v.stderr)
assert b.parent==pathlib.Path('/mnt/data/shaoshitong')and b.name.startswith('search-g1-prefix-diversity200-')and b.resolve()==b
c=read(b/'preparation/launch-config.json');assert c['run_id']==p['run_id']and read(b/'preparation/submit-receipt.json')['job_id']==p['job']
if p['phase']=='deploy' and not d.exists():
 if segment=='resource5s-v2':
  previous=read(b/'preparation/resource5s-v1/intent.json');assert previous['run_id']==p['run_id']and previous['job']==p['job']
  assert previous['base']==str(b)and previous['unit']=='g1-resource5s-'+p['run_id'][:12]+'-'+p['job'],'Previous sampler identity differs'
  assert sha(b/'preparation/resource5s-v1/reader.py')==previous['reader_sha256'],'Previous sampler evidence changed'
  prior=subprocess.run(['systemctl','show',previous['unit'],'--property=ActiveState,MainPID'],capture_output=True,text=True,timeout=15)
  props=dict(line.split('=',1)for line in prior.stdout.splitlines()if'='in line)
  assert prior.returncode==0 and props.get('ActiveState')in('inactive','failed')and props.get('MainPID')=='0','Prior sampler must be actually terminal'
 native=subprocess.run(['scontrol','show','job',p['job'],'-o'],capture_output=True,text=True,timeout=15)
 assert native.returncode==0 and 'JobState=RUNNING' in native.stdout and 'WorkDir='+str(b) in native.stdout
 assert 'LoadState=not-found' in unit_state()['stdout'],'Reconcile existing unit, never repeat submission'
 d.mkdir(mode=0o700)
 once(d/'intent.json',dict(unit=unit,job=p['job'],base=str(b),run_id=p['run_id'],source_commit=p['commit'],reader_sha256=p['sha256'],created_unix=time.time(),training_modified=False))
 reader=d/'reader.py';reader.write_text(p['source']);reader.chmod(0o600);assert sha(reader)==p['sha256']
 command=['systemd-run','--unit='+unit,'--property=CPUQuota=10%','--property=MemoryMax='+('1G' if segment=='resource5s-v2' else '128M'),'--property=Nice=19','--property=TasksMax=32','--property=RuntimeMaxSec=172900','--property=StandardOutput=append:'+str(d/'unit.log'),'--property=StandardError=append:'+str(d/'unit.log'),'--setenv=CUDA_VISIBLE_DEVICES=','/usr/bin/python3',str(reader),'--base',str(b),'--job',p['job'],'--output',str(d)]
 try:
  v=subprocess.run(command,capture_output=True,text=True,timeout=30)
  receipt=dict(command=command,returncode=v.returncode,stdout=v.stdout,stderr=v.stderr,status='SUBMITTED' if v.returncode==0 else 'REJECTED_NO_REPLAY')
 except subprocess.TimeoutExpired as error:receipt=dict(command=command,status='UNKNOWN_RECONCILE_UNIT_BEFORE_ACTION',error=repr(error))
 once(d/'submit-receipt.json',receipt)
assert d.exists(),'No existing task telemetry directory'
intent=read(d/'intent.json');assert intent['run_id']==p['run_id']and intent['job']==p['job']and sha(d/'reader.py')==intent['reader_sha256']
summary=read(d/'summary.json')if(d/'summary.json').exists()else None
print(json.dumps(dict(unit=unit,directory=str(d),intent=intent,receipt=read(d/'submit-receipt.json')if(d/'submit-receipt.json').exists()else None,unit_state=unit_state(),summary=summary,log_tail=(d/'unit.log').read_text(errors='replace')[-2000:]if(d/'unit.log').exists()else None,stream_bytes=(d/'samples.jsonl').stat().st_size if(d/'samples.jsonl').exists()else 0,observed_unix=time.time(),training_modified=False),allow_nan=False))
'''


def main():
 parser=argparse.ArgumentParser();parser.add_argument('phase',choices=('deploy','query'));parser.add_argument('--config',required=True);parser.add_argument('--receipts',required=True);parser.add_argument('--segment',choices=('resource5s-v1','resource5s-v2'),default='resource5s-v1');args=parser.parse_args()
 c=json.loads(Path(args.config).read_text());repo=Path(__file__).resolve().parents[2];commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo,text=True).strip()
 if args.phase=='deploy':
  branch=subprocess.check_output(['git','branch','--show-current'],cwd=repo,text=True).strip()
  assert subprocess.check_output(['git','ls-remote','origin','refs/heads/'+branch],cwd=repo,text=True).split()[0]==commit
 source=subprocess.check_output(['git','show',commit+':reproduction/prefix_diversity200/resource_sample.py'],cwd=repo).decode()
 w=Path(args.receipts);job=json.loads((w/'LATEST_ACTUAL.json').read_text())['native']['JobId']
 payload=dict(phase=args.phase,segment=args.segment,base=c['base'],run_id=c['run_id'],job=str(job),commit=commit,source=source,sha256=hashlib.sha256(source.encode()).hexdigest())
 path=w/('resource5s-'+args.phase+'-'+str(time.time_ns())+'.json');value=json.loads(remote(REMOTE.replace('PAYLOAD',repr(payload)),path,target=c['node'],established_controller=True))
 result=path.with_name(path.stem+'-result.json');result.write_text(json.dumps(value,indent=2)+'\n');(w/'RESOURCE5S_LATEST.json').write_text(result.read_text())
 print(json.dumps(dict(evidence=str(result),result=value),allow_nan=False))


if __name__=='__main__':main()
