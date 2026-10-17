"""Published code-only check/deploy/observe control for future actual150/200 CPU audits."""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import subprocess
import time

FILES=('reproduction/prefix_diversity200/__init__.py','reproduction/prefix_diversity200/actor.py',
       'reproduction/prefix_diversity200/diversity.py','reproduction/prefix_diversity200/checkpoint_cpu_once.py',
       'reproduction/prefix_diversity200/checkpoint_joint_components_cpu.py','reproduction/prefix20_online/pool.py')


def script(phase,config,commit,files):
    return '''import pathlib,json,hashlib,base64,subprocess,time,os
p=PAYLOAD;c=p['config'];b=pathlib.Path(c['base']);n=c['outer'];d=b/c['control_name'];joint_path=b/'outers'/('outer-%06d'%n)/'complete.json'
def read(path):return json.loads(path.read_text())
def once(path,value):
 with path.open('x')as f:json.dump(value,f,sort_keys=True);f.flush();os.fsync(f.fileno())
assert n in (150,200)and b.parent==pathlib.Path('/mnt/data/shaoshitong')and b.name.startswith('search-g1-prefix-diversity200-')
assert hashlib.sha256((b/'preparation/launch-config.json').read_bytes()).hexdigest()==c['runtime_config_sha256']
actual=read(b/'preparation/launch-config.json');assert actual['run_id']==c['run_id']and actual['source_commit']==c['runtime_source_commit']
r={'observed_unix':time.time(),'base':str(b),'outer':n,'control_root':str(d),'helper_commit':p['commit'],'audit_intent_created':False}
candidate=None
if joint_path.is_file():
 j=read(joint_path);a=j['actor'];assert j['run_id']==c['run_id']and j['source_commit']==c['runtime_source_commit']and j['outer']==n
 assert a['completed_outer']==j['probe']['completed_outer']==n and a['actual_adam_steps_per_rank']==j['probe']['actual_adam_steps_per_rank']==[20*n]*8 and a['scheduler_completed_outer']==n
 jobs=read(b/'logs/process-ledger.json')['jobs'];attempt=pathlib.Path(a['state_path']).name.split('-attempt-',1)[1]
 writers={name:jobs.get(name)for name in ('outer-%06d-actor-%s'%(n,attempt),'outer-%06d-probe-update'%n)}
 stopped=all(v and v.get('state')=='STOPPED'and v.get('exit_code')==0 for v in writers.values())
 candidate={'joint_sha256':hashlib.sha256(joint_path.read_bytes()).hexdigest(),'writers':writers,'writers_stopped':stopped}
r['candidate']=candidate
if PHASE=='check':
 r['status']='PENDING_ACTUAL_JOINT_NO_INTENT'if candidate is None else('SMALL_JOINT_CANDIDATE_REQUIRES_READER_STABILITY_CHECK'if candidate['writers_stopped']else'PENDING_CHECKPOINT_WRITER_NO_INTENT')
 print(json.dumps(r));raise SystemExit
if PHASE=='deploy':
 assert candidate and candidate['writers_stopped'],'Actual joint and stopped writers required; no CPU submission'
 assert not d.exists(),'Existing control directory requires reconciliation; never blind replay'
 assert not list((b/'preparation').glob('joint-milestone-%04d-independent-full-sha-cpu-once-v1*.json'%n)),'Prior strict audit intent/proof prevents duplicate full read'
 d.mkdir(mode=0o700)
 for name,row in p['files'].items():
  rel=pathlib.PurePosixPath(name);assert not rel.is_absolute()and'..'not in rel.parts
  raw=base64.b64decode(row['data']);assert hashlib.sha256(raw).hexdigest()==row['sha256'];target=d/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(raw)
 once(d/'control-config.json',c);once(d/'deploy-receipt.json',dict(helper_commit=p['commit'],config=c,files={k:{'sha256':v['sha256'],'bytes':v['bytes']}for k,v in p['files'].items()},created_unix=time.time(),training_modified=False,runtime_source_modified=False))
 cmd=['systemd-run','--user','--unit='+c['unit'],'--property=MemoryMax=12G','--property=CPUQuota=100%','--property=Nice=19','--property=IOWeight=10','--property=RuntimeMaxSec=3600','--property=StandardOutput=append:'+str(d/'reader.log'),'--property=StandardError=append:'+str(d/'reader.log'),'--setenv=CUDA_VISIBLE_DEVICES=','--setenv=OMP_NUM_THREADS=1','--setenv=MKL_NUM_THREADS=1','--setenv=PYTHONPATH='+str(d),'--working-directory='+str(d),c['python'],'-u','-m','reproduction.prefix_diversity200.checkpoint_joint_components_cpu','--control-config',str(d/'control-config.json')]
 q=subprocess.run(cmd,capture_output=True,text=True);once(d/'systemd-submit.json',dict(command=cmd,returncode=q.returncode,stdout=q.stdout,stderr=q.stderr,observed_unix=time.time()));assert q.returncode==0,q.stderr
for name in ('intent.json','systemd-submit.json'):
 if(d/name).exists():r[name]=read(d/name)
if(d/'reader.log').exists():r['log_bytes']=(d/'reader.log').stat().st_size;r['log_tail']=(d/'reader.log').read_text()[-7000:]
r['systemd']=subprocess.run(['systemctl','--user','show',c['unit']]+['--property='+key for key in ('ActiveState','SubState','Result','ExecMainPID','ExecMainStatus','MemoryCurrent','CPUUsageNSec')],capture_output=True,text=True).stdout
if(d/'proof.json').exists():r['proof']=read(d/'proof.json');r['proof_sha256']=hashlib.sha256((d/'proof.json').read_bytes()).hexdigest()
r['audit_intent_created']=(d/'intent.json').exists();print(json.dumps(r))
'''.replace('PAYLOAD',repr(dict(config=config,commit=commit,files=files))).replace('PHASE',repr(phase))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase',choices=('check','deploy','observe'));parser.add_argument('--config',required=True)
    parser.add_argument('--receipts',required=True);args=parser.parse_args();source=Path(__file__).resolve().parents[2]
    commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()
    branch=subprocess.check_output(['git','branch','--show-current'],cwd=source,text=True).strip()
    remote_sha=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+branch],cwd=source,text=True).split()[0]
    assert remote_sha==commit,'Normal source push is required before using the control'
    files={}
    for name in (*FILES,args.config):
        raw=subprocess.check_output(['git','show',commit+':'+name],cwd=source)
        files[name]=dict(data=base64.b64encode(raw).decode(),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
    config=json.loads(base64.b64decode(files[args.config]['data']));assert config['outer']in(150,200)and config['target']=='h100-10'
    from reproduction.h1008_qprobe.management_transport import remote
    out=Path(args.receipts);out.mkdir(parents=True,exist_ok=True)
    transport=out/('joint%d-components-'%config['outer']+args.phase+'-'+str(time.time_ns())+'.json')
    result=json.loads(remote(script(args.phase,config,commit,files),transport,target=config['target']))
    path=transport.with_name(transport.stem+'-result.json');path.write_text(json.dumps(result,indent=2)+'\n')
    small={k:v for k,v in result.items()if k!='proof'}
    if 'proof'in result:small['proof_status']=result['proof']['status'];small['proof_error']=result['proof'].get('error')
    print(json.dumps(dict(receipt=str(path),result=small)))


if __name__=='__main__':main()
