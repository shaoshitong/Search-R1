"""Stop only the explicitly identified run; preserve small terminal evidence."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from reproduction.h1008_qprobe.management_transport import remote


SCRIPT = r'''
import hashlib,json,pathlib,subprocess,time
p=PAYLOAD;b=pathlib.Path(p['base']);job=p['job']
def read(path):return json.loads(path.read_text())
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def call(args):
 r=subprocess.run(args,capture_output=True,text=True,timeout=30)
 return dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
assert b.resolve()==b and b.parent==pathlib.Path('/mnt/data/shaoshitong')
assert b.name=='search-g1-prefix-diversity200-20261016-f3ee1435e501-resume4'
c=read(b/'preparation/launch-config.json')
assert c['run_id']==p['run_id'] and sha(b/'preparation/launch-config.json')==p['config_sha256']
assert read(b/'preparation/submit-receipt.json')['job_id']==job
native=call(['scontrol','show','job',job,'-o'])
assert native['returncode']==0,native
f=dict(x.split('=',1)for x in native['stdout'].split()if'='in x)
assert f['JobId']==job and f['WorkDir']==str(b) and f['NodeList']=='h100-10'
assert f['Comment']=='g1-prefix200:'+p['run_id']
intent=b/'preparation/human-stop-i1i6-20261017.json'
joint=read(b/'latest-joint.json')
snapshot=dict(observed_unix=time.time(),native=native,fields=f,
 stage=read(b/'logs/pipeline-status.json'),joint_outer=joint['outer'],
 actor_adam=joint['actor']['actual_adam_steps_per_rank'],
 probe_adam=joint['probe']['actual_adam_steps_per_rank'],
 latest_joint_sha256=sha(b/'latest-joint.json'),
 gpu=call(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_memory','--format=csv,noheader,nounits']),
 native_log_tail=(b/'logs'/('native-'+job+'.log')).read_text(errors='replace')[-4000:])
if p['phase']=='stop' and not intent.exists():
 before=dict(human_authorization=p['authorization'],run_id=p['run_id'],job=job,base=str(b),before=snapshot,
  scope='Only current3427. RepairI1–I6 then fresh official actor200; no warmup. No other task cancellation.')
 with intent.open('x')as out:json.dump(before,out,sort_keys=True)
 if f['JobState'] in ('RUNNING','PENDING','CONFIGURING','SUSPENDED'):
  cancel=call(['scancel',job])
 else:cancel=dict(already_terminal_or_completing=f['JobState'])
 with (b/'preparation/human-stop-i1i6-cancel-result.json').open('x')as out:json.dump(cancel,out)
 sampler=b/'preparation/resource5s-v1'
 if(sampler/'intent.json').exists():
  spec=read(sampler/'intent.json');assert spec['run_id']==p['run_id'] and spec['job']==job
  expected='g1-resource5s-'+p['run_id'][:12]+'-'+job;assert spec['unit']==expected
  sampler_stop=call(['systemctl','stop',expected])
  with(b/'preparation/human-stop-i1i6-sampler-result.json').open('x')as out:json.dump(sampler_stop,out)
assert intent.exists(),'Stop must precede observation'
assert read(intent)['human_authorization']['message_id']==p['authorization']['message_id']
ledger=read(b/'logs/process-ledger.json');present=[]
for entry in ledger['jobs'].values():
 for row in entry['ledger'].values():
  proc=pathlib.Path('/proc')/str(row['pid'])/'stat'
  try:
   fields=proc.read_text().split(') ',1)[1].split()
   saved=row.get('start_ticks',row.get('starttime',row.get('start_tick')))
   present.append(dict(pid=row['pid'],state=fields[0],actual_start_ticks=int(fields[19]),saved_birth=row,
    birth_matches=None if saved is None else int(saved)==int(fields[19])))
  except(FileNotFoundError,ProcessLookupError):pass
result=dict(run_id=p['run_id'],job=job,base=str(b),phase=p['phase'],snapshot=snapshot,
 stop_intent=read(intent),cancel_result=read(b/'preparation/human-stop-i1i6-cancel-result.json'),
 native_after=call(['scontrol','show','job',job,'-o']),
 queue=call(['squeue','-h','-j',job,'-o','%i|%T|%j|%N']),
 owned_birth_candidates_still_present=present,
 cleanup=read(b/'logs/cleanup-status.json')if(b/'logs/cleanup-status.json').exists()else None,
 sampler=read(b/'preparation/resource5s-v1/summary.json')if(b/'preparation/resource5s-v1/summary.json').exists()else None,
 observed_unix=time.time(),foreign_processes_killed=False,models_read=False)
print(json.dumps(result,allow_nan=False))
'''


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('phase',choices=('stop','observe'))
    parser.add_argument('--config',required=True)
    parser.add_argument('--authorization',required=True)
    parser.add_argument('--receipts',required=True)
    parser.add_argument('--control-path',required=True)
    args=parser.parse_args()
    config_path=Path(args.config);c=json.loads(config_path.read_text())
    authorization=json.loads(Path(args.authorization).read_text())
    assert authorization['message_id']=='01a14849-d31c-7a80-b223-6a46bc3a5a08'
    payload=dict(phase=args.phase,job='3427',base=c['base'],run_id=c['run_id'],
        config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),authorization=authorization)
    receipts=Path(args.receipts);receipts.mkdir(parents=True,exist_ok=True)
    path=receipts/(args.phase+'-'+str(time.time_ns())+'-transport.json')
    result=json.loads(remote(SCRIPT.replace('PAYLOAD',repr(payload)),path,
        control_path=args.control_path,target='h100-10'))
    out=path.with_name(path.name.replace('-transport','-result'))
    out.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(evidence=str(out),job=result['job'],before_outer=result['snapshot']['joint_outer'],
        native_after=result['native_after'],queue=result['queue'],
        owned_birth_candidates_still_present=result['owned_birth_candidates_still_present'],cleanup=result['cleanup'])))


if __name__=='__main__':main()
