"""Read only our exact IP14 native birth through the verified same-cluster LAN.

Alternative observation when retained18's management SSH stalls. This neither
launches a worker nor changes SSH, mounts, credentials or the live runtime.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import time
from .control import OUT

SCRIPT = r'''
import json,pathlib,time,hashlib
proc=pathlib.Path('/proc')/str(PID)
stat=(proc/'stat').read_text() if proc.exists() else None
same=bool(stat and int(stat.rsplit(')',1)[1].split()[19])==START and stat.rsplit(')',1)[1].split()[0]!='Z')
result=dict(observed_unix=time.time(),pid=PID,start_ticks=START,same_birth=same,training_modified=False)
if same:
 cmd=(proc/'cmdline').read_bytes().split(b'\0')
 assert BASE.encode() in cmd,'Native birth does not own the exact requested base'
 root=proc/'root';base=root/BASE.lstrip('/')
 def read(path):
  try:return json.loads(path.read_text())
  except FileNotFoundError:return None
 result['stage']=read(base/'logs/pipeline-status.json')
 result['cleanup']=read(base/'logs/cleanup-status.json')
 outer=(result['stage'] or {}).get('outer',0)
 result['progress']={str(p.relative_to(base)):read(p) for p in (base/'outers'/('outer-%06d'%outer)).rglob('*progress*.json') if p.stat().st_size<15000}
 result['actor']=[]
 for p in sorted((base/'actor/outers').glob('outer-%04d-attempt-*/rank-*-updates.jsonl'%outer)):
  restore=None;writes=0;opportunities=0;last=None;skips=0
  with p.open('rb') as stream:
   for line in stream:
    if not line.endswith(b'\n'):continue
    row=json.loads(line);event=row.get('event')
    if event=='ACTUAL_RESTORE':restore={k:row.get(k) for k in ('rank','restored_completed_outer','actual_adam_before','restored_scheduler','actual_model_master_dtypes','actual_learning_rates','actual_global_micro')}
    elif event=='ACTUAL_ADAM_WRITE':writes+=1;last=row.get('actual_adam_steps')
    elif event=='OPPORTUNITY':opportunities+=1
    elif 'SKIP' in str(event):skips+=1
  result['actor'].append(dict(path=str(p.relative_to(base)),restore=restore,writes=writes,opportunities=opportunities,last_adam=last,skip_events=skips))
 if AUDIT:
  # All large metadata is parsed here on H100. Only compact facts leave it.
  latest=base/'latest-joint.json';raw=latest.read_bytes();joint=json.loads(raw)
  committed=joint['outer'];complete=base/'outers'/('outer-%06d'%committed)/'complete.json'
  sha=hashlib.sha256(raw).hexdigest()
  assert hashlib.sha256(complete.read_bytes()).hexdigest()==sha,'Latest atomic joint differs from sealed complete'
  ar=[]
  for receipt in joint['actor']['rank_receipts']:
   p=root/receipt['ledger_path'].lstrip('/');before=p.stat()
   ledger=p.read_bytes();after=p.stat()
   assert (before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
   assert hashlib.sha256(ledger).hexdigest()==receipt['ledger_sha256']
   rows=[json.loads(line) for line in ledger.splitlines()]
   restore=next(row for row in rows if row['event']=='ACTUAL_RESTORE')
   writes=[row for row in rows if row['event']=='ACTUAL_ADAM_WRITE']
   ar.append(dict(rank=receipt['rank'],restored_outer=restore['restored_completed_outer'],adam_before=restore['actual_adam_before'],scheduler_before=restore['restored_scheduler'],opportunities=sum(row['event']=='OPPORTUNITY' for row in rows),writes=len(writes),last_adam=writes[-1]['actual_adam_steps'] if writes else None,ledger_sha_matches=True))
  pr=[]
  for p in sorted((base/'outers'/('outer-%06d'%committed)/'probe-update').glob('rank-*-updates.jsonl')):
   rows=[json.loads(line) for line in p.read_bytes().splitlines()]
   rates={row.get('actual_learning_rate',row.get('learning_rate')) for row in rows}
   pr.append(dict(rank=rows[0]['rank'],writes=len(rows),first_adam=rows[0]['actual_adam_steps'],last_adam=rows[-1]['actual_adam_steps'],learning_rates=sorted(rate for rate in rates if rate is not None),learning_rate_record_missing=None in rates))
  probe_meta={k:v for k,v in joint['probe'].items() if isinstance(v,(str,int,float,bool,type(None))) or (isinstance(v,list) and len(v)<=16 and all(isinstance(x,(str,int,float,bool,type(None))) for x in v))}
  result['joint']=dict(outer=committed,sha256=sha,actor=joint['actor']['actual_adam_steps_per_rank'],scheduler=joint['actor']['scheduler_completed_outer'],probe=joint['probe']['actual_adam_steps_per_rank'],probe_meta=probe_meta,probe_ranks=pr,phase_timings=joint.get('phase_timings'),selection_metrics=joint.get('selection_metrics'),ranks=ar)
print(json.dumps(result))
'''


def read(base, pid, start, *, audit_joint=False):
    from reproduction.h1008_qprobe.established_controller_transport import _controller
    ctl = _controller()
    c = ctl.H100_ACCESS
    if not base.startswith('/mnt/data/sihe-share5/workspace/training_data_node_10/shaoshitong/search-g1-prefix-selection200-reference-replica-b-diverse10to5-'):
        raise ValueError('Require this experiment exact canonical base')
    assert c['targets']['h100-11'] == '10.33.4.14'
    assert c['targets']['h100-5'] == '10.33.0.5'
    keys = c['trusted_target_host_keys']['h100-11']
    prefix = ['env','-u','LD_LIBRARY_PATH','ssh','-T','-i',c['management_identity_file'],
              '-o','IdentitiesOnly=yes','-o','BatchMode=yes','-o','StrictHostKeyChecking=yes',
              '-o','UpdateHostKeys=no','-o','ConnectTimeout=10','-o','ConnectionAttempts=1']
    gateway = shlex.join(prefix+['-W','%h:%p',c['user']+'@10.33.0.5'])
    worker = prefix+['-o','HostKeyAlias=10.33.4.14','-o','ProxyCommand='+gateway,
                     '-o','KnownHostsCommand='+shlex.join(['/bin/echo','\n'.join(keys)]),
                     c['user']+'@10.32.0.11','/usr/bin/python3 -']
    script = 'BASE='+repr(base)+'\nPID='+repr(int(pid))+'\nSTART='+repr(int(start))+'\nAUDIT='+repr(bool(audit_joint))+'\n'+SCRIPT
    path = OUT/('ip14-native-read-'+str(time.time_ns())+'.json')
    record = dict(route='mgmt-host -> known5 SSH-W -> IP14 verified LAN alias',
                  base=base,pid=pid,start_ticks=start,script_sha256=hashlib.sha256(script.encode()).hexdigest(),
                  training_modified=False,global_configuration_modified=False)
    # Reserve an evidence path; observation is never a training submission.
    with path.open('x') as evidence:
        try:
            r = subprocess.run(ctl.ssh('mgmt-host')+[shlex.join(worker)],input=script,
                               text=True,capture_output=True,timeout=60)
            record.update(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
        except subprocess.TimeoutExpired:
            record.update(returncode=None,error='READ_ONLY_TIMEOUT_CURRENT_NATIVE_UNKNOWN')
        json.dump(record,evidence,indent=2)
    if record['returncode'] != 0:
        raise RuntimeError(str(path)+' '+record.get('stderr',record.get('error','')))
    result=json.loads(record['stdout']);result['evidence']=str(path)
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--base',required=True);parser.add_argument('--pid',type=int,required=True)
    parser.add_argument('--start',type=int,required=True)
    parser.add_argument('--audit-joint',action='store_true')
    a=parser.parse_args();print(json.dumps(read(a.base,a.pid,a.start,audit_joint=a.audit_joint),ensure_ascii=False))
