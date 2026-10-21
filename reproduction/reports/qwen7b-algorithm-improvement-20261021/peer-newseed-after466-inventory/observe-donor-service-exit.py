"""Read-only exact donor service exit/progress observation; no producer writes."""
import importlib.util
import json
from pathlib import Path
import shlex
import subprocess

HERE = Path(__file__).resolve().parent
WORKER = r'''import pathlib,json,hashlib,time,subprocess,os,re
B=pathlib.Path('/mnt/shared/workspace/training_data_node_10/shaoshitong/qwen7b-resample-seed20261021-g5-20261020')
units=['sst-q7b-g5-relaygen-queued-owner-v2-downstream-r1-70188b8e50fd.service','sst-q7b-g5-relaygen-queued-supp-v1-slot-gpu4-tp1-r1-70188b8e50fd.service']
properties=['Id','LoadState','ActiveState','SubState','MainPID','Result','ExecMainCode','ExecMainStatus','ExecMainStartTimestamp','ExecMainExitTimestamp','ActiveEnterTimestamp','InactiveEnterTimestamp','StateChangeTimestamp','FragmentPath','StandardOutput','StandardError','NRestarts','ControlGroup']
rows=[]
for unit in units:
 args=['systemctl','show',unit]
 for key in properties:args.extend(['-p',key])
 p=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)
 x=dict(line.split('=',1)for line in p.stdout.splitlines()if'='in line)
 x['query_exit_code']=p.returncode;pid=int(x.get('MainPID','0'));proc=pathlib.Path('/proc')/str(pid)
 x['pid_exists']=pid>0 and(proc/'stat').exists()
 if x['pid_exists']:
  stat=(proc/'stat').read_text().rsplit(')',1)[1].split();x['actual_pid_birth_ticks']=int(stat[19]);x['actual_pid_state']=stat[0]
  cmd=(proc/'cmdline').read_bytes().replace(b'\x00',b' ').decode(errors='replace')
  x['source_code_or_receipt_arguments']=sorted(set(p for p in re.findall(r'/mnt/(?:data|shared)/[^\s\x00\"\x27;]+',cmd) if not any(s in p.lower() for s in ['secret','credential','token'])))
 # Journal text is interpreted remotely, only bounded diagnostic categories return.
 p=subprocess.run(['journalctl','-u',unit,'-n','45','-o','json','--no-pager'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)
 events=[]
 for line in p.stdout.splitlines():
  try:v=json.loads(line)
  except ValueError:continue
  message=v.get('MESSAGE','');message=message if isinstance(message,str)else''
  terms=[s for s in ['Traceback','OutOfMemoryError','CUDA out of memory','Permission denied','No space left on device','ConnectionError','TimeoutError','RuntimeError','ValueError','Killed','failed','Failed','Finished','finished','completed','Completed','success','exit-code']if s in message]
  if terms:events.append(dict(realtime_microseconds=v.get('__REALTIME_TIMESTAMP'),pid=v.get('_PID'),priority=v.get('PRIORITY'),diagnostic_terms=terms))
 x['safe_journal_diagnostic_events']=events[-12:];rows.append(x)
p=subprocess.run(['systemctl','list-units','--type=service','--all','--no-legend','--plain'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)
related=[dict(unit=line.split()[0],load=line.split()[1],active=line.split()[2],sub=line.split()[3])for line in p.stdout.splitlines()if'70188b8e50fd'in line and'sst-q7b'in line]
seals=sorted((B/'sampling').glob('shard-*/batches/batch*/complete.json'));q=t=0;latest=None
for path in seals:
 raw=path.read_bytes();v=json.loads(raw)
 assert v['run_id']=='70188b8e50fd4b45bc420ce58014f46e'and v['model_revision']=='a09a35458c702b33eeacc393d103063234e8bc28'
 q+=v['questions'];t+=v['trajectories'];row=dict(path=str(path.relative_to(B)),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),mtime=path.stat().st_mtime)
 if latest is None or row['mtime']>latest['mtime']:latest=row
top=[dict(name=p.name,directory=p.is_dir())for p in sorted(B.iterdir())]
small=[]
for folder in [B]+[p for p in B.iterdir()if p.is_dir()and p.name not in ['inputs','sampling']]:
 for path in folder.glob('*.json'):
  st=path.stat()
  if st.st_size>128*1024:continue
  raw=path.read_bytes()
  try:v=json.loads(raw)
  except ValueError:continue
  keys=['status','stage','phase','epoch','run_id','questions','trajectories','completed_batches','batches','seed','attempt','actor_optimizer_updates','error_type','type']
  safe={k:v[k]for k in keys if k in v and isinstance(v[k],(str,int,float,bool,type(None)))}
  small.append(dict(path=str(path),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),mtime=st.st_mtime,safe_fields=safe,root_keys=sorted(v)if isinstance(v,dict)else[]))
print(json.dumps(dict(status='READONLY_EXACT_DONOR_SERVICES_AND_SEALED_PROGRESS',epoch=time.time(),hostname=os.uname().nodename,source_base=str(B),services=rows,related_services=related,sealed_batches=len(seals),sealed_questions=q,sealed_new5_trajectories=t,latest_seal=latest,source_all169028_complete=q==169028,top_level_metadata_inventory=top,small_root_receipts=small,frozen_cohort_questions=124194,frozen_cohort_expanded=False,producer_modified=False,payload_read=False,GPU_operation=False),sort_keys=True))
'''

def main():
    spec = importlib.util.spec_from_file_location('ctl', '/Users/shaoshitong/Documents/remote_codex_control/control.py')
    ctl = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ctl)
    result = subprocess.run(ctl.route('h100-10', 'python3 -c '+shlex.quote(WORKER)),
                            stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=70)
    if result.returncode:
        raise RuntimeError('Read-only donor query failed; no producer action: '+result.stderr.decode(errors='replace')[-700:])
    value = json.loads(result.stdout.decode().splitlines()[-1])
    (HERE/'actual-donor-service-exit.json').write_text(json.dumps(value, indent=2)+'\n')
    print(json.dumps(value, sort_keys=True))

if __name__ == '__main__':
    main()
