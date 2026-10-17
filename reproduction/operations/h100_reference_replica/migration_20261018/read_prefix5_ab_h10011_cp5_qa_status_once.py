# Archived actual control; reconcile live handles/once proofs before use.
import json,subprocess,importlib.util,datetime
from pathlib import Path
spec=importlib.util.spec_from_file_location('ctl','/Users/shaoshitong/Documents/remote_codex_control/control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl)
a=json.loads(Path('work/prefix5-ab-assigned-arms.json').read_text())
script=r'''import json,hashlib,time
from pathlib import Path
ARMS=__ARMS__
r={}
for s,a in ARMS.items():
 b=Path(a['base'].replace('/mnt/data/sihe-share5','/mnt/shared'));prep=b/'preparation';v={}
 for name in ['ab-cp5-cpu-control-receipt.json','ab-joint-0005-independent-full-sha-cpu.json']:
  p=prep/name
  if p.exists():
   data=p.read_bytes();d=json.loads(data)
   if name.startswith('ab-joint'):
    v['cpu_proof']={'path':str(p),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'result':d}
   else:
    pid=d['pid'];q=Path('/proc/'+str(pid));same=False
    if q.exists():
     try:same=int((q/'stat').read_text().rsplit(')',1)[1].split()[19])==d['start_ticks']
     except FileNotFoundError:pass
    v['cpu_control']=dict(d,same_birth_now=same)
 log=prep/'ab-cp5-cpu-control.log'
 if log.exists():v['cpu_log_tail']=log.read_text()[-3500:]
 qa=b/'outers/outer-000005/validation';q=qa/'complete.json'
 if q.exists():
  data=q.read_bytes();d=json.loads(data);v['qa']={'path':str(q),'sha256':hashlib.sha256(data).hexdigest(),'result':d}
 else:
  v['qa_worker_complete']=[]
  for p in sorted(qa.glob('rank-*-complete.json')):
   d=json.loads(p.read_text());v['qa_worker_complete'].append({k:d.get(k)for k in ['rank','qa_count','actor_version','actual_compute_dtype','elapsed_seconds','initialization_oom_attempts']})
  v['qa_worker_logs']=[]
  for p in sorted(qa.glob('rank-*.log')):
   with p.open('rb')as f:f.seek(max(0,p.stat().st_size-2000));tail=f.read().decode(errors='replace')
   v['qa_worker_logs'].append({'path':str(p),'bytes':p.stat().st_size,'tail':tail})
 p=b/'logs/qa-pending.json'
 if p.exists():v['qa_pending']=json.loads(p.read_text())
 r[s]=v
print(json.dumps({'utc_unix':time.time(),'arms':r}))
'''.replace('__ARMS__',repr(a))
p=subprocess.run(ctl.route('h100-11','/usr/bin/python3 -'),input=script,text=True,capture_output=True,timeout=65);assert p.returncode==0,p.stderr;d=json.loads(p.stdout);tag=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');dest=Path('work')/('prefix5-ab-cp5-qa-status-'+tag+'.json');dest.write_text(json.dumps(d,indent=2)+'\n');Path('work/prefix5-ab-cp5-qa-status-latest.json').write_text(json.dumps(d,indent=2)+'\n')
print('actual',str(dest))
for s,v in d['arms'].items():
 print(s,'CPU',v.get('cpu_proof',{}).get('result',{}).get('status'),v.get('cpu_control'),'log',v.get('cpu_log_tail'),'QA',v.get('qa',{}).get('result'),'QA_pending',v.get('qa_pending'),'workers',v.get('qa_worker_complete'))
 for log in v.get('qa_worker_logs',[]):print(s,log)
