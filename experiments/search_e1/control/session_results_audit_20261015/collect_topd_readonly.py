"""Read-only output audit of both TOP-D settings and delegated CoSearch final."""
from pathlib import Path
import subprocess,json,datetime,sys
D=Path(__file__).resolve().parent
helper=Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5/work/hpc3_private_control.py')
CODE=r'''
from pathlib import Path
import json,time,datetime,hashlib,subprocess
B=Path('/data/user/jxu957/experiments/topd-20261008')
def load(p):
 if not p.exists():return None
 raw=p.read_bytes();return {'path':str(p),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw),'mtime':p.stat().st_mtime,'data':json.loads(raw)}
o={'observed_at':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'epoch':time.time(),'runs':[]}
for name,jobs in [('TOP-D alpha0.1',[707690]),('TOP-D alpha1',[712204,713717])]:
 run={'name':name,'training':[],'eval':[]}
 for job in jobs:
  r=B/f'runtime/logs/topd-hpc3-{job}/evidence'
  p=r/'actual-outer-updates.json';receipt=load(p)
  if receipt:
   run.setdefault('ledger_sources',[]).append({k:v for k,v in receipt.items() if k!='data'})
   d=receipt['data'];rows=d.get('records',[])
   for v in rows:
    step=v['outer_step']
    if job==712204 and step>5:continue
    m=v.get('metrics',{})
    run['training'].append({'iteration':step,'em':m.get('critic/score/mean'),'reward':m.get('critic/rewards/mean'),'finish_ratio':m.get('env/finish_ratio'),'source':str(p)})
  for p in sorted(r.glob('stage512-step*.json')):
   receipt=load(p);d=receipt['data'];step=d.get('actual_completed_outer',d.get('step'))
   if job==712204 and (step is None or step>5):continue
   m=d.get('metrics',{})
   run['eval'].append({'iteration':step,'em':m.get('val/em/all'),'f1':m.get('val/f1/all'),'n':m.get('val/qa_count/all',d.get('rows',d.get('evaluated_rows'))),'source':{k:v for k,v in receipt.items() if k!='data'}})
  run.setdefault('exceptions',[]).append(load(r/'trainer-first-exception.json'))
 final='topd200-final-acceptance.json' if name.endswith('0.1') else 'topd-alpha1-final-acceptance.json'
 run['final_acceptance']=load(B/'outputs/training-preparation'/final)
 run['complete']=load(B/('runtime/logs/topd-final51713-recovery-707690/complete.json' if name.endswith('0.1') else 'runtime/logs/topd-alpha1-final51713-recovery-713717/complete.json'))
 if run['final_acceptance'] and run['complete']:
  a=run['final_acceptance']['data'];d=run['complete']['data'];assert a['final_completion_receipt_sha256']==run['complete']['sha256']
  assert a['final_rows']==d['rows']==51713 and a['EM']==d['metrics']['val/em/all'] and a['F1']==d['metrics']['val/f1/all']
  run['fresh_acceptance_completion_binding_pass']=True
 o['runs'].append(run)
C=Path('/data/user/jxu957/experiments/cosearch-continue300-x203-20261008/runtime/cosearch-final200-four-gpu-20261015')
o['cosearch_final']={'root':str(C),'progress':load(C/'progress.json'),'complete':load(C/'complete.json'),'acceptance':load(C/'final-acceptance.json'),'summary':load(C/'latest-control-summary.json'),'rows_on_disk':len(list((C/'rows').glob('*.json'))) if (C/'rows').exists() else 0}
o['queue']=subprocess.run(['squeue','-h','-j','721542','-o','%i|%T|%M|%N'],universal_newlines=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE).stdout
o['original_fixed512_crosschecks']=[]
for p in sorted((B/'runtime/logs').glob('*712215*')):
 if p.is_dir():
  for q in p.glob('*.json'):
   if q.stat().st_size<100000 and ('complete' in q.name or 'summary' in q.name or 'metric' in q.name):o['original_fixed512_crosschecks'].append(load(q))
print(json.dumps(o,ensure_ascii=False))
'''
if __name__=='__main__':
 p=subprocess.run([sys.executable,str(helper)],input="python3 -B - <<'REMOTE'\n"+CODE+'\nREMOTE\n',capture_output=True,text=True,timeout=100)
 if p.returncode:raise RuntimeError(p.stderr[-1800:])
 d=json.loads(p.stdout);target=D/('topd-readonly-'+datetime.datetime.now().strftime('%H%M%S')+'.json');target.write_text(json.dumps(d,ensure_ascii=False,indent=2));(D/'topd-latest.json').write_text(json.dumps(d,ensure_ascii=False,indent=2))
 print(json.dumps({'file':str(target),'observed_at':d['observed_at'],'runs':[{ 'name':r['name'],'last':r['training'][-1] if r['training'] else None,'fixed512':r['eval'],'final':r['final_acceptance']['data'] if r['final_acceptance'] else None} for r in d['runs']],'cosearch_rows':d['cosearch_final']['rows_on_disk'],'queue':d['queue']},ensure_ascii=False))
