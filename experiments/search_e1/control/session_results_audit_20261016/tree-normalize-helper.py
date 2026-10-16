import json,hashlib,datetime,re,statistics
from pathlib import Path
D=Path(__file__).resolve().parent
def mk(name,session,segments,at,target=200,note=''):
 train={};fixed={};full=None;proof=[]
 for e,lo,hi in segments:
  for l in e['logs']:
   proof.append({'path':l['path'],'bytes':l['bytes'],'mtime':l['mtime']})
   for r in l['rows']:
    i=r['step'];m=r['metrics']
    if lo<=i<=hi and 'critic/score/mean' in m:train[i]={'iter':i,'em':m['critic/score/mean'],'path':l['path'],'line':r['line']}
    n=m.get('val/qa_count/all')
    if n==512 and lo<=i<=hi:fixed[i]={'iter':i,'em':m.get('val/em/all'),'f1':m.get('val/f1/all'),'n':512,'path':l['path'],'line':r['line']}
    if n==51713:full={'complete':True,'em':m.get('val/em/all'),'f1':m.get('val/f1/all'),'n':51713,'path':l['path'],'line':r['line'],'note':'Final evaluation is logged as step201; weights are final completed outer200.'}
  for r in e['receipts']:
   d=r['data'];m=d.get('metrics',{});n=m.get('val/qa_count/all');i=d.get('actual_completed_outer',d.get('step'))
   if n==512 and i is not None and lo<=i<=hi:
    fixed[i]={'iter':i,'em':m.get('val/em/all'),'f1':m.get('val/f1/all'),'n':512,'path':r['path'],'sha256':r['sha256'],'mtime':r['mtime']}
   if n==51713:full={'complete':True,'em':m.get('val/em/all'),'f1':m.get('val/f1/all'),'n':51713,'path':r['path'],'sha256':r['sha256'],'mtime':r['mtime'],'status':d.get('status'),'evaluated_rows':d.get('evaluated_rows',d.get('evaluated_count',d.get('count')))}
 if full:full['correct']=round(full['em']*51713)
 last=max(train) if train else None;last10=[train[i]['em'] for i in sorted(train)[-10:]]
 return {'name':name,'session':session,'output_dir':segments[-1][0]['root'],'completed_iter':last,'target_iter':target,'train_latest_em':train[last]['em'] if last else None,'train_recent10_mean':sum(last10)/len(last10) if last10 else None,'train_recent10_n':len(last10),'train_series':[train[i] for i in sorted(train)],'fixed512':[fixed.get(i,{'iter':i,'em':None,'f1':None,'n':None,'path':None,'reason':'No completed scheduled fixed512 result for this checkpoint yet.'}) for i in [5,50,100,150,200]],'additional_fixed512':[fixed[i] for i in sorted(fixed) if i not in [5,50,100,150,200]],'final51713':full or {'complete':False,'em':None,'f1':None,'n':0,'path':None,'reason':'This training setting has not reached its final outer200; no full final51713 score.'},'status':'training_complete' if last==target else 'training_running','note':note,'reader_started_epoch':at,'log_evidence':proof}
