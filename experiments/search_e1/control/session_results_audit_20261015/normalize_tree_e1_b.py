import json,hashlib,datetime,re
from pathlib import Path
D=Path(__file__).resolve().parent
def read(n):return json.loads((D/n).read_text())
T=read('tree-observation-3.json');TP=read('tree-observation-4.json');A=read('a-c1-observation-3.json');B=read('b-observation-3.json')
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
tree_sid='01a0fcbc-0b46-7502-8b66-d8b21a0e032d';a_sid='01a1028b-b626-7f91-93db-0099ff1623c5';b_sid='01a12e27-764b-7432-a5f2-74d88ee4a8d4'
runs=[]
runs.append(mk('Tree-GRPO original guarded200',tree_sid,[(TP['experiments'][1],1,5),(TP['experiments'][0],6,6),(T['experiments'][-1],7,200)],T['at'],note='Legacy original run3341 is complete; do not join its curve with the new nested or BRANPO experiments. Formal train series refreshed from outer5 onwards; outer1–4 were not re-read here.'))
runs.append(mk('Tree BRANPO fresh200 3360',tree_sid,[(T['experiments'][3],1,200)],T['at'],note='Complete older last-action-backoff setting; final51713 summary+done SHA validated.'))
runs.append(mk('Tree nested branching fresh200',tree_sid,[(T['experiments'][1],1,1),(T['experiments'][2],2,5),(T['experiments'][0],6,200)],T['at'],note='Current source778; ancestor1 and ancestor2–5 only, failed parent outer2/6 discarded from effective history. Complete134, native rolling134 saved,135 sampling. Observation-length>500 warnings are present; they are not a terminal training error.'))
runs.append(mk('Recursive A',a_sid,[(A['experiments'][0],1,200)],A['at'],note='Original first100 + recovery-from100 outers101–200; later recovery metrics override abandoned original101–118. Final51713 summary hash read back, prior unique-coverage acceptance agrees.'))
runs.append(mk('C1 root-only ablation150',a_sid,[(A['experiments'][1],1,150)],A['at'],target=150,note='Newest setting in this session; 150-outer ablation, not the completed RecursiveA. Completed97,98 next; development milestones5/50/100/150. This150-outer condition has no scheduled outer200/final51713 in its current goal.'))
runs[-1]['final51713']={'complete':False,'em':None,'f1':None,'n':None,'path':None,'reason':'Current condition targets150 iterations; no200/final51713 is scheduled.'}
runs.append(mk('Recursive B four GPU',b_sid,[(B['experiments'][0],1,200)],B['at'],note='HKUST5 GPUs0–3 RTX4090; effective ancestors1–3,4–8,current9–130. Complete130/native full finite manifest130/2600 Adam each rank;131 recursive generation still progressing.'))
for r in runs:
 for f in r['fixed512']+r['additional_fixed512']:
  if f['em'] is not None:f['correct']=round(f['em']*512)
checks=[]
for n in ['tree','a-c1','b']:
 vals=[]
 for k in [1,2,3]:
  p=D/f'{n}-observation-{k}.json'
  if not p.exists():continue
  d=json.loads(p.read_text())
  if 'experiments' not in d:continue
  vals.append({'file':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'reader_start_epoch':d['at'],'logs':[{'run':e['name'],'path':l['path'],'bytes':l['bytes'],'mtime':l['mtime'],'last_step':l['rows'][-1]['step'] if l['rows'] else None} for e in d['experiments'] for l in e['logs']]})
 checks.append({'task':n,'observations':vals})
f=next(x for x in T['experiments'][3]['receipts'] if x['path'].endswith('/scores/final-metrics.json'));done=next(x for x in T['experiments'][3]['receipts'] if x['path'].endswith('/scores/final-metrics.done.json'));assert f['sha256']==done['data']['sha256']
checks.append({'name':'BRANPO final summary SHA256 matches done marker','path':f['path'],'sha256':f['sha256'],'passed':True})
obj={'created_cst':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'scope':'Read-only session location + direct remote emitted scalar logs/small completed receipts. No messages, signals, training/config edits, model loading or raw rollout download.','runs':runs,'checks':checks,'connection_note':'Initial Tree observer direct cold route timed out90s; existing authenticated Tree socket require-socket read succeeded twice (2,3) and prefix4. Does not imply training failure.','metric_note':'Training EM is mean over actual scored retained rollout trajectories for512 sampled questions, not fixed512 heldout; Tree/Recursive groups differ from5 independent rollouts. EM/F1 fractions in[0,1].','acceptance_limit':'Completed full evaluation summary/receipt hashes verified; this audit did not re-score every prediction or rehash model tensors.'}
(D/'tree-e1-b-audit.json').write_text(json.dumps(obj,ensure_ascii=False,indent=2)+'\n')
print(json.dumps([{k:r[k] for k in ['name','completed_iter','train_latest_em','train_recent10_mean','fixed512','final51713']} for r in runs],ensure_ascii=False))
