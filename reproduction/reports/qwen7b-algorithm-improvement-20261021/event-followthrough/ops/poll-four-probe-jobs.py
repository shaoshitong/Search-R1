"""One-shot read-only poll; caller controls spacing, never submits/restarts anything."""
import argparse,json,pathlib,subprocess,sys
p=argparse.ArgumentParser();p.add_argument('--index',type=int,required=True);a=p.parse_args()
O=pathlib.Path(__file__).resolve().parent
r=subprocess.run([sys.executable,str(O/'balanced-control.py'),str(O/'four-probe-jobs-readonly-status.py'),'--timeout','55'],
    stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)
if r.returncode:
 print(json.dumps(dict(status='READ_FAILED_NO_MUTATION',code=r.returncode,stdout_tail=r.stdout[-1200:],stderr_tail=r.stderr[-1200:])));sys.exit(r.returncode)
rows=[json.loads(line) for line in r.stdout.splitlines() if line.startswith('{')];assert rows
d=rows[-1];dest=O/'reports'/('four-probe-jobs-actual-%d.json'%a.index);dest.write_text(json.dumps(d,indent=2)+'\n')
states={}
for line in d.get('sacct',{}).get('stdout','').splitlines():
 cols=line.split('|')
 if len(cols)>2:states[cols[0]]=dict(state=cols[1],exit_code=cols[2])
out=dict(index=a.index,observed_epoch=d['observed_epoch'],variants={},needs_attention=False)
for name,v in d['variants'].items():
 row=dict(job_id=v.get('job_id'),cursor=v.get('progress.json',{}).get('cursor'),saved_cursor=v.get('latest.json',{}).get('cursor'),
     scheduler=states.get(v.get('job_id')),seconds_per_update=v.get('last_actual_seconds_per_update'))
 if 'evaluation-progress.json' in v:row['evaluation_progress']=v['evaluation-progress.json']
 if 'prediction_files' in v:row['prediction_files']=v['prediction_files']
 if 'complete.json' in v:
  done=v['complete.json'];row['complete_status']=done['status'];row['metrics']={}
  for split,m in done['metrics'].items():
   pool=m['pool'];chosen=m['selected512'];row['metrics'][split]=dict(BA=pool['mixed_balanced_accuracy'],
    Mixed_recall=pool['mixed_recall'],nonMixed_recall=pool['nonmixed_recall'],AUC=pool['mixed_roc_auc'],
    selected_Mixed=chosen['mixed_questions'],selected_questions=chosen['questions'],selected_Mixed_fraction=chosen['mixed_fraction'])
 if 'failure.json' in v:row['failure']=v['failure.json'];out['needs_attention']=True
 if row['scheduler'] and row['scheduler']['state'] in ('FAILED','TIMEOUT','OUT_OF_MEMORY','CANCELLED','NODE_FAIL'):out['needs_attention']=True
 if row['scheduler'] and row['scheduler']['state']=='COMPLETED' and 'complete_status' not in row:out['needs_attention']=True
 out['variants'][name]=row
out['all_complete']=all('complete_status' in row for row in out['variants'].values())
if 'evaluation_sstat' in d:out['evaluation_sstat']=d['evaluation_sstat']
print(json.dumps(out))
