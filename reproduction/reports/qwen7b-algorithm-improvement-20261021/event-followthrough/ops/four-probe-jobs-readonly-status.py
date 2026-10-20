"""Read-only actual progress for hard/random and matched real-G10 suites."""
import json,pathlib,subprocess,time
A=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021')
def read(p):return json.loads(p.read_text())
def command(cmd):
 r=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=20)
 return dict(code=r.returncode,stdout=r.stdout.strip(),stderr=r.stderr[-1000:])
out=dict(observed_epoch=time.time(),variants={});ids=[]
for family,names in [('hard-vs-random-E4-477-v1',('targeted','random_control')),('g10-matched6528-v1',('random','global-weighted-balanced'))]:
 for name in names:
  root=A/family/name;v={};out['variants'][family+'/'+name]=v
  if (root/'job.id').exists():
   job=(root/'job.id').read_text().strip();assert job.isdigit();ids.append(job);v['job_id']=job
  for fn in ('controller.json','progress.json','latest.json','failure.json','interrupted.json','evaluation-progress.json','gpu-memory.json','complete.json'):
   if (root/fn).exists():
    row=read(root/fn);v[fn]={k:row[k] for k in ('status','cursor','steps','source_steps','loss','time','epoch_time','pid','message','traceback','path','bytes','sha256','actor_updates','completed_questions','completed_trajectories','metrics','peak_allocated_bytes','peak_reserved_bytes') if k in row}
    if fn=='progress.json' and 'grouping' in row:
     s=row['grouping'];v['grouping']={k:s[k] for k in ('mode','questions','views','mixed_views','nonmixed_views','count_balanced','count_mixed_fraction','weighted_class_mass_balanced','weighted_mixed_fraction','weighted_mixed_mass','weighted_nonmixed_mass','per_question_total_weight_exactly_one','view_weight_always_one_quarter','exactly_four_unique_views_per_question') if k in s}
  v['replays']={}
  for p in list(root.glob('next-update-replay-cursor*.json'))+list(root.glob('next-update-replay.json')):
   row=read(p);v['replays'][p.name]={k:row[k] for k in ('status','passed','source_steps','accepted_steps','candidate_steps') if k in row};v['replays'][p.name]['all_component_hashes_equal']=row.get('expected')==row.get('actual')
  curve=root/'curve.jsonl'
  if curve.exists():
   rows=[json.loads(x) for x in curve.read_text().splitlines() if x.endswith('}')]
   complete=[r for r in rows if 'cursor' in r and 'steps' in r];tail=complete[-8:]
   v['actual_curve_rows']=len(complete)
   if len(tail)>1:
    times=[r.get('time',r.get('epoch_time')) for r in tail];v['last_actual_seconds_per_update']=(times[-1]-times[0])/(tail[-1]['cursor']-tail[0]['cursor'])
  logs=sorted((root/'logs').glob('*.log')) if (root/'logs').exists() else []
  if logs:
   with logs[-1].open('rb') as f:f.seek(max(0,logs[-1].stat().st_size-1500));v['log_tail']=f.read().decode(errors='replace')
if ids:
 out['squeue']=command(['squeue','-h','-j',','.join(ids),'-o','%i|%j|%T|%P|%M|%l|%R'])
 out['sacct']=command(['sacct','-X','-n','-P','-j',','.join(ids),'--format=JobID,State,ExitCode,Elapsed,Timelimit'])
print(json.dumps(out))
