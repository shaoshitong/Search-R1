"""Normalize direct scalar observations; preserve A/C1/C2 as separate conditions."""
import copy,datetime,hashlib,json
from pathlib import Path
P=Path(__file__).resolve().parent
D=json.loads((P/'a-observation-2.json').read_text())
BASE=json.loads((P/'all-seven-results.json').read_text())
OLD={x['name']:x for x in BASE['runs']}
TZ=datetime.timezone(datetime.timedelta(hours=8))
def stamp(t):return datetime.datetime.fromtimestamp(t,TZ).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
observed=stamp(D['finished_at'])
lives=[json.loads((P/f'a-live-{i}.json').read_text()) for i in [1,2]]
assert all(x['entry_process']['pid']==2558506 and x['entry_exit'] is None and x['actual_exit'] is None for x in lives)
assert lives[1]['time']>lives[0]['time'] and lives[1]['log']['bytes']>lives[0]['log']['bytes']
proof=dict(observed_start_cst=stamp(lives[0]['time']),observed_end_cst=stamp(lives[1]['time']),same_owner=lives[1]['entry_process'],completed_outer_before=lives[0]['progress']['completed_outer'],completed_outer_after=lives[1]['progress']['completed_outer'],log_bytes_growth=lives[1]['log']['bytes']-lives[0]['log']['bytes'],next_outer=49,positive_trajectory_events=[x for x in lives[1]['log']['evidence'] if 'trajectory_complete' in x and '"outer": 49' in x],last_events=lives[1]['log']['evidence'][-2:],entry_exit=None,study_exit=None,remote_mutations=False)
assert proof['positive_trajectory_events']
(P/'a-progress-proof.json').write_text(json.dumps(proof,ensure_ascii=False,indent=2)+'\n')
runs=[]
for e in D['experiments']:
 name=e['name'];target=200 if name=='Recursive A' else 150
 r=copy.deepcopy(OLD.get(name,dict(name=name,session='01a1028b-b626-7f91-93db-0099ff1623c5',output_dir=e['root'],additional_fixed512=[])))
 allrows={}
 for lg in e['logs']:
  for row in lg['rows']:
   if name=='Recursive A' and 'recovery-from100' not in lg['path'] and row['iter']>100:continue
   if row['iter']>target:continue
   m=row['metrics']
   if m.get('training/actual_outer_updates')!=row['iter']:continue
   if 'critic/score/mean' not in m:continue
   allrows[row['iter']]=dict(iter=row['iter'],em=m['critic/score/mean'],path=lg['path'],line=row['line'],sha256=row['sha256'],sha256_scope='complete emitted scalar line',metrics=m)
 series=[];fixed=[]
 for it,row in sorted(allrows.items()):
  m=row['metrics'];series.append({k:v for k,v in row.items() if k!='metrics'})
  if 'val/em/all' in m:
   n=m.get('val/qa_count/all')
   assert n==512
   fixed.append(dict(iter=it,em=m['val/em/all'],f1=m.get('val/f1/all'),n=512,correct=round(m['val/em/all']*512),path=row['path'],line=row['line'],sha256=row['sha256'],sha256_scope='complete emitted scalar line'))
 assert series
 previous={x['iter']:x['em'] for x in r.get('train_series',[])}
 overlap=[(s['iter'],abs(s['em']-previous[s['iter']])) for s in series if s['iter'] in previous]
 assert all(x[1]<1e-10 for x in overlap)
 if name=='Recursive A':
  assert len(series)==200 and series[-1]['iter']==200
  final=next(x for x in e['receipts'] if x['path'].endswith('/summary.json'))
  assert final['sha256']==r['final51713']['sha256']
  m=final['data']['metrics'];r['final51713']['source_metrics']=m
  r['final51713']['verified_at_cst']=observed
  completed=200;status='training_complete'
  note='Original first100 + recovery-from100101–200, excluding abandoned original101–118. Complete series and final51713 summary hash freshly agree with previous audit. No new training.'
 elif name.startswith('C1'):
  done=next(x for x in e['receipts'] if x['path'].endswith('/attempt-b977dbd562cf446694bb189a67cbe438/training-complete.json'))
  ex=next(x for x in e['receipts'] if x['path'].endswith('/attempt-b977dbd562cf446694bb189a67cbe438/entry-exit.json'))
  assert done['data']['completed_outer']==150 and ex['data']['rc']==0
  assert all(v==3000 for v in done['data']['actual_optimizer_calls_per_rank'].values())
  completed=150;status='training_complete'
  r['terminal_receipt']=dict(path=done['path'],sha256=done['sha256'],entry_exit_path=ex['path'],entry_exit_sha256=ex['sha256'],actual_exit_code=0,optimizer_calls_per_rank=done['data']['actual_optimizer_calls_per_rank'])
  r['final51713']=dict(complete=False,n=None,em=None,f1=None,path=None,reason='150-outer mechanism ablation; full51713 evaluation is not scheduled in this goal, not a failed/missing final evaluation.')
  note='Root-only ablation completed150/150 and actual entry exit0. Each rank Adam3000. Native scalar outer1 has no emitted critic/score/mean, so curve starts2; no fabricated first point. Final val duplicate at step151 is not training and excluded. Separately retain original A and newer C2.'
 else:
  progress=next(x for x in e['receipts'] if x['path'].endswith('/attempt-f4a75df68f084f90997dbaa2f9e5ca8c/progress.json'))
  completed=progress['data']['completed_outer'];assert completed==series[-1]['iter']==48
  assert all(v==960 for v in progress['data']['optimizer_calls_per_rank'].values())
  status='training_running'
  r['final51713']=dict(complete=False,n=None,em=None,f1=None,path=None,reason='150-outer random-final mechanism ablation; full51713 evaluation is not scheduled in current goal.')
  r['runtime_progress']=proof
  r['checkpoint_manifest']=dict(**progress['data']['checkpoint'],completed=completed,optimizer_calls_per_rank=progress['data']['optimizer_calls_per_rank'])
  note='New independent C2 random-final ablation: constructs originalA reward-guided tree, keeps root and uniformly draws4 real nonroot complete candidates for training, not reward-blind tree generation. Complete48/150, Adam960/rank, next49 positive sampled-token events observed twice. Formaldev50 not generated yet; onlydev5 is present.'
 r.update(target_iter=target,completed_iter=completed,train_series=series,fixed512=fixed,train_latest_em=series[-1]['em'],train_recent10_mean=sum(x['em'] for x in series[-10:])/len(series[-10:]),train_recent10_n=len(series[-10:]),status=status,note=note,reader_started_epoch=D['at'],observed_at_cst=observed,audit_file=str(P/'a-observation-2.json'),log_evidence=[{k:x[k] for k in ['path','bytes','mtime']} for x in e['logs']],overlap_audit=dict(previous_overlap_n=len(overlap),max_abs_delta=max([x[1] for x in overlap],default=None)))
 runs.append(r)
out=dict(observed_at_cst=observed,runs=runs,read_only=True,training_agent_messages_sent=0,remote_state_changed=False,observations=[dict(path=str(P/'a-observation-2.json'),sha256=sha(P/'a-observation-2.json')),dict(path=str(P/'a-live-1.json'),sha256=sha(P/'a-live-1.json')),dict(path=str(P/'a-live-2.json'),sha256=sha(P/'a-live-2.json'))],progress_proof=str(P/'a-progress-proof.json'),limitations=['First full reader timed out after60seconds; optimized retry succeeded. Timeout is an observation failure, not a training exit.','OriginalA recovered101–200 is canonical; abandoned original101–118 excluded.','C1/C2 target150 and full51713 not scheduled.'])
(P/'a-updated-runs.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
print(json.dumps(dict(observed_at_cst=observed,runs=[dict(name=r['name'],complete=r['completed_iter'],latest_train_em=r['train_latest_em'],fixed=[(x['iter'],x['em'],x['f1']) for x in r['fixed512']]) for r in runs]),ensure_ascii=False))
