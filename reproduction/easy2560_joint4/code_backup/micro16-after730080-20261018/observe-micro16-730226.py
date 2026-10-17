import json,os,time,subprocess,hashlib
from pathlib import Path
b=Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-micro16-after730080-20261018')
def read(p):return json.loads(p.read_text()) if p.is_file() else None
def command(argv):
 r=subprocess.run(argv,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=20);return dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
def tail(p,size=12000):
 if not p.is_file():return ''
 with p.open('rb') as f:f.seek(max(0,p.stat().st_size-size));return f.read().decode(errors='replace')
run=read(b/'logs/run-manifest.json');binding=read(b/'preparation/run-source.json');status=read(b/'logs/pipeline-status.json');allocation=read(b/'logs/native-allocation-admission.json')
updates=[]
p=b/'logs/actual-updates.jsonl'
if p.is_file():
 for raw in p.read_bytes().splitlines():
  try:r=json.loads(raw)
  except ValueError:continue
  updates.append({k:r[k] for k in ('run_id','step','attempt_id','attempt_index','optimizer_writes_before','optimizer_writes_after','actual_optimizer_writes') if k in r})
ranks={}
for rank in range(4):
 p=b/'logs'/('optimizer-rank-%d.jsonl'%rank)
 if not p.is_file():continue
 terminal=[];lastwrite=None;rows=0
 for raw in p.read_bytes().splitlines():
  try:r=json.loads(raw)
  except ValueError:continue
  rows+=1
  if r.get('kind')=='OPTIMIZER_WRITE':lastwrite={k:r[k] for k in ('write_count','optimizer_call_count','context') if k in r}
  if r.get('kind') in ('OUTER_BUDGET','ABORTED_ATTEMPT'):terminal.append({k:r[k] for k in ('kind','rank','logical_update','cumulative_writes','optimizer_writes','local_micro','local_logical_micro','local_mini','gradient_accumulation','attempt_id','attempt_index') if k in r})
 ranks[str(rank)]=dict(bytes=p.stat().st_size,rows=rows,lastwrite=lastwrite,last_terminal=terminal[-2:])
statfiles=[]
for category in ('mixed-candidate-pools','mixed-selection','behavior-traces'):
 root=b/'artifacts'/category
 if root.exists():
  for p in root.rglob('*'):
   if p.is_file():
    st=p.stat();statfiles.append(dict(category=category,path=str(p.relative_to(b)),bytes=st.st_size,mtime_ns=st.st_mtime_ns))
proof=dict(checked_unix=time.time(),base=str(b),job='730226',job_id=(b/'job.id').read_text().strip(),job_state=command(['sacct','-j','730226','--format=JobIDRaw,State,ExitCode,NodeList,Elapsed','--noheader','--parsable2']),queue=command(['squeue','-h','-j','730226','--format=%i|%T|%N|%R']),run_id=run.get('run_id') if run else None,source_commit=run.get('source_commit') if run else None,pipeline=status,allocation=allocation,configuration=read(b/'logs/easy2560-configuration-prepared.json'),fresh_resume_path=run.get('full_state_restore') if run else None,run_fields={k:run.get(k) for k in ('actor_micro_batch_per_rank','actor_logical_micro_batch_per_rank','all_correct_virtual_zero','data_selection_mode','selection_candidate_questions','selection_selected_questions','selection_fixed_candidate_budget','training_world_size','checkpoint_dir','question_probe','completed_updates')} if run else None,source_binding_commit=binding.get('source_commit') if binding else None,completed_updates=updates,ranks=ranks,artifact_metadata=statfiles,console_tail=tail(b/'logs/training-730226.log'),trainer_tail=tail(Path(run['train_log'])) if run and run.get('train_log') else '',tensor_contents_read=False)
print(json.dumps(proof,allow_nan=False))
