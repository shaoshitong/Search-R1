import json,sys,time,subprocess
from pathlib import Path
sys.path.insert(0,'/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fastpath-probe-20261022/source/ab8c44c8b68d912399b77b354fee5a832b5e9f9a')
from reproduction.operations.easy7b_20261021.observe_lean_transition_hpc import observe
base=Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fa3-searchstop-r1-20261023')
r=observe(str(base),'06c5f91772d03358c703dbfc0c2ed935c1784153')
job=r['job'];jobs='745198'+(','+job if job else '')
r['queue']=subprocess.run(['squeue','-h','-j',jobs,'-o','%i|%u|%T|%N|%R'],capture_output=True,text=True,timeout=12).stdout
r['candidate_progress']=[]
for p in (base/'artifacts/mixed-candidate-pools').glob('*/progress.json'):
 try:
  v=json.loads(p.read_text());before=p.parent/'0-before-generation.pt'
  r['candidate_progress'].append({'step':v.get('step'),'complete_questions':len(v.get('complete_ordinals',[])),'source_consumed':v.get('source_consumed'),'first_before_generation':before.stat().st_mtime if before.exists() else None,'saved_chunks':[{'name':q.name,'bytes':q.stat().st_size,'mtime':q.stat().st_mtime} for q in p.parent.glob('*.pt') if q.stem.isdecimal()]})
 except json.JSONDecodeError:pass
r['candidate_start_boundaries']=[{'pool':p.parent.name,'mtime':p.stat().st_mtime,'bytes':p.stat().st_size} for p in (base/'artifacts/mixed-candidate-pools').glob('*/0-before-generation.pt')]
r['selection_summary']=[]
for p in (base/'artifacts/mixed-selection').glob('*.json'):
 try:
  v=json.loads(p.read_text());r['selection_summary'].append({k:v.get(k) for k in ('step','candidate_fill_seconds','selected_mixed','raw_generated_trajectories','raw_generated_policy_tokens','raw_actual_retrieval_calls')})
 except json.JSONDecodeError:pass
r['transition_documents']={}
for name in ['lean-transition-retrieval-launch-claim.json','lean-transition-retrieval-launched.json','lean-transition-retrieval-exit.json','fa3-source-stop-result.json','fa3-transition-source-ready.json','fa3-transition-submitted.json','training-submission.json']:
 p=base/'preparation'/name
 if p.exists():
  try:r['transition_documents'][name]=json.loads(p.read_text())
  except json.JSONDecodeError:pass
p=base/'logs/lean-transition-retrieval-cpu-audit.log'
if p.exists():
 with p.open('rb') as f:
  f.seek(max(0,p.stat().st_size-6000));r['transition_cpu_log_tail']=f.read().decode(errors='replace')
if job:r['accounting']=subprocess.run(['sacct','-n','-X','-j',job,'--format=JobIDRaw,User,State,ExitCode,Elapsed,NodeList','-P'],capture_output=True,text=True,timeout=12).stdout
print(json.dumps(r),flush=True)
