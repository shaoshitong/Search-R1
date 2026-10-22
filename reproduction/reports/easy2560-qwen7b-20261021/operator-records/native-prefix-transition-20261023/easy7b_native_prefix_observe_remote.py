import json,sys,time,subprocess
from pathlib import Path
sys.path.insert(0,'/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fastpath-probe-20261022/source/ab8c44c8b68d912399b77b354fee5a832b5e9f9a')
from reproduction.operations.easy7b_20261021.observe_lean_transition_hpc import observe
base=Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fa3-nativeprefix-r1-20261022')
r=observe(str(base),'c7976da3afc863a1c20b99daa7cc99de3bb61f15')
job=r['job'];jobs='745198'+(','+job if job else '')
r['queue']=subprocess.run(['squeue','-h','-j',jobs,'-o','%i|%u|%T|%N|%R'],capture_output=True,text=True,timeout=12).stdout
r['candidate_progress']=[]
for p in (base/'artifacts/mixed-candidate-pools').glob('*/progress.json'):
 try:
  v=json.loads(p.read_text());before=p.parent/'0-before-generation.pt'
  r['candidate_progress'].append({'step':v.get('step'),'complete_questions':len(v.get('complete_ordinals',[])),'source_consumed':v.get('source_consumed'),'first_before_generation':before.stat().st_mtime if before.exists() else None,'saved_chunks':[{'name':q.name,'bytes':q.stat().st_size,'mtime':q.stat().st_mtime} for q in p.parent.glob('*.pt') if q.stem.isdecimal()]})
 except json.JSONDecodeError:pass
r['selection_summary']=[]
for p in (base/'artifacts/mixed-selection').glob('*.json'):
 try:
  v=json.loads(p.read_text());r['selection_summary'].append({k:v.get(k) for k in ('step','candidate_fill_seconds','selected_mixed','raw_generated_trajectories','raw_generated_policy_tokens','raw_actual_retrieval_calls')})
 except json.JSONDecodeError:pass
print(json.dumps(r),flush=True)
