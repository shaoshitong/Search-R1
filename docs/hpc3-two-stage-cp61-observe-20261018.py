import json,shlex,subprocess
from pathlib import Path
from reproduction.hpc3_prefix5.direct_control import run
from reproduction.hpc3_prefix5.control import PYTHON,OUTPUT
from reproduction.hpc3_prefix5.observe import query_for
b='/data/user/jxu957/experiments/dynrpo-prefix5-thread5-20261017-r4'
extra=r'''
result['full_pool']=[{k:v.get(k) for k in ['stage','rank','completed_trajectories','expected_trajectories','actual_sampled_tokens','elapsed_seconds']} if (v:=read(r/'full-pool'/('rank-%d-progress.json'%i))) else None for i in range(4)] if status else None
result['selected']={k:joint['rollout'].get(k) for k in ['questions','trajectories','actual_selected_mixed_questions','actual_selected_mixed_fraction','fallback_count','projected_generation_requests']} if joint else None
if status:
 result['active_log_tails']={}
 for name,item in (ledger or {}).get('jobs',{}).items():
  if item.get('state')!='STOPPED':
   p=b/'logs'/(name+'.log')
   if p.exists():
    with p.open('rb') as f:f.seek(max(0,p.stat().st_size-1800));result['active_log_tails'][name]=f.read().decode(errors='replace')
'''
code=query_for(b).replace('print(json.dumps(result,allow_nan=False))',extra+'\nprint(json.dumps(result,allow_nan=False))')
addr=subprocess.check_output(['ipconfig','getifaddr','en0'],text=True).strip();r=run(PYTHON+' -c '+shlex.quote(code),account='mou176',source_address=addr,timeout=25)
if r.returncode:raise RuntimeError(r.stdout.decode()[-1800:]+r.stderr.decode()[-500:])
v=json.loads(r.stdout);p=Path(OUTPUT);(p/'two-stage-cp61-current.json').write_text(json.dumps(v,indent=2)+'\n')
with (p/'two-stage-cp61-observations.jsonl').open('a') as f:f.write(json.dumps(v)+'\n')
print(json.dumps(v))
