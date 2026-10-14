from pathlib import Path
import sys,json
from h100_remote import call
D=Path(__file__).resolve().parent
c='''from pathlib import Path
import json,re,datetime,subprocess
r=Path('/mnt/data/shaoshitong/search-g1-final200-bs256-20261014-v2');p=r/'logs/final-evaluator.log';progress=[];errors=[]
for l in p.open(errors='replace'):
 if 'SEARCH_G1_FINAL_RESUME_PROGRESS=' in l:
  a=l.split('SEARCH_G1_FINAL_RESUME_PROGRESS=',1)[1].strip();progress.append(json.loads(a))
 if 'Traceback' in l or 'Error:' in l:errors.append(l[-220:])
o={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'job':subprocess.run(['squeue','-j','3395','-h','-o','%i %T %M'],text=True,capture_output=True).stdout,'root':str(r),'progress':progress[-2:],'error_lines':errors[-4:],'log_bytes':p.stat().st_size,'log_mtime':p.stat().st_mtime,'status':json.load((r/'logs/final-pipeline-status.json').open()),'journal_files':[{'name':str(p.relative_to(r/'journal')),'bytes':p.stat().st_size} for p in (r/'journal').rglob('*') if p.is_file() and not p.name.endswith(('.jsonl','.npz'))]}
print(json.dumps(o))'''
v=call('h100-8',c);(D/'final3395-progress-second.json').write_text(json.dumps(v,indent=2));print(json.dumps(v))
