"""Read actual Slurm state and small progress evidence; no recurring watcher."""
import argparse
import json
from pathlib import Path
import shlex
import time

from .common import BASE
from .control import CONTROL_PYTHON
from .local_control import run


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    code='''import json,subprocess,time
from pathlib import Path
b=Path(BASE);o=dict(sampled_unix=time.time())
names=['preparation/preparation-submission.json','preparation/formal-submission.json',
       'preparation/progress.json','warmup/ready.json','logs/pipeline-status.json',
       'logs/cleanup-status.json','logs/full-state-restored-driver.json']
for name in names:
 p=b/name
 if p.is_file():o[name]=json.loads(p.read_text())
for phase in ['preparation','formal']:
 r=o.get('preparation/'+phase+'-submission.json',{})
 if not r.get('job'):continue
 job=r['job']
 o[phase+'_queue']=subprocess.check_output(['squeue','-h','-j',job,'-o','%i|%T|%P|%R|%S'],text=True)
 o[phase+'_accounting']=subprocess.check_output(['sacct','-nP','-X','-j',job,'--format=JobID,State,ExitCode,NodeList'],text=True)
 p=b/'logs'/(phase+'-'+job+'.log')
 if p.exists():
  with p.open('rb') as f:f.seek(max(0,p.stat().st_size-6000));o[phase+'_log_tail']=f.read().decode(errors='replace')
for p in sorted((b/'artifacts/question-probe/periodic-refits').glob('*/latest-state.json')):
 o[str(p.relative_to(b))]=json.loads(p.read_text())
for p in sorted((b/'artifacts/question-probe/periodic-refits').glob('*/complete.json')):
 value=json.loads(p.read_text());o[str(p.relative_to(b))]={k:value[k] for k in ['stage','outer','fit_epochs','head_adam_steps_after','trajectories']}
for name in ['logs/actual-updates.jsonl']+['logs/optimizer-rank-'+str(i)+'.jsonl' for i in range(4)]:
 p=b/name
 if p.is_file():
  with p.open('rb') as f:f.seek(max(0,p.stat().st_size-14000));lines=f.read().decode(errors='replace').splitlines()
  rows=[]
  for line in lines[-3:]:
   try:rows.append(json.loads(line))
   except ValueError:pass
  o[name]=rows
print(json.dumps(o,ensure_ascii=False))
'''
    result=run(CONTROL_PYTHON+' -c '+shlex.quote('BASE='+repr(str(BASE))+'\n'+code))
    record=dict(control_returncode=result.returncode,control_error=result.stderr.decode(),checked_unix=time.time())
    if not result.returncode:record.update(json.loads(result.stdout))
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'latest-sample.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    with (args.output/'samples.jsonl').open('a') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
    print(json.dumps({k:v for k,v in record.items() if k not in ('preparation_log_tail','formal_log_tail')},ensure_ascii=False))
    print(record.get('formal_log_tail',record.get('preparation_log_tail',''))[-1500:])
    raise SystemExit(result.returncode)


if __name__=='__main__':main()
