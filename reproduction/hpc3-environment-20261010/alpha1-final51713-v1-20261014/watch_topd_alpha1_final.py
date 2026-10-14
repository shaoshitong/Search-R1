"""Finite read-only checks of alpha1 final evaluation, with small audit mirrors."""
import argparse
import json
import pathlib
import subprocess
import sys

REMOTE = r'''
import pathlib,json,subprocess,time,datetime
BASE=pathlib.Path('/data/user/jxu957/experiments/topd-20261008')
OUT=BASE/'outputs/training-preparation'
JOB=(OUT/'topd-alpha1-final51713-job.id').read_text().strip();assert JOB.isdigit()
FINAL=BASE/'runtime/logs/topd-alpha1-final51713-recovery-713717'
def snapshot():
 q=subprocess.run(['squeue','-j',JOB,'-h','-o','%i|%T|%M|%R'],text=True,capture_output=True)
 progress=[]
 for rank in range(4):
  p=FINAL/('rank%d-progress.json'%rank)
  if p.exists():
   x=json.loads(p.read_text());progress.append({k:v for k,v in x.items() if k!='metrics'})
 errors={};tails={}
 for p in [BASE/('runtime/logs/topd-alpha1-final51713-'+JOB+'.log'),FINAL/'retriever.log']+list(FINAL.glob('rank*.log')):
  if not p.exists():continue
  with p.open('rb') as f:f.seek(max(0,p.stat().st_size-4500));s=f.read().decode(errors='replace')
  tails[p.name]=s[-500:]
  found=[l[-400:] for l in s.splitlines() if any(k in l for k in ['Traceback','RuntimeError','CUDA error','OutOfMemoryError','Killed'])]
  if found:errors[p.name]=found[-5:]
 d={'epoch':time.time(),'observed_at':datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(),'evaluation_job':int(JOB),'training_job':713717,'checkpoint_outer':200,'queue':q.stdout.splitlines(),'queue_rc':q.returncode,'rank_progress':progress,'scored_rows':sum(x['rows'] for x in progress),'total_rows':51713,'complete_receipt_present':(FINAL/'complete.json').is_file(),'checkpoint_audited_steps':sorted(int(p.name.split('-')[1]) for p in FINAL.glob('checkpoint-*-audit.json')),'historical_step5_unavailable':(FINAL/'historical-step5-unavailable.json').is_file(),'errors':errors,'tails':tails,'final_acceptance_complete':(OUT/'topd-alpha1-final-acceptance.json').is_file()}
 if not q.stdout.strip():d['accounting']=subprocess.check_output(['sacct','-j',JOB,'-n','-P','--format=JobID,State,ExitCode,Elapsed'],text=True).splitlines()
 return d
samples=[]
for i in range(COUNT):
 d=snapshot();samples.append(d);print(json.dumps({'kind':'sample','data':d}),flush=True)
 if i<COUNT-1:time.sleep(INTERVAL)
report={'samples':samples,'rows_between_samples':d['scored_rows']-samples[0]['scored_rows'],'latest':d,'historical_step5_missing_never_counted_as_current_checkpoint_pass':True}
p=OUT/'alpha1-final51713-current.json';tmp=p.with_suffix('.json.tmp');tmp.write_text(json.dumps(report,indent=2)+'\n');tmp.replace(p)
print(json.dumps({'kind':'saved','name':p.name,'data':report}),flush=True)
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=int, choices=range(2, 7), default=2)
    parser.add_argument('--interval', type=int, choices=range(1, 56), default=55)
    args = parser.parse_args()
    code = 'COUNT=%d\nINTERVAL=%d\n'%(args.samples, args.interval)+REMOTE
    command = '/data/user/jxu957/experiments/topd-20261008/environment/topd-py310/bin/python - <<\'REMOTE\'\n'+code+'REMOTE\n'
    process = subprocess.Popen([sys.executable, 'work/hpc3_private_control.py'],
                               stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True)
    process.stdin.write(command)
    process.stdin.close()
    for line in process.stdout:
        if not line.startswith('{'):
            print(line.rstrip(), flush=True)
            continue
        item = json.loads(line)
        if item['kind'] == 'saved':
            p = pathlib.Path('outputs/training-preparation')/item['name']
            p.write_text(json.dumps(item['data'], indent=2)+'\n')
            print(json.dumps({'saved': item['name']}), flush=True)
        else:
            print(line.rstrip(), flush=True)
    if process.wait():
        raise SystemExit('Observation failed; inspect connection before inferring job failure.')


if __name__ == '__main__':
    main()
