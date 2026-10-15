"""Raise one owned PENDING job by one actually existing partition tier."""
import argparse
import json
from pathlib import Path
import shlex

from .common import BASE
from .control import CONTROL_PYTHON
from .local_control import run


def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=('preparation','formal'),default='formal')
    p.add_argument('--to',choices=('acd_ue','emergency_acd'),required=True)
    p.add_argument('--output',required=True,type=Path);args=p.parse_args()
    code='''import fcntl,json,os,re,subprocess,time
from pathlib import Path
b=Path(BASE);r=json.loads((b/'preparation'/(PHASE+'-submission.json')).read_text());job=r['job']
assert str(job).isdigit()
with (b/'preparation'/(PHASE+'-submit.lock')).open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 raw=subprocess.check_output(['scontrol','show','job','-o',job],text=True)
 fields=dict(re.findall(r'(\\w+)=([^ ]+)',raw))
 assert fields['UserId'].endswith('('+str(os.getuid())+')') and fields['JobName']=='dynrpo-periodic5-'+PHASE
 assert fields['WorkDir']==str(b) and fields['JobId']==job
 if fields['JobState']!='PENDING':
  print(json.dumps(dict(status='NO_PRIORITY_CHANGE_ALREADY_'+fields['JobState'],job=job)));raise SystemExit(0)
 expected={'acd_ue':'acd_u','emergency_acd':'acd_ue'}[TARGET]
 assert fields['Partition']==expected,'Only one verified upward tier is permitted'
 decision=dict(job=job,from_partition=expected,to_partition=TARGET,
  reason='latest human requested low first then gradual increase; actual current pending scheduling evidence',
  observed_job=fields,checked_unix=time.time())
 path=b/'preparation'/(PHASE+'-priority-decision-'+str(time.time_ns())+'.json')
 with path.open('x') as f:json.dump(decision,f);f.flush();os.fsync(f.fileno())
 r=subprocess.run(['scontrol','update','JobId='+job,'Partition='+TARGET],text=True,capture_output=True)
 result=dict(job=job,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,decision_file=str(path))
 result['actual_queue']=subprocess.check_output(['squeue','-h','-j',job,'-o','%i|%T|%P|%R|%S'],text=True)
 result_path=path.with_name(path.stem+'-result.json')
 with result_path.open('x') as f:json.dump(result,f);f.flush();os.fsync(f.fileno())
 print(json.dumps(result))
 raise SystemExit(r.returncode)
'''
    prefix='BASE='+repr(str(BASE))+'\nPHASE='+repr(args.phase)+'\nTARGET='+repr(args.to)+'\n'
    result=run(CONTROL_PYTHON+' -c '+shlex.quote(prefix+code))
    value=dict(control_returncode=result.returncode,stdout=result.stdout.decode(),stderr=result.stderr.decode())
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/(args.phase+'-priority-'+args.to+'.json')).write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps(value));raise SystemExit(result.returncode)


if __name__=='__main__':main()
