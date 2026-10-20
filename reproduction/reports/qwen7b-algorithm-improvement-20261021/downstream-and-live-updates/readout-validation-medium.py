import pathlib,json,subprocess,time
B=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019');A=B/'algorithm-improvement-20261021';suite=A/'l27-readout128-v1'
def read(p):return json.loads(pathlib.Path(p).read_text())
def run(args):
 p=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=20)
 assert p.returncode==0,p.stdout+p.stderr
 return p.stdout.strip()
targets=[]
for name,item in read(suite/'preparation.json')['variants'].items():
 base=pathlib.Path(item['base']);targets.append(((base/'job.id').read_text().strip(),item['job_name'],base/'fit.sbatch'))
for label in ('A','B'):
 base=A/('validation-g10-20261021/materialized/event-'+label)
 targets.append(((base/'job.id').read_text().strip(),'q7b-validG10-'+label+'-20261021',base/'sampling.sbatch'))
record=[]
for job,name,script in targets:
 assert job.isdigit()
 before=run(['scontrol','show','job',job,'-o'])
 assert 'UserId=mou176(21930)' in before and 'JobName='+name+' ' in before and 'Command='+str(script)+' ' in before
 if 'JobState=PENDING ' in before and 'Partition=acd_u ' in before:
  estimate=run(['squeue','--start','-h','-j',job,'-o','%i|%S|%R'])
  run(['scontrol','update','JobId='+job,'Partition=acd_ue']);action='NORMAL_TO_MEDIUM_SAME_ID'
 else:estimate='';action='ALREADY_STARTED_OR_ESCALATED_NO_CHANGE'
 after=run(['scontrol','show','job',job,'-o'])
 record.append(dict(job_id=job,name=name,action=action,estimate_before=estimate,before=before,after=after))
result=dict(status='OWN_READOUT_VALIDATION_MEDIUM_RECONCILED',reason='Normal validation estimates02:20 and04:20 at01:30; gradual medium escalation, no emergency',records=record,source_or_config_changed=False,time=time.time())
dest=A/'queue-adjustments';dest.mkdir(exist_ok=True);(dest/('readout-validation-medium-%d.json'%result['time'])).write_text(json.dumps(result,indent=2));print(json.dumps(result))
