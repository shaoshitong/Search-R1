import pathlib,json,subprocess,time
B=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019');A=B/'algorithm-improvement-20261021';S=A/'matched-loss128-followup-v1/event';p=json.loads((S/'preparation.json').read_text());targets=[]
for name,item in p['variants'].items():
 b=pathlib.Path(item['base']);targets.append(((b/'job.id').read_text().strip(),item['job_name'],b/'fit.sbatch'))
for folder,name,script in [(B/'replay-extended-qualification-20261021','q7b-replay-extqual','replay-extended.sbatch'),(A/'training-extraG5-128-20261021/materialized','q7b-G10pilot-128-20261021','sampling.sbatch')]:targets.append(((folder/'job.id').read_text().strip(),name,folder/script))
def command(args):
 p=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=20);assert p.returncode==0,p.stdout+p.stderr;return p.stdout.strip()
records=[]
for j,name,script in targets:
 before=command(['scontrol','show','job',j,'-o']);assert 'UserId=mou176(21930)' in before and 'JobName='+name+' ' in before and 'Command='+str(script)+' ' in before
 if 'JobState=PENDING ' in before and 'Partition=acd_u ' in before:
  command(['scontrol','update','JobId='+j,'Partition=acd_ue']);action='PENDING_NORMAL_TO_MEDIUM_IN_PLACE'
 else:action='ALREADY_STARTED_OR_ESCALATED_NO_CHANGE'
 after=command(['scontrol','show','job',j,'-o']);records.append(dict(job_id=j,job_name=name,action=action,before=before,after=after))
out=dict(status='IDENTIFIED_OWN_JOBS_STEPWISE_MEDIUM_QUEUE_RECONCILED',reason='Fresh normal-priority estimates02:02..03:51 at01:09 (53..162minutes); medium tier200, no emergency tier300',source_config_weights_changed=False,original_job_ids_preserved=True,records=records,time=time.time());dest=A/'queue-adjustments';dest.mkdir(exist_ok=True);(dest/('medium-%d.json'%out['time'])).write_text(json.dumps(out,indent=2));print(json.dumps(out))
