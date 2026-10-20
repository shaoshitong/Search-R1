import pathlib,subprocess,json,time
S=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/matched-loss128-v2')
def cmd(args):
 p=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=20);assert p.returncode==0,p.stdout+p.stderr;return p.stdout.strip()
records=[]
for name,job in [('trajectory','742868'),('joint','742869'),('event','742870')]:
 b=S/name;assert (b/'job.id').read_text().strip()==job
 before=cmd(['scontrol','show','job',job,'-o']);assert 'JobName=q7b-loss128-'+name+' ' in before and 'UserId=mou176(21930)' in before and 'Command='+str(b/'fit.sbatch') in before
 if 'JobState=PENDING ' in before:
  cmd(['scontrol','update','JobId='+job,'Partition=acd_ue','TimeLimit=01:00:00']);after=cmd(['scontrol','show','job',job,'-o']);assert 'Partition=acd_ue ' in after and 'TimeLimit=01:00:00' in after;action='PENDING_JOB_UPDATED_IN_PLACE_TO_MEDIUM'
 else:after=before;action='ALREADY_STARTED_NOT_CHANGED'
 records.append(dict(variant=name,job_id=job,action=action,before=before,after=after))
r=dict(status='MATCHED_SUITE_QUEUE_ADJUSTMENT_RECONCILED',reason='Ordinary acd_u actual scheduler estimate2026-10-21T07:39/09:52 (~7-9h); stepwise advance to middle tier200, no emergency tier300',records=records,source_and_config_changed=False,no_resubmission=True,queue=cmd(['squeue','-h','-j','742868,742869,742870','-o','%i|%j|%T|%P|%R']),estimate=cmd(['squeue','--start','-h','-j','742868,742869,742870','-o','%i|%T|%S|%R']),time=time.time());(S/'queue-escalation.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
