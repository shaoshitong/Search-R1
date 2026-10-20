import pathlib,subprocess,json,time
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/replay-layer-qualification-20261021');j=(b/'job.id').read_text().strip();assert j=='742884'
def command(args):
 p=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=15);assert p.returncode==0,p.stdout+p.stderr;return p.stdout.strip()
before=command(['scontrol','show','job',j,'-o']);assert 'UserId=mou176(21930)' in before and 'JobName=q7b-replay-qual ' in before and 'Command='+str(b/'replay-qualification.sbatch') in before
if 'JobState=PENDING ' in before:
 command(['scontrol','update','JobId='+j,'Partition=acd_ue','TimeLimit=01:00:00']);action='PENDING_NORMAL_TO_MEDIUM_IN_PLACE'
else:action='ALREADY_STARTED_NO_CHANGE'
after=command(['scontrol','show','job',j,'-o']);record=dict(job_id=j,action=action,reason='Normal acd_u actual fresh estimate03:48:29 around00:45; stepwise medium, no emergency',before=before,after=after,no_resubmit=True,source_unchanged=True,time=time.time());(b/'queue-escalation.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
