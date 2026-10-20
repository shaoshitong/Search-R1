import json,pathlib,subprocess,time
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/matched-loss128')
out={}
for j,v in [('742835','trajectory'),('742839','joint'),('742840','event')]:
 d=b/v;assert (d/'job.id').read_text().strip()==j
 for n in ['controller.json','progress.json','latest.json','next-update-replay.json']:
  assert not (d/n).exists(),'Unexpected training state; stop retirement'
 s=subprocess.check_output(['scontrol','show','job',j,'-o'],universal_newlines=True,timeout=15);assert 'JobName=q7b-loss128-'+v+' ' in s and 'UserId=mou176(' in s
 if 'JobState=PENDING ' in s:
  assert 'Reason=JobHeldUser ' in s;subprocess.check_call(['scancel',j],timeout=15)
 else:assert 'JobState=FAILED ' in s
 out[j]=subprocess.check_output(['sacct','-n','-P','-j',j,'-o','JobID,State,ExitCode'],universal_newlines=True,timeout=15)
record=dict(status='DEFECTIVE_UNSTARTED_ENTRYPOINTS_RETIRED',jobs=out,updates=0,baseline_unchanged=True,time=time.time())
(b/'entrypoint-retired.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
