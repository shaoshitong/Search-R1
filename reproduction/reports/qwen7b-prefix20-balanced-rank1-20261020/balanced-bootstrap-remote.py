import json,pathlib,subprocess,time
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/balanced-rank1-from-epoch1-prefix20-20261020')
out={'time':time.time()}
for name in ['preparation/source-publication.json','config.json','preparation/acceptance.json','submission.json','preparation/handoff.json']:
 p=b/name
 if p.exists():
  v=json.loads(p.read_text())
  if name=='preparation/handoff.json' and 'source_checkpoint' in v:v['source_checkpoint']={k:x for k,x in v['source_checkpoint'].items() if k!='metrics'}
  out[name]=v
for key,args in [('queue',['squeue','-h','-u','mou176','-n','q7b-bal-rank1','-o','%i|%T|%P|%M|%N|%R']),('old_accounting',['sacct','-n','-X','-P','-j','742046','--format=JobID,State,ExitCode'])]:
 r=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=15);out[key]={'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
print(json.dumps(out))
