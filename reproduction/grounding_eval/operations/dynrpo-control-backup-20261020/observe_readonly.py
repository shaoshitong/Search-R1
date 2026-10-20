import pathlib,json,subprocess,time,os
root=pathlib.Path('/data/user/jxu957/experiments/dynrpo-nq-grounding-20261020');job=(root/'job.id').read_text().strip();r=dict(checked_unix=time.time(),job=job)
r['queue']=subprocess.run(['squeue','-j',job,'-h','-o','%i|%T|%N|%R'],capture_output=True,text=True).stdout
r['accounting']=subprocess.run(['sacct','-j',job,'-n','-P','--format=JobIDRaw,State,ExitCode,NodeList'],capture_output=True,text=True).stdout
for rel in ('control/worker.json','control/progress.json','control/failure.json','control/complete.json','normal/progress.json','normal/complete.json','cf-v4/summary.json','cf-v5/summary.json'):
 p=root/rel
 if p.is_file():
  d=json.loads(p.read_text())
  r[rel]=d if rel!='control/complete.json' else dict(status=d['status'])
r['logs']={}
for p in (root/'control').glob('*.log'):
 with p.open('rb') as f:f.seek(max(0,p.stat().st_size-2000));tail=f.read().decode(errors='replace')
 r['logs'][p.name]=dict(bytes=p.stat().st_size,tail=tail)
for name in ('cf-v4','cf-v5'):
 p=root/name
 if p.exists():r[name+'-results']=dict(count=len(list((p/'results').glob('*.json'))),bytes=sum(x.stat().st_size for x in (p/'results').glob('*.json')))
print(json.dumps(r))
