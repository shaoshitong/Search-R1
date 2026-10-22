import pathlib,json,subprocess,time,datetime
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/frozen-final4096-generation-20261022-v1')
plan=json.loads((b/'prepared.json').read_bytes());out=[]
for item in plan['arrays']:
 root=b/item['key'];v={'key':item['key'],'name':item['job_name']}
 for n in ('root-submission-intent.json','root-submission.json'):
  p=root/n
  if p.exists():v[n]=json.loads(p.read_bytes())
 for cmd,key in ((['squeue','-h','-u','mou176','-n',item['job_name'],'-o','%i|%j|%T|%R'],'active'),(['sacct','-X','-n','-P','-u','mou176','--name='+item['job_name'],'--starttime=2026-10-22T00:00:00','--format=JobIDRaw,JobName,State,Submit,ExitCode'],'history')):
  p=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=15);assert p.returncode==0,p.stderr;v[key]=p.stdout.strip()
 out.append(v)
print(json.dumps(dict(time=time.time(),arrays=out)))
