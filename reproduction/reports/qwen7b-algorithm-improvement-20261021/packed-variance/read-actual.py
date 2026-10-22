import pathlib,json,subprocess,time
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/online-packed-M3to10-P20-variance-8192-20261022-v1')
receipt=json.loads((b/'root-submission.json').read_bytes());job=receipt['job_id'];assert job=='749515'
x=dict(time=time.time(),job=job)
for key,args in [('queue',['squeue','--start','-h','-j',job,'-o','%i|%j|%P|%T|%S|%R']),('accounting',['sacct','-X','-n','-P','-j',job,'--format=JobIDRaw,State,ExitCode,Elapsed,NodeList'])]:
 r=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=15);assert r.returncode==0,r.stderr;x[key]=r.stdout.strip()
for name in ('controller.json','progress-rank-0.json','progress-rank-1.json','evaluation-progress-rank-0.json','evaluation-progress-rank-1.json','latest.json','complete.json'):
 p=b/name
 if p.exists():x[name]=json.loads(p.read_bytes())
for pattern in ('failure-rank-*.json','evaluation-epoch-*.json','coverage-epoch-*.json'):
 for p in b.glob(pattern):x[p.name]=json.loads(p.read_bytes())
p=b/('train-'+job+'.log')
if p.exists():
 with p.open('rb')as f:f.seek(max(0,p.stat().st_size-1800));x['log_tail']=f.read().decode(errors='replace')
print(json.dumps(x))
