import pathlib,json,subprocess,time
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/online-packed-full169028-Z-seed19137-plan2026102302-P20-20261023-v1');o={'time':time.time()}
for name in ('root-submission.json','controller.json','progress-rank-0.json','progress-rank-1.json','latest.json','complete.json'):
 if(b/name).is_file():o[name]=json.loads((b/name).read_bytes())
o['failures']={p.name:json.loads(p.read_bytes())for p in b.glob('failure*.json')}
job=o['root-submission.json']['job_id']
for name,args in [('accounting',['sacct','-X','-n','-P','-j',job,'--format=JobIDRaw,State,ExitCode,Elapsed,NodeList']),('queue',['squeue','--start','-h','-j',job,'-o','%i|%j|%P|%T|%S|%R'])]:
 p=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=15);o[name]={'returncode':p.returncode,'stdout':p.stdout.strip(),'stderr':p.stderr[-1000:]}
print(json.dumps(o))
