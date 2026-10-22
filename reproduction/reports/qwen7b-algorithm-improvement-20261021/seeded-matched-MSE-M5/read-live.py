import pathlib,json,subprocess,time
base=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/online-packed-matched-MSE-M5-seed19137-plan2026102302-L15P80-8192-20261023-v1/mse-M5')
def run(args):
 p=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=15)
 return dict(returncode=p.returncode,stdout=p.stdout.strip(),stderr=p.stderr.strip())
out=dict(time=time.time(),job_id='750001',squeue=run(['squeue','-h','-j','750001','-o','%i|%T|%R|%N|%M']),sacct=run(['sacct','-n','-X','-j','750001','--format=JobIDRaw,State,ExitCode,Elapsed,NodeList','-P']),metadata_only=True,final4096_read=False)
for name in ('controller.json','progress-rank-0.json','progress-rank-1.json','latest.json','complete.json'):
 p=base/name
 if p.exists():out[name]=json.loads(p.read_bytes())
for p in base.glob('failure-rank-*.json'):out[p.name]=json.loads(p.read_bytes())
print(json.dumps(out))
