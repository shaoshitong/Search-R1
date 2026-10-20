import pathlib,json,subprocess,time
base=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/g10-pilot128-qualification-v1')
result=[]
for name in ('random','per-question-balanced'):
 b=base/name;j=(b/'job.id').read_text().strip();assert j.isdigit()
 p=subprocess.run(['squeue','--start','-h','-j',j,'-o','%i|%j|%T|%P|%S|%R'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=20);assert p.returncode==0,p.stderr
 result.append(dict(variant=name,job_id=j,estimate=p.stdout.strip()))
print(json.dumps(dict(time=time.time(),records=result)))
