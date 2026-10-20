import json,subprocess,time,pathlib
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019');out=dict(time=time.time())
for name,args in [('matched_queue',['squeue','--start','-h','-j','742868,742869,742870','-o','%i|%T|%P|%S|%R']),('baseline_usage',['sstat','-n','-P','-j','742276.0','--format=JobID,AveCPU,MaxRSS,AveDiskRead,AveDiskWrite'])]:
 p=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=15);out[name]=dict(code=p.returncode,stdout=p.stdout,stderr=p.stderr)
p=b/'balanced-rank1-from-epoch1-prefix20-20261020/probe'
for name in ['metrics-latest.json','latest.json','phase.json']:
 if (p/name).exists():
  a=json.loads((p/name).read_text());out[name]={k:v for k,v in a.items() if k!='metrics'}
out['selection_e3_exists']=(p/'predictions/epoch-0003-unseen2048.jsonl.gz').exists()
print(json.dumps(out))
