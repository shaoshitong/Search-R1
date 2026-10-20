import pathlib,json,subprocess,time
A=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/multilayer-four-epoch-v1')
r={'epoch':time.time(),'variants':{}}
for layer in (21,24,27):
 b=A/('layer'+str(layer));o={}
 for name in ('job.id','complete.json','failure.json','interrupted.json','progress.json','latest.json'):
  p=b/name
  if p.exists():
   v=json.loads(p.read_text()) if name.endswith('.json') else p.read_text().strip()
   if name=='complete.json':v={k:v[k] for k in ('status','cursor','steps','epochs','config_sha256') if k in v}
   o[name]=v
 o['epochs']={}
 for epoch in range(1,5):
  p=b/('evaluation-epoch-%04d.json'%epoch)
  if p.exists():
   v=json.loads(p.read_text());m={}
   for role,a in v['metrics'].items():m[role]={'pool':a['pool'],'selected512':a['selected512']}
   o['epochs'][str(epoch)]={'cursor':v['cursor'],'metrics':m}
 r['variants'][str(layer)]=o
r['squeue']=subprocess.run(['squeue','-h','-j','743126,743127,743128','-o','%i|%T|%R'],stdout=subprocess.PIPE,universal_newlines=True).stdout
r['sacct']=subprocess.run(['sacct','-X','-n','-P','-j','743126,743127,743128','--format=JobID,State,ExitCode,Elapsed'],stdout=subprocess.PIPE,universal_newlines=True).stdout
print(json.dumps(r))
