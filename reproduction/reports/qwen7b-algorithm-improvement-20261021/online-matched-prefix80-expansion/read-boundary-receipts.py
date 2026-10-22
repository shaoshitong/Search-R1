import pathlib,json,time,subprocess
S=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/online-mixed-layer-prefix-20261022-2gpu-v1');out={'time':time.time(),'jobs':{}}
for l in(15,18):
 b=S/('layer%d-prefix80'%l);o={}
 for name in('latest.json','walltime-boundary-CPU-accepted.json','walltime-handoff-transaction.json','root-submission.json','progress-rank-0.json','progress-rank-1.json'):
  if(b/name).exists():o[name]=json.loads((b/name).read_bytes())
 out['jobs'][str(l)]=o
out['processes']=[]
for p in pathlib.Path('/proc').glob('[0-9]*'):
 try:
  cmd=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode()
  if 'accept_online_mixed_epoch_boundary.py'in cmd and ' -c 'not in cmd:out['processes'].append({'pid':p.name,'stat':(p/'stat').read_text().rsplit(')',1)[1].split()[0]})
 except(OSError,UnicodeError):pass
print(json.dumps(out))
