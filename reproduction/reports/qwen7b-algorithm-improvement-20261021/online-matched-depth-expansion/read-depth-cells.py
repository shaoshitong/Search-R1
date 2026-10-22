import json,pathlib,subprocess,time
S=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/online-mixed-layer-prefix-20261022-2gpu-v1')
out={'time':time.time(),'cells':[]}
for layer in(15,18,24):
 b=S/('layer%d-prefix20'%layer);x={'layer':layer,'prefix':20}
 for n in('root-submission.json','controller.json','latest.json','progress-rank-0.json','progress-rank-1.json','complete.json','coverage-epoch-01.json','coverage-epoch-02.json','coverage-epoch-03.json'):
  p=b/n
  if p.exists():
   try:x[n]=json.loads(p.read_bytes())
   except Exception as e:x[n]={'read_error':str(e)}
 if 'root-submission.json'in x:
  job=x['root-submission.json']['job_id']
  for name,args in [('queue',['squeue','-h','-j',job,'-o','%i|%T|%R|%S|%P|%q']),('accounting',['sacct','-X','-n','-P','-j',job,'--format=JobIDRaw,State,ExitCode,Elapsed,NodeList'])]:
   z=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=15);x[name]=z.stdout.strip() if not z.returncode else {'error':z.stderr[-500:]}
 for p in sorted(b.glob('evaluation-epoch-*.json')):
  z=json.loads(p.read_bytes());x[p.name]={'epoch':z.get('epoch'),'cursor':z.get('cursor'),'metrics':z.get('metrics'),'seconds':z.get('seconds')}
 logs=sorted(b.glob('train-*.log'),key=lambda p:p.stat().st_mtime)
 if logs:
  with logs[-1].open('rb')as f:f.seek(max(0,logs[-1].stat().st_size-2000));x['log_tail']=f.read().decode(errors='replace')
 out['cells'].append(x)
print(json.dumps(out))
