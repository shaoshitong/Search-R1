import pathlib,json,subprocess,time
S=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/online-mixed-layer-prefix-20261022-2gpu-v1');out={'time':time.time(),'experiments':{}}
for layer in (15,18):
 b=S/('layer%d-prefix80'%layer);cell={}
 for name in ['prefix80-ready.json','root-submission.json','controller.json','progress-rank-0.json','progress-rank-1.json','latest.json','complete.json']:
  if(b/name).exists():cell[name]=json.loads((b/name).read_bytes())
 for p in b.glob('failure-rank-*.json'):cell[p.name]=json.loads(p.read_bytes())
 for p in b.glob('evaluation-epoch-*.json'):
  e=json.loads(p.read_bytes());cell[p.name]={'cursor':e['cursor'],'epoch':e['epoch'],'metrics':{role:{state:{k:val[k]for k in ['balanced_accuracy','binary_Mixed_BCE','selected512','Mixed_recall','nonMixed_false_positive_rate']if k in val}for state,val in values.items()}for role,values in e['metrics'].items()}}
 if'root-submission.json'in cell:
  job=cell['root-submission.json']['job_id']
  for label,cmd in [('accounting',['sacct','-X','-n','-P','-j',job,'--format=JobIDRaw,State,ExitCode,Elapsed,NodeList']),('queue',['squeue','--start','-h','-j',job,'-o','%i|%j|%P|%T|%S|%R'])]:
   p=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=15);cell[label]={'returncode':p.returncode,'stdout':p.stdout.strip(),'stderr':p.stderr.strip()}
 out['experiments'][str(layer)]=cell
print(json.dumps(out))
