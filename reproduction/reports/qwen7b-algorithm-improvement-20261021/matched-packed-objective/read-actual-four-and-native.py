import pathlib,json,subprocess,time,gzip,hashlib,collections
A=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021')
out={'time':time.time(),'cells':{}}
def command(args):
 p=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=15)
 return {'returncode':p.returncode,'stdout':p.stdout.strip(),'stderr':p.stderr[-1000:]}
def load(p):return json.loads(p.read_bytes())
for cell in ('mse-Mrandom','mse-M5','mixed-bce-Mrandom','mixed-bce-M5'):
 d=A/'online-packed-matched-L15P80-8192-20261023-v1'/cell;c={}
 for n in ('root-submission.json','controller.json','progress-rank-0.json','progress-rank-1.json','latest.json','complete.json'):
  if(d/n).is_file():c[n]=load(d/n)
 c['failures']={p.name:load(p)for p in d.glob('failure*.json')}
 c['evaluation_files']=[p.name for p in d.glob('evaluation*.json')]
 c['epochs']={}
 for p in sorted(d.glob('evaluation-epoch-*.json')):
  e=load(p);c['epochs'][str(e['epoch'])]={'cursor':e['cursor'],'seconds':e.get('seconds'),'metrics':e['metrics']}
 if(d/'evaluation-epoch-00.json').is_file():
  c['E0_exact_score_and_view_SHA']={}
  for role in ('fixed_train2048_G5','fixed_validation2048_G5'):
   path=d/('epoch-00-'+role+'-predictions.jsonl.gz')
   if not path.is_file():continue
   with gzip.open(path,'rt')as f:rows=sorted((json.loads(line)for line in f),key=lambda x:x['qid'])
   assert len(rows)==2048
   raw=json.dumps(rows,sort_keys=True,separators=(',',':')).encode();c['E0_exact_score_and_view_SHA'][role]=hashlib.sha256(raw).hexdigest()
 if(d/'curve.jsonl').is_file():
  with(d/'curve.jsonl').open()as f:last=collections.deque(f,maxlen=6)
  keep=('time','epoch','cursor','actual_Adam_updates','EMA_updates','training_loss','native_gradient_L2','head_gradient_L2','actual_global_batch')
  c['last_six_real_updates']=[{k:v for k,v in json.loads(line).items()if k in keep}for line in last]
 logs=sorted(d.glob('train-*.log'))
 if logs:
  p=logs[-1];c['log_tail']=p.read_text(errors='replace')[-1800:]
 out['cells'][cell]=c
out['accounting']=command(['sacct','-X','-n','-P','-j','749856,749857,749858,749859,749782,749754','--format=JobIDRaw,State,ExitCode,Elapsed,NodeList'])
out['queue']=command(['squeue','--start','-h','-j','749856,749857,749858,749859,749782','-o','%i|%j|%P|%T|%S|%R'])
d=A/'online-mixed-layer-prefix-20261022-2gpu-v1/layer18-prefix80'
for p in sorted(d.glob('evaluation-epoch-*.json')):
 e=load(p);out[p.name]={'cursor':e.get('cursor'),'epoch':e.get('epoch'),'metrics':{role:{state:{k:v[k]for k in ('balanced_accuracy','binary_Mixed_BCE','selected512','Mixed_recall','nonMixed_false_positive_rate')if k in v}for state,v in values.items()}for role,values in e.get('metrics',{}).items()}}
print(json.dumps(out))
