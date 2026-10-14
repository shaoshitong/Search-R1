import pathlib,json,subprocess,datetime
b=pathlib.Path('/mnt/data/shaoshitong/search-g1-final200-bs256-20261014-v2');o={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'job':3395,'native':subprocess.run(['scontrol','show','job','-o','3395'],capture_output=True,text=True).stdout,'files':{}}
for n in ['logs/final-pipeline-status.json','logs/final-native-allocation.json','logs/cleanup-status.json','journal/loaded-workers-3395.json','journal/final-result.json']:
 p=b/n
 if p.exists():
  v=json.loads(p.read_text());o['files'][n]=({k:v.get(k) for k in ['qa_count','weighted_em','weighted_f1','source_metrics']} if n.endswith('final-result.json') else v)
ps=sorted((b/'journal').glob('batch-????????-????????.json'));o['batch_count']=len(ps);o['completed_questions']=int(ps[-1].stem.split('-')[-1]) if ps else 0;o['batch_files']=[p.name for p in ps[-2:]]
for n in ['logs/native-3395.log','logs/final-retriever.log','logs/final-evaluator.log']:
 p=b/n
 if p.exists():
  with p.open('rb') as f:f.seek(max(0,p.stat().st_size-3500));o[n]={'bytes':p.stat().st_size,'tail':f.read().decode(errors='replace')}
o['gpu']=subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,memory.total','--format=csv,noheader'],capture_output=True,text=True).stdout
print(json.dumps(o))
