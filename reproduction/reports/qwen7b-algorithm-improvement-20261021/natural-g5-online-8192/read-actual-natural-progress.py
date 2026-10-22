import pathlib,json,subprocess,time
base=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/online-natural-g5-8192-20261022-v1')
rows=[]
for layer,job in((21,'749459'),(27,'749460')):
 root=base/('layer%d-prefix20'%layer);assert(root/'job.id').read_text().strip()==job
 own=dict(layer=layer,job_id=job,squeue=subprocess.check_output(['squeue','-h','-j',job,'-o','%i|%T|%j|%N|%R'],universal_newlines=True).strip().splitlines(),sacct=subprocess.check_output(['sacct','-n','-P','-j',job,'-o','JobIDRaw,State,ExitCode,Elapsed,NodeList'],universal_newlines=True).strip().splitlines(),files={})
 for name in('controller.json','progress-rank-0.json','progress-rank-1.json','failure-rank-0.json','failure-rank-1.json','latest.json','complete.json'):
  p=root/name
  if p.exists():own['files'][name]=json.loads(p.read_text())
 own['epochs_with_evaluations']=[int(p.stem.rsplit('-',1)[1])for p in sorted(root.glob('evaluation-epoch-*.json'))]
 rows.append(own)
print(json.dumps(dict(epoch=time.time(),same_original_jobs=True,variants=rows,final4096_read=False)))
