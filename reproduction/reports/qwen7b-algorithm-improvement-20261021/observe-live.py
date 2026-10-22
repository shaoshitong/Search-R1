import json,pathlib,subprocess,sys,time

repo=pathlib.Path(__file__).resolve().parents[3]
root=repo.parents[1]
report=pathlib.Path(__file__).resolve().parent
scripts={
 'main':'mixed-direct-e1-unique-g5/read-current.py',
 'pilot':'online-layer-prefix/read-suite-two.py',
 'uniform':'uniform-g5-control/read-current.py',
 'final':'frozen-final4096-preparation/read-generation-progress-blind.py',
}
for key in sys.argv[1:]:
 script=report/scripts[key]
 r=subprocess.run([sys.executable,str(root/'outputs/qwen7b-warmup-20261019/balanced-control.py'),str(script),'--timeout','30'],cwd=root,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=45)
 rows=[x for x in r.stdout.splitlines() if x.strip().startswith('{')]
 if r.returncode or not rows:raise RuntimeError(key+' remote observation failed; no current numbers accepted')
 data=json.loads(rows[-1]);dest=report/'live-observations';dest.mkdir(exist_ok=True)
 (dest/(key+'-latest.json')).write_text(json.dumps(data,sort_keys=True)+'\n')
 def metric(v):
  if not isinstance(v,dict):return v
  return {k:(metric(x) if isinstance(x,dict) else x) for k,x in v.items() if k not in ('classwise','k_histogram','timing','classwise_binaryMixedBCE','log_tail','slurm_identity')}
 if key=='pilot':
  out={'time':data['time'],'variants':[{k:metric(v) for k,v in x.items() if k in ('job','layer','prefix','sacct','progress-rank-0.json','progress-rank-1.json','failure.json','complete.json','latest.json') or k.startswith(('evaluation','epoch','failure'))}for x in data['variants']]}
 elif key in ('main','uniform'):
  out={k:metric(v) for k,v in data.items() if k in ('time','job_id','squeue','sacct','checkpoint','progress.json','failure.json','complete.json','phase-transition.json') or k.startswith(('independent','training'))}
 else:out=data
 print(json.dumps({'kind':key,'observation':out}))
