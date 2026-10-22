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
  out={'time':data['time'],'variants':[]}
  for x in data['variants']:
   v={k:x[k]for k in ('job','layer','prefix','sacct','complete.json','latest.json')if k in x}
   for k,z in x.items():
    if k.startswith('progress-rank'):v[k]={f:z[f]for f in ('cursor','epoch','time','native_gradient_L2','head_gradient_L2','gradient_accumulation_microbatches','world_size')}
    if k.startswith('failure'):v[k]=z
    if k.startswith('evaluation-epoch'):
     v[k]={pool:{m:{f:a[f]for f in ('balanced_accuracy','binary_Mixed_BCE','Mixed_recall','nonMixed_false_positive_rate','selected512')}for m,a in ms.items()}for pool,ms in z['metrics'].items()}
   out['variants'].append(v)
 elif key in ('main','uniform'):
  out={k:metric(v) for k,v in data.items() if k in ('time','job_id','squeue','sacct','checkpoint','failure.json','complete.json','phase-transition.json')}
  if 'progress.json'in data:
   z=data['progress.json'];out['progress']={f:z[f]for f in ('cursor','new_updates','time','training_phase','native_steps','head_steps','native_frozen','native_lr','head_lr','epoch_unique_question_coverage','pre_Adam_OOM_retries')if f in z}
  for k,v in data.items():
   if k.startswith(('independent','training'))and isinstance(v,dict)and'metrics'in v:
    out[k]={}
    for m,pools in v['metrics'].items():
     out[k][m]={}
     for pool,z in pools.items():
      met=z.get('pool',{});out[k][m][pool]={f:met[f]for f in ('questions','mixed_questions','mixed_balanced_accuracy','mixed_recall','nonmixed_false_positive_rate')if f in met};out[k][m][pool].update(BCE=z.get('binaryMixedBCE'),selected512=z.get('selected512',{}).get('mixed_questions'))
 else:out=data
 print(json.dumps({'kind':key,'observation':out}))
