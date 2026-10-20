import pathlib,json,time
B=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019');A=B/'algorithm-improvement-20261021';base=A/'peer-newseed8192-g10-batched-replay-20261021';v=json.loads((base/'preparation.json').read_text());rows=[json.loads(pathlib.Path(x['path']).read_text()) for x in v['selections']];refs=[(x['question_plan'],x['source_ingest']) for x in rows];assert all(x==refs[0] for x in refs);r={'observed_epoch':time.time(),'identity_plan':refs[0][0],'raw_complete':refs[0][1],'capture_base':str(base),'directories':[str(x) for x in base.iterdir() if x.is_dir()],'shards':[]}
for i in range(4):
 root=base/('shard-%02d'%i)/'capture';s={'shard':i,'capture':str(root),'complete':(root/'complete.json').is_file()}
 for file in ['progress.json','failure.json']:
  p=root/file
  if p.is_file():
   try:own=json.loads(p.read_text());s[file]={k:own[k] for k in ['status','trajectories','expected_trajectories','elapsed_seconds','error','error_type'] if k in own}
   except OSError as error:s[file]={'read_failed':str(error)}
 r['shards'].append(s)
print(json.dumps(r))
