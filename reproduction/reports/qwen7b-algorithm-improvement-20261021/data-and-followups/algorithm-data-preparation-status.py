import pathlib,json,time
A=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021');out=dict(time=time.time(),jobs={})
for name in ['validation-g10-20261021','training-extraG5-128-20261021']:
 b=A/name;d={}
 for n in ['launch.json','controller.json','result.json','failure.json']:
  p=b/n
  if p.exists():
   a=json.loads(p.read_text())
   if n=='result.json' and a.get('result'):a['result']={k:v for k,v in a['result'].items() if k in ['status','base','source_commit','questions','trajectories','source_qid_plan_sha256','config','input_manifest','sampling_script','isolation','final_exclusion_plan']}
   d[n]=a
 launch=d.get('controller.json') or d.get('launch.json')
 if launch:
  try:d['process_identity_live']=pathlib.Path('/proc/%d/stat'%launch['pid']).read_text().rsplit(')',1)[1].split()[19]==str(launch['start_ticks'])
  except FileNotFoundError:d['process_identity_live']=False
 p=b/'prepare.log'
 if p.exists():
  d['log_bytes']=p.stat().st_size
  with p.open('rb') as f:f.seek(max(0,p.stat().st_size-2600));d['log_tail']=f.read().decode(errors='replace')
 out['jobs'][name]=d
print(json.dumps(out))
