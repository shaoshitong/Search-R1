"""Small read-only three-branch saved/replay/intent audit; never torch/model IO."""
import json,pathlib,time,hashlib,subprocess
A=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/multilayer-four-epoch-v1')
EXPECTED={'layer21':'743126','layer24':'743127','layer27':'743128'}
def read(p):return json.loads(p.read_text())
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
 return h.hexdigest()
def fp(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
r=dict(observed_epoch=time.time(),variants={},torch_imported=False,checkpoint_tensor_loaded=False)
for name,job in EXPECTED.items():
 b=A/name;config=read(b/'config.json');own=dict(job_id=job,config_sha256=sha(b/'config.json'))
 assert (b/'job.id').read_text().strip()==job
 assert config['block_layer']==int(name[5:]) and config['epochs']==4 and config['update_budget']==512
 for n in ('progress.json','update-intent.json','restore-acceptance.json','failure.json','interrupted.json'):
  if (b/n).exists():
   v=read(b/n);own[n]={k:v[k] for k in ('status','cursor','steps','epoch','within_epoch_update','next_cursor','phase','loss','message','epoch_time') if k in v}
 p=b/'next-update-replay-cursor000000.json'
 if p.exists():
  proof=read(p);assert proof['passed'] is True and proof['expected']==proof['actual'] and proof['steps']==1 and proof['source_cursor']==0
  own['first_actual_replay']=dict(path=str(p),sha256=sha(p),passed=True,source_cursor=0,steps=1,fullstate_hashes=proof['actual'])
 p=b/'latest.json'
 if p.exists():
  v=read(p);cp=pathlib.Path(v['path']);assert v['cursor']==v['steps'] and cp.is_file() and cp.stat().st_size==v['bytes'] and v['config_sha256']==own['config_sha256']
  own['latest_saved_fullstate']=dict(path=str(cp),actual_bytes=cp.stat().st_size,sha256_from_GPU_save_independent_reload=v['sha256'],cursor=v['cursor'],steps=v['steps'],phase=v['phase'],accepted_hashes=v['accepted_hashes'],receipt=fp(p))
 for epoch in range(1,5):
  p=b/('evaluation-epoch-%04d.json'%epoch)
  if p.exists():
   v=read(p);assert v['config_sha256']==own['config_sha256'] and v['cursor']==128*epoch
   own['evaluation_epoch_%d'%epoch]=dict(path=str(p),sha256=sha(p),cursor=v['cursor'],epoch=v['epoch'],checkpoint=v['checkpoint'],metrics=v['metrics'])
 r['variants'][name]=own
print(json.dumps(r))
