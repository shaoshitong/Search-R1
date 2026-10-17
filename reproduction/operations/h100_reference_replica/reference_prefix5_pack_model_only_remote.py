"""Pack identical EMA tensors, excluding raw/Adam/RNG never imported online."""
from pathlib import Path
import importlib.util,subprocess,json,os
from reference_prefix5_copy_assets import SOURCE
spec=importlib.util.spec_from_file_location('reference_route','/Users/shaoshitong/Documents/Search-G1/actorhidden-prefix5-prune-aware-resume-20261017/reproduction/operations/a800_actorhidden_prefix5_control.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
prefix=SOURCE.split('out=sys.stdout.buffer;')[0]
inner=prefix+r'''
import torch,tarfile,time,io
torch.set_num_threads(1)
root=pathlib.Path('/data/shaoshitong/reference-prefix5-assets-transfer-model-only-20261018');root.mkdir(exist_ok=True);p=root/'pack-complete.json'
if p.exists():print(p.read_text());raise SystemExit
sys.path.insert(0,str(b/'repo'))
from reproduction.prefix_selection200.prefix5_model import warmup_model_weights
source=torch.load(c['probe_source_checkpoint'],map_location='cpu',weights_only=False)
model,buffers,extraction=warmup_model_weights(source,epoch=5,model_kind='ema',source_run=c['probe_source_run'])
keys=['role','binding','epoch','completed_epoch','next_update_in_epoch','counters','actual_adam_steps_per_rank','ema','actor_updates','epoch_metrics','epoch_metrics_sha256']
export={k:source[k]for k in keys};export['ema']={k:source['ema'][k]for k in ['model','nonpersistent_buffers','updates','decay']}
ep=root/'warmup-epoch5-EMA-model-only.pt';torch.save(export,ep)
def digest(p):
 h=hashlib.sha256()
 with pathlib.Path(p).open('rb')as f:
  for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
 return h.hexdigest()
loaded=torch.load(ep,map_location='cpu',weights_only=False);loaded_model,loaded_buffers,_=warmup_model_weights(loaded,epoch=5,model_kind='ema',source_run=c['probe_source_run'])
th={}
for prefix,a,z in [('model',model,loaded_model),('buffers',buffers,loaded_buffers)]:
 for k,v in a.items():
  assert torch.equal(v,z[k]);th[prefix+'.'+k]=hashlib.sha256(v.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()
original_sha=digest(c['probe_source_checkpoint']);assert original_sha=='b5c81ae76c12f677a0663d6e6b0f0482ca63fde0507c2a018aa6034282105594'
for r in items:
 if r['role']=='probe_source_checkpoint':r.update(source=str(ep),bytes=ep.stat().st_size,sha256=digest(ep),original_full_checkpoint_sha256=original_sha,transport_model_only=True,model_and_buffer_sha256=th)
out=root/'reference-assets-model-only.tar.gz';proof=[]
with tarfile.open(out,'w:gz',compresslevel=1)as tar:
 for r in items:
  path=pathlib.Path(r['source']);before=path.stat();h=hashlib.sha256()
  class Reader:
   def __init__(self,f):self.f=f
   def read(self,n):
    raw=self.f.read(n);h.update(raw);return raw
  with path.open('rb')as f:tar.addfile(tar.gettarinfo(str(path),arcname=r['filename']),Reader(f))
  after=path.stat();assert(before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns);assert r['sha256']in(None,h.hexdigest());proof.append(dict(r,actual_source_sha256=h.hexdigest()))
 raw=(json.dumps({'source_proof':proof})+'\n').encode();info=tarfile.TarInfo('source-full-sha-proof.json');info.size=len(raw);tar.addfile(info,io.BytesIO(raw))
v={'archive':str(out),'bytes':out.stat().st_size,'sha256':digest(out),'files':len(proof),'model_via_Mac':False,'original_checkpoint_sha256':original_sha,'EMA_tensors_bitwise_equal':True,'export_bytes':ep.stat().st_size,'finished_unix':time.time()}
with p.open('x')as f:json.dump(v,f)
print(json.dumps(v))
'''
code='import subprocess,os\nr=subprocess.run(["/data/search-g1/env/bin/python","-c",'+repr(inner)+'],env=dict(os.environ,CUDA_VISIBLE_DEVICES=""),text=True,capture_output=True,timeout=150);assert r.returncode==0,r.stderr;print(r.stdout)'
r=subprocess.run(m.route(code),capture_output=True,timeout=170);p=Path('work/prefix5-reference-model-only-pack.json');p.write_text(json.dumps({'returncode':r.returncode,'stdout':r.stdout.decode(),'stderr':r.stderr.decode()},indent=2)+'\n');assert r.returncode==0,r.stderr;v=json.loads(r.stdout);p=Path('work/prefix5-reference-asset-pack-summary.json');p.write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v))
