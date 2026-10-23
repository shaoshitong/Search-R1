import os,json,time,pathlib,fcntl
root=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019');out=pathlib.Path('/data/user/jxu957/experiments/probe-feature-cleanup-20261023')
lock=(out/'inventory.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
start=time.time();found=[];protected=[];visited=0;last=0
for current,dirs,files in os.walk(str(root),followlinks=False):
 dirs[:]=[d for d in dirs if not (d.startswith(('repo','code-','env','venv','.git')) or d in ('__pycache__','node_modules','cache'))]
 visited+=1
 if 'complete.json' in files:
  path=pathlib.Path(current)/'complete.json'
  try:
   if path.stat().st_size<2000000:
    m=json.loads(path.read_text())
    if m.get('status') in ('LOSSLESS_NATIVE_FEATURES_COMPLETE','LOSSLESS_HF_REPLAY_FEATURES_COMPLETE'):
     for name in ('question.bin','delta.bin','policy.bin','prototype.bin','index.jsonl'):
      p=path.parent/name
      if p.is_file() and not p.is_symlink():
       st=p.stat();found.append(dict(path=str(p),bytes=st.st_size,allocated_bytes=st.st_blocks*512,inode=st.st_ino,mtime_ns=st.st_mtime_ns,manifest=str(path),status=m['status']))
  except (ValueError,OSError):pass
 for name in files:
  if name.endswith(('.pt','.safetensors')):
   p=pathlib.Path(current)/name
   if p.is_file():
    st=p.stat();protected.append(dict(path=str(p),bytes=st.st_size,inode=st.st_ino,mtime_ns=st.st_mtime_ns))
 if time.time()-last>5:
  (out/'inventory-progress.json').write_text(json.dumps(dict(time=time.time(),directories=visited,feature_files=len(found),feature_bytes=sum(x['bytes'] for x in found),weights_or_pt_preserved=len(protected))))
  last=time.time()
result=dict(status='FEATURE_PAYLOAD_INVENTORY_COMPLETE',root=str(root),time=time.time(),seconds=time.time()-start,visited_directories=visited,files=found,protected_weights_and_pt=protected)
(out/'inventory.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ('files','protected_weights_and_pt')}),flush=True)
