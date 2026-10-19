"""Private repo metadata only, using an already-authorized remote task token in RAM."""
import json,time,shlex,subprocess,importlib.util
from pathlib import Path
D=Path('outputs/grounding-eval-20261019')
code=r'''import json,time,pathlib
out=dict(checked_epoch=time.time(),repo='Sutoonq/TMP_MODEL',read_only=True,weight_files_downloaded=0)
try:
 from huggingface_hub import HfApi
 p=pathlib.Path('/mnt/lustre/shaoshitong/hf-checkpoint-upload-20261005/task-token')
 if not p.is_file():raise FileNotFoundError('Authorized task token file unavailable')
 token=p.read_text().strip();assert token
 api=HfApi(token=token)
 info=api.repo_info('Sutoonq/TMP_MODEL',repo_type='model',files_metadata=True,timeout=20)
 assert info.private is True,'Expected repository is no longer private'
 files=[]
 for f in info.siblings:
  l=getattr(f,'lfs',None)
  def field(obj,key):return obj.get(key)if isinstance(obj,dict)else getattr(obj,key,None)
  files.append(dict(path=f.rfilename,size=getattr(f,'size',None),git_blob_id=getattr(f,'blob_id',None),lfs_sha256=field(l,'sha256')if l else None,lfs_size=field(l,'size')if l else None))
 out.update(ok=True,private=info.private,revision=info.sha,file_count=len(files),files=files)
except Exception as e:
 out.update(ok=False,error_type=type(e).__name__,error='Metadata probe unavailable; no credential or request headers retained',http_status=getattr(getattr(e,'response',None),'status_code',None))
print(json.dumps(out))'''
spec=importlib.util.spec_from_file_location('grounding_controller','/Users/shaoshitong/Documents/remote_codex_control/control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl)
r=subprocess.run(ctl.route('mgmt-host','python3 -B -c '+shlex.quote(code)),capture_output=True,text=True,timeout=50)
if r.returncode:
 value=dict(ok=False,transport_exit=r.returncode,error='mgmt-host existing route failed',stderr_tail=r.stderr[-200:])
else:value=json.loads(r.stdout)
(D/'hf-private-metadata-20261019.json').write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps({k:v for k,v in value.items()if k!='files'}))
