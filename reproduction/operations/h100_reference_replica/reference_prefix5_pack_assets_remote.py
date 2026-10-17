from pathlib import Path
import importlib.util,subprocess,json,datetime
from reference_prefix5_copy_assets import SOURCE
spec=importlib.util.spec_from_file_location('reference_route','/Users/shaoshitong/Documents/Search-G1/actorhidden-prefix5-prune-aware-resume-20261017/reproduction/operations/a800_actorhidden_prefix5_control.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
prefix=SOURCE.split('out=sys.stdout.buffer;')[0]
code=prefix+r'''
import tarfile,time
b=pathlib.Path('/data/shaoshitong/reference-prefix5-assets-transfer-20261018');b.mkdir(exist_ok=True);p=b/'pack-complete.json'
if p.exists():print(p.read_text());raise SystemExit
out=b/'reference-assets.tar.gz';proof=[]
with tarfile.open(out,'w:gz',compresslevel=1)as tar:
 for r in items:
  path=pathlib.Path(r['source']);before=path.stat();h=hashlib.sha256()
  class Reader:
   def __init__(self,f):self.f=f
   def read(self,n):
    raw=self.f.read(n);h.update(raw);return raw
  info=tar.gettarinfo(str(path),arcname=r['filename'])
  with path.open('rb')as f:tar.addfile(info,Reader(f))
  after=path.stat();assert(before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
  assert r['sha256']in(None,h.hexdigest());proof.append(dict(r,actual_source_sha256=h.hexdigest()))
 raw=(json.dumps({'source_proof':proof})+'\n').encode();import io
 info=tarfile.TarInfo('source-full-sha-proof.json');info.size=len(raw);tar.addfile(info,io.BytesIO(raw))
h=hashlib.sha256()
with out.open('rb')as f:
 for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk)
v={'archive':str(out),'bytes':out.stat().st_size,'sha256':h.hexdigest(),'files':len(proof),'model_via_Mac':False,'finished_unix':time.time()}
with p.open('x')as f:json.dump(v,f)
print(json.dumps(v))
'''
r=subprocess.run(m.route(code),capture_output=True,timeout=180);p=Path('work/prefix5-reference-asset-pack.json');p.write_text(json.dumps({'returncode':r.returncode,'stdout':r.stdout.decode(),'stderr':r.stderr.decode()},indent=2)+'\n');assert r.returncode==0,r.stderr;v=json.loads(r.stdout);p=Path('work/prefix5-reference-asset-pack-summary.json');p.write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v))
