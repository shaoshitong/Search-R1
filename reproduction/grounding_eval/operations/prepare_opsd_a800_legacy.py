"""Only deploy tiny preserved Python/config assets, never model/data or inference."""
import sys,subprocess,json,pathlib,hashlib,io,tarfile,time,shlex
sys.path.insert(0,'/Users/shaoshitong/Documents/remote_codex_control');import control
src=r'''
import pathlib,tarfile,io,hashlib,json,sys
b=pathlib.Path('/mnt/data/shaoshitong/grounding-eval-paper-strict-v1-20261019');r=b/'runtime/opsd';files=[]
for p in sorted(r.rglob('*')):
 if p.is_file() and (p.suffix=='.py' or str(p.relative_to(r))=='verl/version/version'):
  assert not p.is_symlink();files.append(('repo/'+str(p.relative_to(r)),p))
files.append(('source-config.json',b/'inputs/opsd/source-config.json'))
assert sum(p.stat().st_size for _,p in files)<16*1024*1024
buf=io.BytesIO();manifest={'origin':str(r),'source_config':str(b/'inputs/opsd/source-config.json'),'source_commit':'fcecbea5d939','files':[]}
with tarfile.open(fileobj=buf,mode='w') as tar:
 for name,p in files:
  a=p.stat();v=p.read_bytes();z=p.stat();assert (a.st_size,a.st_mtime_ns)==(z.st_size,z.st_mtime_ns)
  info=tarfile.TarInfo(name);info.size=len(v);info.mode=0o600;tar.addfile(info,io.BytesIO(v));manifest['files'].append({'path':name,'bytes':len(v),'sha256':hashlib.sha256(v).hexdigest()})
 v=json.dumps(manifest,sort_keys=True).encode();info=tarfile.TarInfo('source-manifest.json');info.size=len(v);info.mode=0o600;tar.addfile(info,io.BytesIO(v))
sys.stdout.buffer.write(buf.getvalue())
'''
r=subprocess.run(control.route('h100-10','/usr/bin/python3 -c '+shlex.quote(src)),capture_output=True,timeout=80)
if r.returncode:raise RuntimeError('Sourcecode read failed without modifying remote')
blob=r.stdout;assert len(blob)<16*1024*1024
with tarfile.open(fileobj=io.BytesIO(blob)) as t:manifest=json.load(t.extractfile('source-manifest.json'))
package_sha=hashlib.sha256(blob).hexdigest();target='/data/shaoshitong/grounding-eval-20261019/runtime-opsd-legacy-'+package_sha[:12]
deploy=r'''
import sys,hashlib,tarfile,io,pathlib,json,os
payload=sys.stdin.buffer.read(16*1024*1024+1);assert len(payload)<=16*1024*1024;assert hashlib.sha256(payload).hexdigest()==EXPECTED
root=pathlib.Path(TARGET);root.mkdir(parents=True,exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(payload)) as tar:
 manifest=json.load(tar.extractfile('source-manifest.json'));expected={x['path']:x for x in manifest['files']}
 expected['source-manifest.json']=None
 for m in tar.getmembers():
  assert m.isfile() and m.name in expected and not pathlib.PurePosixPath(m.name).is_absolute() and '..' not in pathlib.PurePosixPath(m.name).parts
  p=root/m.name;p.parent.mkdir(parents=True,exist_ok=True);v=tar.extractfile(m).read()
  if expected[m.name]:assert len(v)==expected[m.name]['bytes'] and hashlib.sha256(v).hexdigest()==expected[m.name]['sha256']
  if p.exists():assert p.read_bytes()==v
  else:
   with p.open('xb') as f:f.write(v);f.flush();os.fsync(f.fileno())
 for name,x in expected.items():
  if x:assert hashlib.sha256((root/name).read_bytes()).hexdigest()==x['sha256']
print(json.dumps({'root':str(root),'package_sha256':EXPECTED,'files':len(manifest['files']),'code_config_bytes':sum(x['bytes'] for x in manifest['files']),'all_target_files_readback_match':True,'weights_or_data_transferred':False,'gpu_initialized':False}))
'''.replace('EXPECTED',repr(package_sha)).replace('TARGET',repr(target))
r=subprocess.run(control.route('baai-a800','/usr/bin/python3 -c '+shlex.quote(deploy)),input=blob,capture_output=True,timeout=80)
if r.returncode:raise RuntimeError('Independent code staging failed; source preserved')
receipt=json.loads(r.stdout);audit={'at':time.time(),'receipt':receipt,'source_manifest':manifest,'source_or_training_modified':False}
p=pathlib.Path('outputs/grounding-eval-20261019/opsd-a800-legacy-code-receipt.json');p.write_text(json.dumps(audit,indent=2));print(json.dumps(receipt))
