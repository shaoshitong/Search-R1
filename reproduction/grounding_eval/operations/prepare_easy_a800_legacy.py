"""Deploy only historical source code plus a stripped identity configuration.

No weights/data/rollouts/full evaluation outputs cross Mac. The source's full
final-result is read in source RAM; only run/data SHA + IDs/source labels become
an inference identity config, without its predictions, metrics or rows.
"""
import sys,subprocess,json,pathlib,hashlib,io,tarfile,time,shlex
sys.path.insert(0,'/Users/shaoshitong/Documents/remote_codex_control');import control
src=r'''import pathlib,tarfile,io,hashlib,json,sys
b=pathlib.Path('/mnt/data/shaoshitong/grounding-eval-paper-strict-v1-20261019');r=b/'runtime/easy';files=[]
for p in sorted(r.rglob('*')):
 if p.is_file() and(p.suffix=='.py'or str(p.relative_to(r))=='verl/version/version'):
  assert not p.is_symlink();files.append(('repo/'+str(p.relative_to(r)),p.read_bytes(),str(p)))
for n in ['source-config.json','historical-final-binding.json']:
 p=b/'inputs/easy'/n;files.append((n,p.read_bytes(),str(p)))
f=b/'inputs/easy/historical-final-result.json';raw=f.read_bytes();original_sha=hashlib.sha256(raw).hexdigest();assert original_sha=='bfaf6cfcc4f13bc2cc8d500dd5231dde0fb419d61fca53d0fadb8879610ff3c5';v=json.loads(raw)
identity={k:v[k]for k in ['run_id','validation_sha256','question_ids','data_sources']};assert len(identity['question_ids'])==len(identity['data_sources'])==51713;assert identity['validation_sha256']=='55d515e116a4c6aa63b45282f9bc87202b030156be41be60420c38a9d3ff0ed1';identity.update(identity_source_path=str(f),identity_source_sha256=original_sha,identity_rows_or_predictions_included=False)
files.append(('identity-51713.json',json.dumps(identity,sort_keys=True).encode(),str(f)+'#run_id+data_sha+ids+sources-only'))
assert sum(len(v)for _,v,_ in files)<16*1024*1024
buf=io.BytesIO();binding=json.loads((b/'inputs/easy/historical-final-binding.json').read_text());manifest={'origin':str(r),'historical_final_source_commit':binding.get('source_commit'),'run_id':identity['run_id'],'identity_source_sha256':original_sha,'files':[]}
with tarfile.open(fileobj=buf,mode='w')as t:
 for name,raw,origin in files:
  item=tarfile.TarInfo(name);item.size=len(raw);item.mode=0o600;t.addfile(item,io.BytesIO(raw));manifest['files'].append({'path':name,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'origin':origin})
 raw=json.dumps(manifest,sort_keys=True).encode();item=tarfile.TarInfo('source-manifest.json');item.size=len(raw);item.mode=0o600;t.addfile(item,io.BytesIO(raw))
sys.stdout.buffer.write(buf.getvalue())'''
a=control.route('h100-10','/usr/bin/python3 -c '+shlex.quote(src));a[1:1]=['-o','ServerAliveInterval=5','-o','ServerAliveCountMax=6'];r=subprocess.run(a,capture_output=True,timeout=160);assert r.returncode==0,'Historical code/config extraction failed';blob=r.stdout;assert len(blob)<16*1024*1024
with tarfile.open(fileobj=io.BytesIO(blob))as t:manifest=json.load(t.extractfile('source-manifest.json'))
package_sha=hashlib.sha256(blob).hexdigest();target='/data/shaoshitong/grounding-eval-20261019/runtime-easy-legacy-'+package_sha[:12]
deploy=r'''import sys,json,pathlib,hashlib,tarfile,io,os
blob=sys.stdin.buffer.read(16*1024*1024+1);assert len(blob)<=16*1024*1024 and hashlib.sha256(blob).hexdigest()==EXPECTED;root=pathlib.Path(TARGET);root.mkdir(parents=True,exist_ok=True)
with tarfile.open(fileobj=io.BytesIO(blob))as t:
 m=json.load(t.extractfile('source-manifest.json'));expected={f['path']:f for f in m['files']};expected['source-manifest.json']=None
 for z in t.getmembers():
  n=pathlib.PurePosixPath(z.name);assert z.isfile()and z.name in expected and not n.is_absolute()and'..'not in n.parts;p=root/z.name;p.parent.mkdir(parents=True,exist_ok=True);raw=t.extractfile(z).read()
  if expected[z.name]:assert len(raw)==expected[z.name]['bytes']and hashlib.sha256(raw).hexdigest()==expected[z.name]['sha256']
  if p.exists():assert p.read_bytes()==raw
  else:
   with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
 for n,v in expected.items():
  if v:assert hashlib.sha256((root/n).read_bytes()).hexdigest()==v['sha256']
print(json.dumps({'root':str(root),'package_sha256':EXPECTED,'files':len(m['files']),'code_config_bytes':sum(v['bytes']for v in m['files']),'run_id':m['run_id'],'all_target_files_readback_match':True,'original_full_result_transferred':False,'weights_or_data_transferred':False,'gpu_initialized':False}))'''.replace('EXPECTED',repr(package_sha)).replace('TARGET',repr(target))
a=control.route('baai-a800','/usr/bin/python3 -c '+shlex.quote(deploy));a[1:1]=['-o','ServerAliveInterval=5','-o','ServerAliveCountMax=6'];r=subprocess.run(a,input=blob,capture_output=True,timeout=160);assert r.returncode==0,'Historical code/config deployment failed';receipt=json.loads(r.stdout);x={'at':time.time(),'receipt':receipt,'source_manifest':manifest,'source_or_training_modified':False};p=pathlib.Path('outputs/grounding-eval-20261019/easy-a800-legacy-code-receipt.json');p.write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(receipt))
