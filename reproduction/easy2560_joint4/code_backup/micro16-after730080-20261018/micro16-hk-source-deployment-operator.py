import sys,json,base64,tarfile,io,hashlib,time,os
from pathlib import Path
p=json.load(sys.stdin);parent=Path('/backup/shaoshitong/easy2560-corollout-hpc3-4gpu-20261017/source');parent.mkdir(parents=True,exist_ok=True);root=parent/p['commit'];seen=set()
if root.exists():
 proof=json.loads((root/'source-publication.json').read_text());assert proof['source_commit']==p['commit'] and proof['files']==p['manifest']
else:
 stage=parent/('.partial-'+p['commit']);stage.mkdir(exist_ok=False)
 with tarfile.open(fileobj=io.BytesIO(base64.b64decode(p['tar_gz'])),mode='r:gz') as tar:
  for m in tar:
   if m.isdir():continue
   if not m.isfile() or m.name not in p['manifest'] or m.name in seen:raise ValueError('HK_SOURCE_MEMBER_DIFFERS')
   seen.add(m.name);target=stage/m.name;target.parent.mkdir(parents=True,exist_ok=True);data=tar.extractfile(m).read();record=p['manifest'][m.name]
   assert len(data)==record['bytes'] and hashlib.sha256(data).hexdigest()==record['sha256'];target.write_bytes(data)
 assert seen==set(p['manifest'])
 proof=dict(source_commit=p['commit'],files=p['manifest'],fresh_source_all_files_sha_verified=True,checked_unix=time.time())
 with (stage/'source-publication.json').open('x') as f:json.dump(proof,f,indent=2);f.flush();os.fsync(f.fileno())
 stage.rename(root)
assert all((root/n).stat().st_size==r['bytes'] and hashlib.sha256((root/n).read_bytes()).hexdigest()==r['sha256'] for n,r in p['manifest'].items())
print(json.dumps(dict(status='HKUST3_INDEPENDENT_PUBLISHED_SOURCE_SHA_PASS',source_commit=p['commit'],source_root=str(root),files=len(p['manifest']),bytes=sum(r['bytes'] for r in p['manifest'].values()),publication_sha256=hashlib.sha256((root/'source-publication.json').read_bytes()).hexdigest(),checked_unix=time.time())))
