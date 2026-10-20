import sys,json,base64,tarfile,io,hashlib,pathlib,os,time
p=json.load(sys.stdin);root=pathlib.Path('/data/user/jxu957/experiments/dynrpo-nq-grounding-20261020');assert not root.exists();source=root/'source'/p['commit'];source.mkdir(parents=True)
raw=base64.b64decode(p['archive']);assert hashlib.sha256(raw).hexdigest()==p['archive_sha256']
with tarfile.open(fileobj=io.BytesIO(raw)) as t:
 for m in t:
  path=pathlib.PurePosixPath(m.name);assert not path.is_absolute() and '..' not in path.parts
  if m.isdir():(source/m.name).mkdir(parents=True,exist_ok=True);continue
  assert m.isfile();f=source/m.name;f.parent.mkdir(parents=True,exist_ok=True);data=t.extractfile(m).read();assert hashlib.sha256(data).hexdigest()==p['files'][m.name]['sha256'];f.write_bytes(data);os.chmod(f,m.mode)
for name,v in p['files'].items():
 f=source/name;assert f.stat().st_size==v['bytes'] and hashlib.sha256(f.read_bytes()).hexdigest()==v['sha256']
receipt=dict(source_commit=p['commit'],source_root=str(source),files=p['files'],archive_sha256=p['archive_sha256'],all_files_SHA_verified=True,checked_unix=time.time())
(source/'source-publication.json').write_text(json.dumps(receipt));(root/'control').mkdir()
print(json.dumps(dict(status='SOURCE_DEPLOYED_FULL_SHA_PASS',commit=p['commit'],source=str(source),files=len(p['files']),bytes=sum(v['bytes'] for v in p['files'].values()),training_modified=False)))
