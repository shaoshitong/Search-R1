import base64,hashlib,json,os,sys,time
from pathlib import Path
x=json.load(sys.stdin);root=Path(x['root']);assert root==Path('/data/user/jxu957/experiments/cosearch-continue300-x203-20261008/runtime/cosearch-instruct200-common-em-20261011/preparation/cp200-tensor-retirement/19ec0e1bc6224d8e85bbcba0b8ba9eb8') and root.resolve()==root and root.stat().st_uid==os.getuid()==21930
code=Path(x['code_root'])/'retire_old_cosearch200.py';assert hashlib.sha256(code.read_bytes()).hexdigest()==x['operator_sha256']
records={}
for name,row in x['files'].items():
 assert name in ('hk-validation-d45-imported.json','reader-guards-before-apply.json');p=root/name;raw=base64.b64decode(row['data_b64']);assert len(raw)==row['bytes'] and hashlib.sha256(raw).hexdigest()==row['sha256']
 if os.path.lexists(p):assert p.resolve()==p and p.is_file() and p.read_bytes()==raw
 else:
  fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
  with os.fdopen(fd,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
 records[name]={'bytes':len(raw),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}
fd=os.open(root,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW);os.fsync(fd);os.close(fd)
for name in ('deletion-claim.json','actions.jsonl','deletion-result.json','deletion-failure.json','complete.json'):assert not os.path.lexists(root/name)
print(json.dumps({'checked_unix':time.time(),'operator_sha256':x['operator_sha256'],'files':records,'deletion_not_started':True,'uid':os.getuid()},sort_keys=True))
