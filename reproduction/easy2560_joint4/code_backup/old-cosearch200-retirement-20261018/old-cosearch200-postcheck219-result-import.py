import base64,hashlib,json,os,socket,sys
from pathlib import Path
x=json.load(sys.stdin);site=x['site'];uid=14090 if site=='jxu' else 1002;assert os.getuid()==uid and socket.gethostname()==('ACD-Login-1' if site=='jxu' else 'xielab3')
allowed={'jxu':'/data/user/jxu957/experiments/easy2560-corollout-joint4-micro16-after730080-20261018/preparation/old-cosearch200-independent-verification/19ec0e1bc6224d8e85bbcba0b8ba9eb8','hk':'/backup/shaoshitong/cosearch-instruct200-common-em-20261011/preparation/cp200-tensor-retirement/19ec0e1bc6224d8e85bbcba0b8ba9eb8'}
root=Path(x['root']);assert str(root)==allowed[site] and root.resolve()==root;root.mkdir(parents=True,exist_ok=True);assert root.stat().st_uid==uid
assert hashlib.sha256((Path(x['code_root'])/'retire_old_cosearch200.py').read_bytes()).hexdigest()=='fbeec1fba246222622b07c681ede4dec3d0c04992581ec2f35e5c269ffb15b42'
raw=base64.b64decode(x['data_b64']);assert hashlib.sha256(raw).hexdigest()==x['sha256'];p=root/'deletion-result-imported.json'
if os.path.lexists(p):assert p.resolve()==p and p.is_file() and p.read_bytes()==raw
else:
 fd=os.open(p,os.O_WRONLY|os.O_CREAT|os.O_EXCL|os.O_NOFOLLOW,0o600)
 with os.fdopen(fd,'wb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
fd=os.open(root,os.O_RDONLY|os.O_DIRECTORY|os.O_NOFOLLOW);os.fsync(fd);os.close(fd)
print(json.dumps({'site':site,'uid':uid,'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'bytes':len(raw)},sort_keys=True))
