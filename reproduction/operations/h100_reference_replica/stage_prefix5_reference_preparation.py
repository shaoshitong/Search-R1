from pathlib import Path
import sys,json,subprocess,hashlib,base64,gzip,datetime
REPO=Path('/Users/shaoshitong/Documents/Search-G1/h100-prefix5-reference-replica-20261018')
sys.path.insert(0,'/Users/shaoshitong/Documents/Search-G1/actorhidden-period20-20261016')
from reproduction.h1008_qprobe.established_controller_transport import remote
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip();assert not subprocess.check_output(['git','status','--porcelain'],cwd=REPO).strip()
manifest={}
for raw in subprocess.check_output(['git','ls-tree','-rz',commit],cwd=REPO).split(b'\0'):
 if not raw:continue
 meta,path=raw.split(b'\t',1);blob=meta.split()[2].decode();data=subprocess.check_output(['git','cat-file','blob',blob],cwd=REPO);manifest[path.decode()]=hashlib.sha256(data).hexdigest()
archive=gzip.compress(subprocess.check_output(['git','archive',commit],cwd=REPO),compresslevel=1)
inventory=json.loads(Path('work/prefix5-reference-h100-inventory.json').read_text());infra={k:v['path']for k,v in inventory['assets'].items()if k in ['environment_root','model_dir','retrieval_index','retrieval_corpus','retrieval_encoder','retriever_gpu_lib']}
payload={'commit':commit,'manifest':manifest,'archive':base64.b64encode(archive).decode(),'archive_sha256':hashlib.sha256(archive).hexdigest(),'infrastructure':infra}
script='P='+repr(payload)+'\n'+r'''
import pathlib,tarfile,io,gzip,hashlib,json,os,time
b=pathlib.Path('/mnt/data/shaoshitong/search-g1-prefix5-reference-preparation-20261018');d=b/'code'/P['commit'];repo=d/'repo';p=d/'source-staged.json'
if p.exists():print(p.read_text());raise SystemExit
repo.mkdir(parents=True,exist_ok=False);raw=__import__('base64').b64decode(P['archive']);assert hashlib.sha256(raw).hexdigest()==P['archive_sha256']
with tarfile.open(fileobj=io.BytesIO(gzip.decompress(raw)),mode='r:')as tar:
 for member in tar:
  name=pathlib.PurePosixPath(member.name);assert not name.is_absolute()and '..'not in name.parts
  target=repo/name
  if member.isdir():target.mkdir(parents=True,exist_ok=True);continue
  assert member.isfile();target.parent.mkdir(parents=True,exist_ok=True)
  with tar.extractfile(member)as f:content=f.read()
  assert hashlib.sha256(content).hexdigest()==P['manifest'][str(name)]
  with target.open('xb')as f:f.write(content)
  target.chmod(0o755 if member.mode&0o111 else 0o644)
actual={str(p.relative_to(repo)):hashlib.sha256(p.read_bytes()).hexdigest()for p in repo.rglob('*')if p.is_file()};assert actual==P['manifest']
infra=b/'infrastructure.json'
if not infra.exists():infra.write_text(json.dumps(P['infrastructure'],indent=2))
v={'status':'EXACT_REFERENCE_REPLICA_CODE_STAGED_FULL_SHA_PASS','repo':str(repo),'commit':P['commit'],'files':len(actual),'all_target_source_sha_match':True,'infrastructure':str(infra),'training_submitted':False,'checked_unix':time.time()}
p.write_text(json.dumps(v,indent=2));print(json.dumps(v))
'''
tag=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');p=Path('work')/('prefix5-reference-code-stage-'+tag+'.transport.json');v=json.loads(remote(script,str(p)));q=Path('work/prefix5-reference-code-stage.json');q.write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v))
