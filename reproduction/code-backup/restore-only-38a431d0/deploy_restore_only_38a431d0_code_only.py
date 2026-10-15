"""Task publication only: code blobs and immutable metadata, no restore execution."""
import base64,json,subprocess,sys,zlib
from pathlib import Path
REPO=Path('/Users/shaoshitong/Documents/Search-G1/qprobe-restore-only-20261016')
WORK=Path(__file__).resolve().parent
sys.path.insert(0,str(REPO))
from reproduction.h1008_qprobe.management_transport import remote
proof=json.loads((WORK/'restore-only-38a431d0-independent-readback.json').read_text())
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()
assert commit==proof['published_control_commit'] and proof['all_blobs_full_sha_match'] is True
assert not subprocess.check_output(['git','status','--porcelain'],cwd=REPO)
parent='c5fc09603a10244655b98ec0ce8eada32668f83b'
listing=subprocess.check_output(['git','ls-tree','-r','--format=%(objectname) %(path)',parent],cwd=REPO,text=True)
import hashlib
parentfiles={}
for row in listing.splitlines():
 oid,name=row.split(' ',1);parentfiles[name]=hashlib.sha256(subprocess.check_output(['git','cat-file','blob',oid],cwd=REPO)).hexdigest()
changes={name:base64.b64encode(subprocess.check_output(['git','show',commit+':'+name],cwd=REPO)).decode() for name,value in proof['files'].items() if parentfiles.get(name)!=value}
assert set(changes)=={name for name in proof['files'] if name.startswith('reproduction/restore_only/')}
meta=dict(commit=commit,parent=parent,parentfiles=parentfiles,files=proof['files'],changes=changes,proof=proof)
script='''import json,hashlib,base64,os,shutil,datetime
from pathlib import Path
meta=META
source=Path('/mnt/data/shaoshitong/search-g1-qprobe3-binary10-c5fc096-20261015')
root=Path('/mnt/data/shaoshitong/search-g1-restore-only-'+meta['commit'][:7]+'-20261016')
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def verify(base,files):
 for name,value in files.items():
  p=base/name;assert not p.is_symlink() and p.is_file() and digest(p)==value,name
def once(p,value):
 with p.open('x') as f:json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno())
if root.exists():
 assert (root/'deploy-intent.json').is_file() and (root/'deploy-receipt.json').is_file(),'OCCUPIED_INSPECT_NO_REPLAY'
 receipt=json.loads((root/'deploy-receipt.json').read_text());assert receipt['source_commit']==meta['commit']
 verify(root/'control',meta['files']);print(json.dumps(dict(existing_matching_receipt=True,verified_files=len(meta['files']))))
else:
 assert json.loads((source/'logs/run-manifest.json').read_text())['source_commit']==meta['parent']
 assert json.loads((source/'preparation/source-files.json').read_text())==meta['parentfiles']
 verify(source/'repo',meta['parentfiles'])
 root.mkdir();control=root/'control';control.mkdir()
 once(root/'deploy-intent.json',dict(role='restore_only_code_only_deployment',source_commit=meta['commit'],frozen_training_runtime=meta['parent'],gpu_started=False,actual_restore_started=False))
 for name,value in meta['files'].items():
  p=control/name;p.parent.mkdir(parents=True,exist_ok=True)
  if name in meta['changes']:p.write_bytes(base64.b64decode(meta['changes'][name]))
  else:shutil.copyfile(source/'repo'/name,p)
 verify(control,meta['files'])
 once(root/'publication-proof.json',meta['proof'])
 once(root/'code-only-configuration.json',dict(control_repo=str(control),existing_environment_root=str((source/'env').resolve()),environment_copied=False,all_five_binding_prepared=False,native_submitted=False,actual_restore_started=False))
 result=dict(role='restore_only_code_only_deployment',source_commit=meta['commit'],control_repo=str(control),verified_files=len(meta['files']),all_target_source_full_sha_match=True,actual_restore_started=False,gpu_started=False,training_modified=False,checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
 once(root/'deploy-receipt.json',result);print(json.dumps(result))
'''.replace('META',repr(meta))
print(remote(script,WORK/'restore-only-38a431d0-code-only-deploy.json','/var/folders/6k/lf0fpt5j2ps0ztkt_tyvwntw0000gn/T/qprobe3-mgmt-88myknuw/control.sock'))
