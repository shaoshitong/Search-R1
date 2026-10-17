from pathlib import Path
import json,shlex,datetime
from reference_prefix5_remote_transfer_route import controller,known,remote_on_mgmt
DEST='/mnt/data/shaoshitong/search-g1-prefix5-reference-assets-20261018-model-only'
RECEIVE=r'''
import pathlib,tarfile,sys,json,hashlib,os,time,socket
b=pathlib.Path(DEST);b.mkdir(exist_ok=True);p=b/'copy-complete.json';assert not p.exists(),'Read the existing proof; do not repeat the copy'
with(b/'copy-intent.json').open('x')as f:json.dump({'time':time.time(),'model_via_Mac':False},f)
files={}
with tarfile.open(fileobj=sys.stdin.buffer,mode='r|gz')as tar:
 for member in tar:
  assert member.isfile()and pathlib.Path(member.name).name==member.name
  path=b/member.name;h=hashlib.sha256();done=0
  with tar.extractfile(member)as source,path.open('xb')as target:
   for chunk in iter(lambda:source.read(8*1024**2),b''):h.update(chunk);target.write(chunk);done+=len(chunk)
   target.flush();os.fsync(target.fileno())
  assert done==member.size
  t=hashlib.sha256()
  with path.open('rb')as f:
   for chunk in iter(lambda:f.read(8*1024**2),b''):t.update(chunk)
  assert h.hexdigest()==t.hexdigest();files[member.name]={'path':str(path),'bytes':done,'sha256':t.hexdigest()}
proof=json.loads((b/'source-full-sha-proof.json').read_text());rows=[]
for r in proof['source_proof']:
 f=files[r['filename']];assert f['bytes']==r['bytes']and f['sha256']==r['actual_source_sha256']and r['sha256']in(None,f['sha256']);rows.append(dict(r,target=f['path'],independent_target_sha256=f['sha256']))
v={'status':'REFERENCE_ASSETS_REMOTE_COPY_SOURCE_TARGET_FULL_SHA_PASS','host':socket.gethostname(),'files':rows,'model_via_Mac':False,'source_training_changed':False,'finished_unix':time.time()}
with p.open('x')as f:json.dump(v,f,indent=2)
print(json.dumps(v))
'''.replace('DEST',repr(DEST))
def main():
 pack=json.loads(Path('work/prefix5-reference-asset-pack-summary.json').read_text());args=controller().route('h100-8','/usr/bin/python3 -c '+shlex.quote(RECEIVE));i=args.index('mgmt-host');assert len(args[i+1:])==1
 mgmt=r'''
import pathlib,tempfile,subprocess,json,os,concurrent.futures,math,hashlib,time,threading
b=pathlib.Path('/tmp/prefix5-reference-parallel-assets-20261018');b.mkdir(exist_ok=True)
with(b/'intent.json').open('x')as f:json.dump({'pack':PACK,'started':time.time(),'model_via_Mac':False},f)
p=b/'known_hosts';p.write_text(HOSTKEY);p.chmod(0o600);guard=threading.Lock();completed=[]
base=['ssh','-oBatchMode=yes','-oConnectTimeout=15','-oConnectionAttempts=1','-oStrictHostKeyChecking=yes','-oUserKnownHostsFile='+str(p),'-oServerAliveInterval=5','-p','2222','job-20e13974-c420-4074-a464-68e93738cfc0-master-0.ldwang.baai-sailing_aquila.dx-calc1.job@ssh.platform-multi.baai.ac.cn']
def copy(i):
 start=i*math.ceil(PACK['bytes']/16);size=min(math.ceil(PACK['bytes']/16),PACK['bytes']-start)
 code='import sys\nf=open('+repr(PACK['archive'])+',"rb");f.seek('+str(start)+');left='+str(size)+'\nwhile left:\n d=f.read(min(left,1048576));assert d;sys.stdout.buffer.write(d);left-=len(d)\n'
 import shlex
 path=b/('part-%02d'%i)
 with path.open('xb')as f:
  r=subprocess.run(base+['python3 -c '+shlex.quote(code)],stdout=f,stderr=subprocess.PIPE,timeout=600)
 assert r.returncode==0,r.stderr.decode();assert path.stat().st_size==size
 with guard:
  completed.append(i);(b/'progress.json').write_text(json.dumps({'complete_parts':sorted(completed),'bytes':sum(x.stat().st_size for x in b.glob('part-*')),'time':time.time()}))
 return path
with concurrent.futures.ThreadPoolExecutor(max_workers=16)as pool:
 paths=list(pool.map(copy,range(16)))
archive=b/'reference-assets.tar.gz';h=hashlib.sha256()
with archive.open('xb')as f:
 for path in paths:
  with path.open('rb')as source:
   for chunk in iter(lambda:source.read(8*1024**2),b''):h.update(chunk);f.write(chunk)
assert archive.stat().st_size==PACK['bytes']and h.hexdigest()==PACK['sha256']
with archive.open('rb')as f:r=subprocess.run(['sh','-c',TARGET_CMD],stdin=f,capture_output=True,timeout=240)
assert r.returncode==0,r.stderr.decode();result=json.loads(r.stdout);(b/'target-complete.json').write_text(json.dumps(result));print(json.dumps(result))
# Clear only this successful management staging, after full destination proof.
for path in paths+[archive]:path.unlink()
'''.replace('PACK',repr(pack)).replace('HOSTKEY',repr(known())).replace('TARGET_CMD',repr(args[i+1]))
 tag=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');r=remote_on_mgmt(mgmt,Path('work')/('prefix5-reference-parallel-copy-'+tag+'.transport.json'),timeout=900);assert r['returncode']==0,r['stderr'];v=json.loads(r['stdout']);p=Path('work/prefix5-reference-assets-copy-complete.json');p.write_text(json.dumps(v,indent=2)+'\n');print(json.dumps({'actual':str(p.resolve()),'files':len(v['files']),'bytes':sum(x['bytes']for x in v['files']),'status':v['status']}))
if __name__=='__main__':main()
