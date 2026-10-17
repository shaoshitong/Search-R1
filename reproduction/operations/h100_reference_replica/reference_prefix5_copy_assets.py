"""Copy exact reference assets A800 -> management host -> H100, never via Mac."""
from pathlib import Path
import json,sys,shlex,datetime,subprocess
from reference_prefix5_remote_transfer_route import controller,known,remote_on_mgmt
ROOT=Path(__file__).resolve().parent
DEST='/mnt/data/shaoshitong/search-g1-prefix5-reference-assets-20261018'
SOURCE=r'''
import pathlib,sys,json,hashlib,os
b=pathlib.Path('/data/shaoshitong/search-g1-prefix-selection200-a800-prefix5-54ac33dcebe64e62a11fe4177f41f116-resume-a4a0c2a18cca40bc8c3850d40eed8554')
c=json.loads((b/'preparation/launch-config.json').read_text());g=json.loads(pathlib.Path(c['grounding_manifest']).read_text());items=[]
keys=['probe_source_checkpoint','canonical_train','validation_questions','final_questions','common_manifest_path','composite_manifest_path','source_question_pool','fixed_test_identity_path','grounding_manifest','grounding_calibration_questions','probe_source_cpu_proof']
for k in keys:
 p=pathlib.Path(c[k]);expected=c.get(k+'_sha256')
 if k=='probe_source_checkpoint':expected=c['probe_source_sha256']
 if k=='grounding_calibration_questions':expected=c['grounding_calibration_sha256']
 if k=='probe_source_cpu_proof':expected=c.get('probe_source_cpu_proof_sha256')
 items.append({'role':k,'source':str(p),'bytes':p.stat().st_size,'sha256':expected,'filename':k+p.suffix})
for k,name in [('boundary','boundary_probe.pkl'),('commit','commit_probe.pkl')]:
 p=pathlib.Path(g[k]);items.append({'role':'grounding_'+k,'source':str(p),'bytes':p.stat().st_size,'sha256':g['head_sha256'][name],'filename':name})
for role,p in [('reference_launch_config',b/'preparation/launch-config.json')]:
 raw=p.read_bytes();items.append({'role':role,'source':str(p),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'filename':role+'.json'})
out=sys.stdout.buffer;out.write((json.dumps({'schema':1,'items':items})+'\n').encode());out.flush();proof=[]
for r in items:
 p=pathlib.Path(r['source']);before=p.stat();h=hashlib.sha256()
 with p.open('rb')as f:
  for chunk in iter(lambda:f.read(8*1024**2),b''):h.update(chunk);out.write(chunk)
 after=p.stat();assert(before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_ino,after.st_size,after.st_mtime_ns)
 r=dict(r,actual_source_sha256=h.hexdigest());assert r['sha256']in(None,h.hexdigest());proof.append(r)
out.write((json.dumps({'source_proof':proof})+'\n').encode());out.flush()
'''
RECEIVER=r'''
import pathlib,json,sys,hashlib,os,time,socket
b=pathlib.Path(DEST);b.mkdir(exist_ok=True);p=b/'copy-complete.json'
if p.exists():raise RuntimeError('Existing complete copy: read it instead of replaying source transfer')
claim=b/'copy-intent.json'
with claim.open('x')as f:json.dump({'started_unix':time.time(),'source':'reference A800','destination_host':socket.gethostname(),'model_via_Mac':False},f)
stream=sys.stdin.buffer;head=json.loads(stream.readline(1024**2));rows=[]
def progress(role,done):
 t=b/'copy-progress.partial';t.write_text(json.dumps({'role':role,'bytes':done,'completed_files':len(rows),'checked_unix':time.time()}));os.replace(t,b/'copy-progress.json')
for r in head['items']:
 assert pathlib.Path(r['filename']).name==r['filename'];path=b/r['filename'];tmp=path.with_name(path.name+'.partial');h=hashlib.sha256();remaining=r['bytes'];done=0;progress(r['role'],0)
 with tmp.open('xb')as f:
  while remaining:
   chunk=stream.read(min(8*1024**2,remaining));assert chunk,'Incomplete remote stream';h.update(chunk);f.write(chunk);remaining-=len(chunk);done+=len(chunk)
   if done%(64*1024**2)<8*1024**2:progress(r['role'],done)
  f.flush();os.fsync(f.fileno())
 assert r['sha256']in(None,h.hexdigest());os.replace(tmp,path)
 # Independently read the actual destination, not only the incoming stream.
 target=hashlib.sha256()
 with path.open('rb')as f:
  for chunk in iter(lambda:f.read(8*1024**2),b''):target.update(chunk)
 assert target.hexdigest()==h.hexdigest();rows.append(dict(r,target=str(path),independent_target_sha256=target.hexdigest()))
tail=json.loads(stream.readline(1024**2));assert len(tail['source_proof'])==len(rows)
for r,s in zip(rows,tail['source_proof']):assert r['role']==s['role']and r['independent_target_sha256']==s['actual_source_sha256'];r['actual_source_sha256']=s['actual_source_sha256']
v={'status':'REFERENCE_ASSETS_REMOTE_COPY_SOURCE_TARGET_FULL_SHA_PASS','host':socket.gethostname(),'files':rows,'model_via_Mac':False,'source_training_changed':False,'finished_unix':time.time()}
with p.open('x')as f:json.dump(v,f,indent=2)
print(json.dumps(v))
'''.replace('DEST',repr(DEST))
def main():
 target=controller().route('h100-8','/usr/bin/python3 -c '+shlex.quote(RECEIVER));i=target.index('mgmt-host');assert len(target[i+1:])==1;inner=target[i+1]
 mgmt=r'''
import pathlib,tempfile,subprocess,json,os
d=pathlib.Path(tempfile.mkdtemp(prefix='prefix5-asset-stream-'));p=d/'known_hosts';p.write_text(HOSTKEY);p.chmod(0o600)
a=['ssh','-oBatchMode=yes','-oConnectTimeout=12','-oStrictHostKeyChecking=yes','-oUserKnownHostsFile='+str(p),'-oServerAliveInterval=5','-p','2222','job-20e13974-c420-4074-a464-68e93738cfc0-master-0.ldwang.baai-sailing_aquila.dx-calc1.job@ssh.platform-multi.baai.ac.cn',SOURCE_CMD]
with (d/'source.stderr').open('wb')as err:
 source=subprocess.Popen(a,stdout=subprocess.PIPE,stderr=err)
 target=subprocess.Popen(['sh','-c',TARGET_CMD],stdin=source.stdout,stdout=subprocess.PIPE,stderr=subprocess.PIPE);source.stdout.close();out,e=target.communicate(timeout=215);s=source.wait(timeout=10)
print(json.dumps({'source_exit':s,'target_exit':target.returncode,'target_stdout':out.decode(),'target_stderr':e.decode(),'source_stderr':(d/'source.stderr').read_text(),'model_via_Mac':False}))
'''.replace('HOSTKEY',repr(known())).replace('SOURCE_CMD',repr('python3 -c '+shlex.quote(SOURCE))).replace('TARGET_CMD',repr(inner))
 tag=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');r=remote_on_mgmt(mgmt,ROOT/('prefix5-reference-asset-copy-'+tag+'.transport.json'));assert r['returncode']==0,r['stderr'];v=json.loads(r['stdout']);assert v['source_exit']==v['target_exit']==0,v;v=json.loads(v['target_stdout']);p=ROOT/'prefix5-reference-assets-copy-complete.json';p.write_text(json.dumps(v,indent=2)+'\n');print(json.dumps({'proof':str(p),'files':len(v['files']),'bytes':sum(x['bytes']for x in v['files']),'status':v['status']}))
if __name__=='__main__':main()
