"""Single-attempt exact Git source deployment plus separate read-only HK SHA check.
No credentials, model assets, training operation, or automatic transport retries.
"""
import base64,gzip,hashlib,io,json,shlex,subprocess,tarfile,time
from pathlib import Path
WS=Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-6')
W=WS/'work/easy2560-co-rollout-hpc3-4gpu-20261017'
OUT=WS/'outputs/easy2560-co-rollout-hpc3-4gpu-20261017'
COMMIT='8033adb58beb85bbf9ec5815486ca710a77d3177'
OPERATOR='reproduction/easy2560_joint4/code_backup/micro16-after730080-20261018/micro16-hk-source-deployment-operator.py'
OPERATOR_SHA='ae270ad97a9e70d53e7df7a46b82373228589eef34ae47e333441d82814843b4'
TAG='recovery5-after730912-8033-HK'
VERIFY=r"""import hashlib,json,os,socket,stat,sys,time
from pathlib import Path
p=json.load(sys.stdin);root=Path('/backup/shaoshitong/easy2560-corollout-hpc3-4gpu-20261017/source')/p['commit'];started=time.time()
assert socket.gethostname()=='xielab3' and os.getuid()==1002
assert root.resolve()==root and not root.is_symlink()
pubraw=(root/'source-publication.json').read_bytes();publication=json.loads(pubraw)
assert publication['source_commit']==p['commit'] and publication['files']==p['manifest'] and publication['fresh_source_all_files_sha_verified'] is True
actual={}
for name,record in p['manifest'].items():
 part=Path(name);assert not part.is_absolute() and '..' not in part.parts
 target=root/part;s=target.lstat();assert stat.S_ISREG(s.st_mode) and s.st_uid==1002 and s.st_nlink==1 and target.resolve()==target
 raw=target.read_bytes();after=target.stat()
 assert (s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)
 got={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()};assert got==record;actual[name]=got
files={str(x.relative_to(root)) for x in root.rglob('*') if x.is_file()}
assert files==set(actual)|{'source-publication.json'}
assert hashlib.sha256(pubraw).hexdigest()==p['publication_sha256']
print(json.dumps(dict(status='HKUST3_SECOND_INDEPENDENT_ALL_FILES_SHA_PASS',source_commit=p['commit'],source_root=str(root),files=len(actual),bytes=sum(r['bytes'] for r in actual.values()),publication_sha256=hashlib.sha256(pubraw).hexdigest(),started_unix=started,checked_unix=time.time(),host=socket.gethostname(),uid=os.getuid(),source_modified=False,jobs_submitted=0,controller_started=False)))
"""
def save(path,value):
 with path.open('x') as stream:json.dump(value,stream,indent=2);stream.write('\n')
def run_ssh(code,payload,evidence):
 argv=['ssh','-T','-oBatchMode=yes','-oStrictHostKeyChecking=yes','-oConnectTimeout=8','-oConnectionAttempts=1','-oServerAliveInterval=15','-oServerAliveCountMax=4','hkust3','python3 -B -c '+shlex.quote(code)]
 started=time.time()
 try:
  result=subprocess.run(argv,input=json.dumps(payload).encode(),capture_output=True,timeout=300)
  row=dict(returncode=result.returncode,stdout=result.stdout.decode(),stderr=result.stderr.decode(),started_unix=started,checked_unix=time.time(),transport_attempts=1,automatic_retry=False)
 except subprocess.TimeoutExpired as err:
  row=dict(returncode=None,status='UNKNOWN_AFTER_SSH_TIMEOUT_DO_NOT_RESEND_MUTATION',stdout=(err.stdout or b'').decode(),stderr=(err.stderr or b'').decode(),started_unix=started,checked_unix=time.time(),transport_attempts=1,automatic_retry=False)
 save(evidence,row)
 if row['returncode']!=0:raise RuntimeError('Single transport failed or uncertain; inspect saved receipt, do not repeat deployment')
 return json.loads(row['stdout'])
def main():
 assert subprocess.check_output(['git','rev-parse',COMMIT],cwd=W,text=True).strip()==COMMIT
 archive=subprocess.check_output(['git','archive','--format=tar',COMMIT],cwd=W);manifest={}
 with tarfile.open(fileobj=io.BytesIO(archive),mode='r:') as tf:
  for member in tf:
   if member.isdir():continue
   assert member.isfile() and member.name not in manifest
   part=Path(member.name);assert not part.is_absolute() and '..' not in part.parts
   raw=tf.extractfile(member).read();manifest[member.name]={'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
 assert len(manifest)==1378 and sum(r['bytes'] for r in manifest.values())==17741147
 operator=subprocess.check_output(['git','show',COMMIT+':'+OPERATOR],cwd=W)
 assert hashlib.sha256(operator).hexdigest()==OPERATOR_SHA
 save(OUT/(TAG+'-single-attempt-start.json'),dict(source_commit=COMMIT,operator_git_path=OPERATOR,operator_sha256=OPERATOR_SHA,manifest_files=len(manifest),manifest_bytes=sum(r['bytes'] for r in manifest.values()),created_unix=time.time(),no_automatic_retries=True,jobs_submitted=0,controller_started=False))
 payload={'commit':COMMIT,'manifest':manifest,'tar_gz':base64.b64encode(gzip.compress(archive)).decode()}
 deployed=run_ssh(operator.decode(),payload,OUT/(TAG+'-source-deployment.json'))
 assert deployed['status']=='HKUST3_INDEPENDENT_PUBLISHED_SOURCE_SHA_PASS' and deployed['source_commit']==COMMIT
 verified=run_ssh(VERIFY,dict(commit=COMMIT,manifest=manifest,publication_sha256=deployed['publication_sha256']),OUT/(TAG+'-source-independent-readback.json'))
 assert verified['status']=='HKUST3_SECOND_INDEPENDENT_ALL_FILES_SHA_PASS' and verified['files']==len(manifest) and verified['bytes']==17741147
 summary=dict(status='EXACT8033_HK_SOURCE_DEPLOYMENT_AND_SEPARATE_FULL_SHA_PASS',source_commit=COMMIT,source_root=verified['source_root'],files=len(manifest),bytes=verified['bytes'],operator_sha256=OPERATOR_SHA,publication_sha256=verified['publication_sha256'],deployment=deployed,independent_readback=verified,checked_unix=time.time(),transport_mutations=1,jobs_submitted=0,controller_started=False,old_controller_modified=False,training_modified=False,credentials_written=False)
 save(OUT/(TAG+'-source-completion.json'),summary);print(json.dumps(summary))
if __name__=='__main__':main()
