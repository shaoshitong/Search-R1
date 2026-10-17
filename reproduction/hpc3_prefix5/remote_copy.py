"""Remote source-to-HKUST2 transfer using a short-lived source-only SSH agent.

The Mac forwards authentication requests only. Asset bytes flow directly from
A800 to HKUST2; private keys never leave the Mac and the global agent is unchanged.
"""
import hashlib,json,os,pathlib,re,shlex,subprocess,time
ROOT=pathlib.Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5')
SOURCE='/data/shaoshitong/thread5-prefix5-hpc3-sync-20261017'
REMOTE = r'''
import json,pathlib,subprocess,sys,shlex,time,hashlib,fcntl,os
p=json.load(sys.stdin);root=pathlib.Path('/data/shaoshitong/thread5-prefix5-hpc3-sync-20261017');root.mkdir(exist_ok=True)
with (root/'copy.lock').open('a') as lock:
 fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 key=root/'source-known-hosts';key.write_text(p['known_hosts']);key.chmod(0o600)
 ssh=['ssh','-T','-oBatchMode=yes','-oStrictHostKeyChecking=yes','-oConnectTimeout=8','-oServerAliveInterval=5','-oUserKnownHostsFile='+str(key),'-oGlobalKnownHostsFile=/dev/null','-oProxyCommand=nc 110.43.160.159 %p','-p','2222',p['user']+'@ssh.platform-multi.baai.ac.cn']
 code="import pathlib,time; p=pathlib.Path('/data/shaoshitong/thread5-prefix5-hpc3-sync-20261017/export-complete.json'); deadline=time.time()+90;\nwhile not p.exists() and time.time()<deadline:time.sleep(1)\nprint(p.read_text())"
 result=subprocess.run(ssh+['python3 -c '+shlex.quote(code)],capture_output=True,timeout=110)
 if result.returncode:raise RuntimeError('Source SSH authentication/receipt failed: '+result.stderr.decode()[:400])
 proof=json.loads(result.stdout);(root/'source-export.json').write_text(json.dumps(proof))
 target=root/'reference-assets.tar';part=root/'reference-assets.tar.part'
 offset=part.stat().st_size if part.exists() else 0
 if not target.exists():
  code="import pathlib,sys;p=pathlib.Path('/data/shaoshitong/thread5-prefix5-hpc3-sync-20261017/reference-assets.tar');f=p.open('rb');f.seek("+str(offset)+");\nfor b in iter(lambda:f.read(4194304),b''):sys.stdout.buffer.write(b)"
  with (root/'source-ssh.log').open('ab') as error:
   child=subprocess.Popen(ssh+['python3 -c '+shlex.quote(code)],stdout=subprocess.PIPE,stderr=error)
   with part.open('ab') as stream:
    for chunk in iter(lambda:child.stdout.read(4*1024**2),b''):
     stream.write(chunk);stream.flush()
     (root/'progress.json').write_text(json.dumps(dict(epoch=time.time(),bytes=stream.tell(),total=proof['archive_bytes'],route='A800_DIRECT_SSH_TO_HKUST2',mac_model_data=False)))
    os.fsync(stream.fileno())
   if child.wait()!=0:raise RuntimeError('Source stream failed; partial retained')
  if part.stat().st_size!=proof['archive_bytes']:raise ValueError('Target archive size differs')
  h=hashlib.sha256()
  with part.open('rb') as stream:
   for chunk in iter(lambda:stream.read(8*1024**2),b''):h.update(chunk)
  if h.hexdigest()!=proof['archive_sha256']:raise ValueError('HKUST2 archive SHA differs')
  part.replace(target)
 proof.update(status='SOURCE_HKUST2_ARCHIVE_SHA_VERIFIED',route='A800_DIRECT_SSH_TO_HKUST2',mac_model_data=False,completed_unix=time.time())
 (root/'copy-complete.json').write_text(json.dumps(proof));print(json.dumps({k:v for k,v in proof.items() if k!='manifest'}))
'''

def main():
 config={}
 for line in subprocess.check_output(['ssh','-G','baai-a800'],text=True,stderr=subprocess.DEVNULL).splitlines():
  key,_,value=line.partition(' ');config.setdefault(key,[]).append(value)
 identity=pathlib.Path(config['identityfile'][0]).expanduser()
 agent=subprocess.check_output(['ssh-agent','-s'],text=True)
 sock=re.search(r'SSH_AUTH_SOCK=([^;]+);',agent).group(1)
 pid=re.search(r'SSH_AGENT_PID=([0-9]+);',agent).group(1)
 env=dict(os.environ,SSH_AUTH_SOCK=sock,SSH_AGENT_PID=pid)
 output=ROOT/'outputs/hpc3-prefix5-switch-20261017';output.mkdir(exist_ok=True)
 try:
  added=subprocess.run(['ssh-add','-t','600',str(identity)],env=env,capture_output=True,text=True,timeout=8)
  if added.returncode:raise RuntimeError('Source-only temporary agent could not load the existing identity')
  known=subprocess.check_output(['ssh-keygen','-F','[ssh.platform-multi.baai.ac.cn]:2222'],text=True)
  request=dict(user=config['user'][0],known_hosts=known)
  result=subprocess.run(['ssh','-oForwardAgent='+sock,'-oConnectTimeout=8','hkust2','python3 -c '+shlex.quote(REMOTE)],input=json.dumps(request).encode(),capture_output=True,timeout=240)
  receipt=dict(returncode=result.returncode,stdout=result.stdout.decode(),stderr=result.stderr.decode(),observed_unix=time.time(),mac_model_data=False,private_key_copied=False,global_agent_changed=False)
  (output/'direct-source-copy.json').write_text(json.dumps(receipt,indent=2));print(json.dumps(receipt))
 finally:
  subprocess.run(['ssh-agent','-k'],env=env,capture_output=True)

if __name__=='__main__':main()
