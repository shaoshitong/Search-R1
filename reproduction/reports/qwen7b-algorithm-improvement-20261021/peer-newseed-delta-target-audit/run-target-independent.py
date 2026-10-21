"""Code-only stage and one finite raw target readback; never starts transport."""
import argparse,base64,hashlib,importlib.util,json,pathlib,shlex,subprocess
REPO=pathlib.Path(__file__).resolve().parents[4];CONTROL=REPO.parents[1]
NAME='reproduction/operations/qwen7b_warmup/audit_peer_delta_target_fresh.py'
REMOTE='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/peer-audited122-target-independent-v1'
TARGET='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/peer-newseed-h100-20261021/audited122-delta-20261021-673b8cb6'
COMMON='''import pathlib,json,hashlib,os,time,fcntl,base64,subprocess
b=pathlib.Path(values['base']);b.mkdir(exist_ok=True);guard=(b/'operator.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
def read(p):return json.loads(pathlib.Path(p).read_text())
def proof(p):
 p=pathlib.Path(p);raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
'''
def main():
 p=argparse.ArgumentParser();p.add_argument('--action',choices=('stage','run','status'),required=True);a=p.parse_args()
 spec=importlib.util.spec_from_file_location('target_audit_private_control',CONTROL/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl);ip=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0']).decode().strip()
 def call(body,**values):
  values.update(base=REMOTE,target=TARGET);command='python3 -c '+shlex.quote('values='+repr(values)+'\n'+COMMON+body)
  if len(command.encode())>=16000:raise ValueError('Bounded code control command required')
  r=ctl.run(command,account='mou176',source_address=ip,timeout=55)
  if r.returncode:raise RuntimeError('Readback outcome not accepted; inspect preserved intent/log: '+r.stderr.decode()[-1000:]+r.stdout.decode()[-2000:])
  return json.loads(next(x for x in reversed(r.stdout.decode().splitlines()) if x.startswith('{')))
 if a.action=='stage':
  commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO).decode().strip();remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/codex/qwen7b-probe-warmup-20261019'],cwd=REPO).decode().split()[0];assert commit==remote
  raw=subprocess.check_output(['git','show',commit+':'+NAME],cwd=REPO);sha=hashlib.sha256(raw).hexdigest()
  for i,start in enumerate(range(0,len(raw),4096)):
   part=raw[start:start+4096];call("root=b/('code-'+values['commit'][:12]);root.mkdir(exist_ok=True);p=root/('part-%04d'%values['index']);raw=base64.b64decode(values['data']);assert hashlib.sha256(raw).hexdigest()==values['sha']\nif p.exists():assert p.read_bytes()==raw\nelse:p.write_bytes(raw)\nprint(json.dumps(dict(part=values['index'])))",commit=commit,index=i,data=base64.b64encode(part).decode(),sha=hashlib.sha256(part).hexdigest())
  result=call("""root=b/('code-'+values['commit'][:12]);raw=b''.join((root/('part-%04d'%i)).read_bytes() for i in range(values['parts']));assert len(raw)==values['bytes'] and hashlib.sha256(raw).hexdigest()==values['sha'];p=root/'audit.py';compile(raw,str(p),'exec')
if p.exists():assert p.read_bytes()==raw
else:p.write_bytes(raw);p.chmod(0o444)
v=dict(commit=values['commit'],remote_sha=values['commit'],source=proof(p),code_root=str(root));q=b/'source-publication.json'
if q.exists():assert read(q)==v
else:q.write_text(json.dumps(v,sort_keys=True)+'\\n')
print(json.dumps(dict(status='FRESH_TARGET_AUDITOR_CODE_ONLY_STAGED',publication=proof(q))))
""",commit=commit,parts=(len(raw)+4095)//4096,bytes=len(raw),sha=sha)
 elif a.action=='run':
  result=call("""target=pathlib.Path(values['target']);output=b/'complete.json'
if not (target/'complete.json').is_file():print(json.dumps(dict(status='WAITING_FOR_ACTUAL_RECEIVER_COMPLETE_NO_BULK_READ')))
elif output.exists():print(json.dumps(dict(status='EXISTING_FRESH_AUDIT_PRESERVED',receipt=proof(output),result=read(output))))
else:
 assert not (b/'intent.json').exists(),'Inspect previous attempt before retry';pub=read(b/'source-publication.json');assert proof(pub['source']['path'])==pub['source']
 (b/'intent.json').write_text(json.dumps(dict(source_publication=proof(b/'source-publication.json'),target_complete=proof(target/'complete.json'),epoch=time.time(),GPU=0))+'\\n')
 with (b/'cpu.log').open('ab') as log:r=subprocess.run(['python3','-u',pub['source']['path'],'--output',str(b)],stdout=log,stderr=subprocess.STDOUT,timeout=45)
 (b/'exit.json').write_text(json.dumps(dict(returncode=r.returncode,epoch=time.time()))+'\\n');assert r.returncode==0,(b/'cpu.log').read_text()[-2200:]
 print(json.dumps(dict(status='ACTUAL_FRESH_TARGET_AUDIT_COMPLETE',receipt=proof(output),result=read(output))))
""")
 else:result=call("target=pathlib.Path(values['target']);out=dict(target_complete_exists=(target/'complete.json').is_file(),epoch=time.time());p=b/'complete.json'\nif p.exists():out.update(receipt=proof(p),result=read(p))\nelse:out.update(status='NOT_YET_INDEPENDENTLY_ACCEPTED',progress=read(b/'progress.json') if (b/'progress.json').exists() else None,exit=read(b/'exit.json') if (b/'exit.json').exists() else None)\nprint(json.dumps(out))")
 out=pathlib.Path(__file__).parent/('actual-'+a.action+'.json');out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
