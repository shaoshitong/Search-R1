"""Small code-only isolated deployment/run for the fixed existing-score audit."""
import argparse,base64,hashlib,importlib.util,json,pathlib,shlex,subprocess
P='reproduction/operations/qwen7b_warmup/'
ROOT='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/existing-predictions-domain-cursor512-v1'
RUNTIME='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/prefix-confidence-incremental-v1/independent-internal-audit-deployment-v1/code-39d5af483d5d'
COMMON='''import base64,fcntl,hashlib,json,os,pathlib,subprocess,time
b=pathlib.Path(values['base']);b.mkdir(exist_ok=True);lock=(b/'operator.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def proof(p):
 p=pathlib.Path(p);raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def read(p):return json.loads(pathlib.Path(p).read_text())
'''

def main():
 p=argparse.ArgumentParser();p.add_argument('--action',choices=['stage','run','status'],required=True);p.add_argument('--control-root',type=pathlib.Path,required=True);a=p.parse_args();repo=pathlib.Path(__file__).resolve().parents[3]
 spec=importlib.util.spec_from_file_location('domain_private_control',a.control_root/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl)
 ip=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0']).decode().strip()
 def call(code,**v):
  v['base']=ROOT;cmd='python3 -c '+shlex.quote('values='+repr(v)+'\n'+COMMON+code)
  if len(cmd.encode())>=16000:raise ValueError('Bounded control command required')
  r=ctl.run(cmd,account='mou176',source_address=ip,timeout=55)
  if r.returncode:raise RuntimeError(r.stderr.decode()[-1500:]+r.stdout.decode()[-1500:])
  return json.loads(next(s for s in reversed(r.stdout.decode().splitlines()) if s.startswith('{')))
 if a.action=='stage':
  commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo).decode().strip();remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/codex/qwen7b-probe-warmup-20261019'],cwd=repo).decode().split()[0];assert commit==remote
  files={}
  for name in ['domain_existing_predictions.py','test_domain_existing_predictions.py']:
   data=subprocess.check_output(['git','show',commit+':'+P+name],cwd=repo);files[name]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
   for i,start in enumerate(range(0,len(data),4096)):
    call("root=b/('code-'+values['commit'][:12]);root.mkdir(exist_ok=True);q=root/(values['name']+'.part-%04d'%values['index']);raw=base64.b64decode(values['data']);assert hashlib.sha256(raw).hexdigest()==values['sha']\nif q.exists():assert q.read_bytes()==raw\nelse:q.write_bytes(raw)\nprint(json.dumps(dict(part=values['index'])))",commit=commit,name=name,index=i,data=base64.b64encode(data[start:start+4096]).decode(),sha=hashlib.sha256(data[start:start+4096]).hexdigest())
  result=call("""root=b/('code-'+values['commit'][:12])
for name,ref in values['files'].items():
 parts=sorted(root.glob(name+'.part-*'));raw=b''.join(p.read_bytes() for p in parts);assert len(raw)==ref['bytes'] and hashlib.sha256(raw).hexdigest()==ref['sha256'];p=root/name
 if p.exists():assert p.read_bytes()==raw
 else:p.write_bytes(raw);p.chmod(0o444)
# The fixture relative import is qualified in an independent namespace root.
pub=dict(commit=values['commit'],remote_sha=values['commit'],code_root=str(root),files=values['files'])
p=b/'source-publication.json'
if p.exists():assert read(p)==pub
else:p.write_text(json.dumps(pub,sort_keys=True)+'\\n')
print(json.dumps(dict(status='DOMAIN_AUDITOR_SOURCE_STAGED',publication=proof(p))))
""",commit=commit,files=files)
 elif a.action=='run':
  result=call("""out=b/'complete.json'
if out.exists():print(json.dumps(dict(status='EXISTING_RESULT_PRESERVED',receipt=proof(out),result=read(out))))
else:
 assert not (b/'intent.json').exists(),'Inspect prior outcome before retry';pub=read(b/'source-publication.json');root=pathlib.Path(pub['code_root'])
 for name,ref in pub['files'].items():assert proof(root/name)['sha256']==ref['sha256']
 (b/'intent.json').write_text(json.dumps(dict(time=time.time(),source_publication=proof(b/'source-publication.json'),CPU=2,GPU=0))+'\\n')
 env=dict(os.environ,QWEN7B_CODE_ROOT=values['runtime'],CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
 with (b/'cpu.log').open('ab') as f:r=subprocess.run(['bash',values['runtime']+'/reproduction/operations/qwen7b_warmup/runtime.sh','-u',str(root/'domain_existing_predictions.py'),'--output',str(out)],env=env,stdout=f,stderr=subprocess.STDOUT,timeout=45)
 (b/'exit.json').write_text(json.dumps(dict(returncode=r.returncode,time=time.time()))+'\\n')
 assert r.returncode==0,(b/'cpu.log').read_text()[-2200:]
 print(json.dumps(dict(status='ACTUAL_DOMAIN_DECOMPOSITION_COMPLETE',receipt=proof(out),result=read(out))))
""",runtime=RUNTIME)
 else:result=call("p=b/'complete.json';print(json.dumps(dict(receipt=proof(p),result=read(p)) if p.exists() else dict(status='WAITING',exit=read(b/'exit.json') if (b/'exit.json').exists() else None,progress=read(b/'progress.json') if (b/'progress.json').exists() else None)))")
 out=repo/'reproduction/reports/qwen7b-algorithm-improvement-20261021/domain-existing-predictions';out.mkdir(exist_ok=True);(out/('actual-'+a.action+'.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
