"""Publish-checked finite CPU label/prior analysis; no training or eval reads."""
import argparse,base64,hashlib,importlib.util,json,pathlib,shlex,subprocess
REPO=pathlib.Path(__file__).resolve().parents[4];CONTROL=REPO.parents[1];OUT=pathlib.Path(__file__).parent
NAME='reproduction/operations/qwen7b_warmup/direct_count_label_prior.py'
BASE='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/direct-count256-label-prior-diagnostic-v2'
TARGET='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/direct-hidden-count30016-train-v2'
COMMON='''import pathlib,json,hashlib,os,fcntl,time,subprocess,base64
b=pathlib.Path(v['base']);b.mkdir(exist_ok=True);lock=(b/'operator.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def proof(p):
 p=pathlib.Path(p);raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def read(p):return json.loads(pathlib.Path(p).read_bytes())
'''
def main():
 p=argparse.ArgumentParser();p.add_argument('--action',choices=('stage','run','status'),required=True);a=p.parse_args()
 spec=importlib.util.spec_from_file_location('finite_label_prior_private_control',CONTROL/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl)
 ip=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0']).decode().strip()
 def call(body,**values):
  values.update(base=BASE,target=TARGET);command='python3 -c '+shlex.quote('v='+repr(values)+'\n'+COMMON+body)
  if len(command.encode())>=16000:raise ValueError('Control command must stay below16KB')
  r=ctl.run(command,account='mou176',source_address=ip,timeout=55)
  if r.returncode:raise RuntimeError('Finite diagnostic failed; preserve intent/log, no blind retry: '+r.stderr.decode()[-1800:]+r.stdout.decode()[-1600:])
  return json.loads(next(x for x in reversed(r.stdout.decode().splitlines())if x.startswith('{')))
 if a.action=='stage':
  commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO).decode().strip();remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/codex/qwen7b-probe-warmup-20261019'],cwd=REPO).decode().split()[0]
  if commit!=remote:raise ValueError('Publish source before execution')
  raw=subprocess.check_output(['git','show',commit+':'+NAME],cwd=REPO)
  if raw!=(REPO/NAME).read_bytes():raise ValueError('Unpublished analysis refused')
  for i,start in enumerate(range(0,len(raw),4096)):
   chunk=raw[start:start+4096]
   call("root=b/('code-'+v['commit'][:12]);root.mkdir(exist_ok=True);p=root/('part-%04d'%v['index']);raw=base64.b64decode(v['data']);assert hashlib.sha256(raw).hexdigest()==v['sha']\nif p.exists():assert p.read_bytes()==raw\nelse:p.write_bytes(raw)\nprint(json.dumps(dict(staged_part=v['index'])))",commit=commit,index=i,data=base64.b64encode(chunk).decode(),sha=hashlib.sha256(chunk).hexdigest())
  result=call("""root=b/('code-'+v['commit'][:12]);raw=b''.join((root/('part-%04d'%i)).read_bytes()for i in range(v['parts']));assert hashlib.sha256(raw).hexdigest()==v['sha'];source=root/'analyze.py';compile(raw,str(source),'exec')
if source.exists():assert source.read_bytes()==raw
else:source.write_bytes(raw);source.chmod(0o444)
pub=dict(commit=v['commit'],remote_sha=v['commit'],source=proof(source));p=b/'source-publication.json'
if p.exists():assert read(p)==pub
else:p.write_text(json.dumps(pub,sort_keys=True)+'\\n')
runtimeroot=read(pathlib.Path(v['target'])/'source-publication.json')['code_root'];env=dict(os.environ,QWEN7B_CODE_ROOT=runtimeroot,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
r=subprocess.run(['bash',runtimeroot+'/reproduction/operations/qwen7b_warmup/runtime.sh',str(source),'--self-test'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,env=env);assert r.returncode==0,r.stderr
print(json.dumps(dict(status='PURE_CPU_CODE_AND_CONSTANT_LOSS_FIXTURES_STAGED',publication=proof(p),fixture_result=r.stdout.strip())))""",commit=commit,parts=(len(raw)+4095)//4096,sha=hashlib.sha256(raw).hexdigest())
 elif a.action=='run':
  result=call("""p=b/'complete.json'
if p.exists():print(json.dumps(dict(status='EXISTING_FINITE_DIAGNOSTIC_PRESERVED',receipt=proof(p),result=read(p))))
else:
 assert not (b/'intent.json').exists(),'Inspect previous diagnostic attempt before retry';pub=read(b/'source-publication.json');assert proof(pub['source']['path'])==pub['source']
 (b/'intent.json').write_text(json.dumps(dict(source_publication=proof(b/'source-publication.json'),epoch=time.time(),GPU=0))+'\\n')
 runtimeroot=read(pathlib.Path(v['target'])/'source-publication.json')['code_root'];env=dict(os.environ,QWEN7B_CODE_ROOT=runtimeroot,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
 with (b/'cpu.log').open('ab')as log:r=subprocess.run(['bash',runtimeroot+'/reproduction/operations/qwen7b_warmup/runtime.sh','-u',pub['source']['path'],'--output',str(b)],stdout=log,stderr=subprocess.STDOUT,env=env,timeout=48)
 (b/'exit.json').write_text(json.dumps(dict(returncode=r.returncode,epoch=time.time()))+'\\n');assert r.returncode==0,(b/'cpu.log').read_text()[-2400:]
 print(json.dumps(dict(status='ACTUAL_FIRST256_TRAINING_LABEL_PRIOR_DIAGNOSTIC_COMPLETE',receipt=proof(p),result=read(p))))""")
 else:result=call("p=b/'complete.json';out=dict(epoch=time.time(),progress=read(b/'progress.json')if (b/'progress.json').exists()else None,exit=read(b/'exit.json')if (b/'exit.json').exists()else None)\nif p.exists():out.update(receipt=proof(p),result=read(p))\nprint(json.dumps(out))")
 (OUT/('actual-label-prior-'+a.action+'.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
