"""Stage a separate finite CPU auditor; run only after real terminal complete."""
import argparse,base64,hashlib,importlib.util,json,pathlib,shlex,subprocess
R=pathlib.Path(__file__).resolve().parents[4];C=R.parents[1];HERE=pathlib.Path(__file__).parent
NAME='reproduction/operations/qwen7b_warmup/audit_direct_count_capacity.py'
BASE='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/direct-count-capacity-independent-audit-v1'
TARGET='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/direct-count-head-capacity1536-v1'
COMMON='''import pathlib,json,hashlib,base64,fcntl,os,subprocess
b=pathlib.Path(v['base']);b.mkdir(exist_ok=True);lock=(b/'operator.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def proof(p):
 p=pathlib.Path(p);raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def read(p):return json.loads(pathlib.Path(p).read_bytes())
'''
def main():
 p=argparse.ArgumentParser();p.add_argument('--action',choices=('stage','run'),required=True);a=p.parse_args()
 spec=importlib.util.spec_from_file_location('headcapacity_independent_private',C/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl);ip=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0']).decode().strip()
 def call(body,**values):
  values.update(base=BASE,target=TARGET);cmd='python3 -c '+shlex.quote('v='+repr(values)+'\n'+COMMON+body)
  if len(cmd.encode())>=16000:raise ValueError('Bounded control commands required')
  r=ctl.run(cmd,account='mou176',source_address=ip,timeout=55)
  if r.returncode:raise RuntimeError('Independent auditor did not pass; preserve inputs: '+r.stderr.decode()[-1800:]+r.stdout.decode()[-2200:])
  return json.loads(next(x for x in reversed(r.stdout.decode().splitlines())if x.startswith('{')))
 if a.action=='stage':
  commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R).decode().strip();remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/codex/qwen7b-probe-warmup-20261019'],cwd=R).decode().split()[0]
  if commit!=remote:raise ValueError('Publish standalone auditor before execution')
  raw=subprocess.check_output(['git','show',commit+':'+NAME],cwd=R)
  for i,start in enumerate(range(0,len(raw),4096)):
   part=raw[start:start+4096];call("root=b/('code-'+v['commit'][:12]);root.mkdir(exist_ok=True);p=root/('part-%04d'%v['i']);raw=base64.b64decode(v['data']);assert hashlib.sha256(raw).hexdigest()==v['sha']\nif p.exists():assert p.read_bytes()==raw\nelse:p.write_bytes(raw)\nprint(json.dumps(dict(part=v['i'])))",commit=commit,i=i,data=base64.b64encode(part).decode(),sha=hashlib.sha256(part).hexdigest())
  result=call("root=b/('code-'+v['commit'][:12]);raw=b''.join((root/('part-%04d'%i)).read_bytes()for i in range(v['parts']));assert hashlib.sha256(raw).hexdigest()==v['sha'];source=root/'audit.py';compile(raw,str(source),'exec')\nif source.exists():assert source.read_bytes()==raw\nelse:source.write_bytes(raw);source.chmod(0o444)\npub=dict(commit=v['commit'],remote_sha=v['commit'],source=proof(source));p=b/'source-publication.json'\nif p.exists():assert read(p)==pub\nelse:p.write_text(json.dumps(pub)+'\\n')\nprint(json.dumps(dict(publication=proof(p))))",commit=commit,parts=(len(raw)+4095)//4096,sha=hashlib.sha256(raw).hexdigest())
 else:
  result=call("""t=pathlib.Path(v['target']);assert (t/'complete.json').exists(),'Wait for genuine three-fork complete'
state=subprocess.run(['sacct','-j','743757','-X','-n','-P','--format=State,ExitCode'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True);assert state.returncode==0 and state.stdout.strip()=='COMPLETED|0:0','Wait for original normal GPU termination: '+state.stdout
pub=read(b/'source-publication.json');assert proof(pub['source']['path'])==pub['source'];rt=read(t/'config.json')['original_runtime'];env=dict(os.environ,QWEN7B_CODE_ROOT=rt,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
if not (b/'complete.json').exists():
 with (b/'cpu.log').open('ab')as log:r=subprocess.run(['bash',rt+'/reproduction/operations/qwen7b_warmup/runtime.sh',pub['source']['path'],'--output',str(b)],stdout=log,stderr=subprocess.STDOUT,env=env,timeout=48)
 assert r.returncode==0,(b/'cpu.log').read_text()[-2500:]
print(json.dumps(dict(receipt=proof(b/'complete.json'),result=read(b/'complete.json'))))""")
 (HERE/('actual-independent-'+a.action+'.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
