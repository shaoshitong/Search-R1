"""One finite immutable CPU-only diagnostic, published code + aggregate output."""
import argparse,base64,gzip,hashlib,importlib.util,json,pathlib,shlex,subprocess
P='reproduction/operations/qwen7b_warmup/'
CONFIG='reproduction/reports/qwen7b-algorithm-improvement-20261021/headlr-fixed-prior-diagnostic/config.json'
BASE='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/headlr-fixed-prior-diagnostic-v1'
BODY='''import pathlib,json,base64,gzip,hashlib,fcntl,subprocess,os,time
b=pathlib.Path(v['base']);b.mkdir(exist_ok=True);lock=(b/'operator.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);code=b/('code-'+v['commit'][:12]);code.mkdir(exist_ok=True)
def read(p):return json.loads(pathlib.Path(p).read_text())
def proof(p):
 raw=pathlib.Path(p).read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def write(p,x):
 if p.exists():assert read(p)==x
 else:p.write_text(json.dumps(x,sort_keys=True)+'\\n')
raw=gzip.decompress(base64.b64decode(v['payload']));assert hashlib.sha256(raw).hexdigest()==v['payload_sha'];files=json.loads(raw)
for name,data in files.items():
 data=base64.b64decode(data);p=code/name
 if p.exists():assert p.read_bytes()==data
 else:p.write_bytes(data);p.chmod(0o444)
publication=dict(commit=v['commit'],files={n:proof(code/n)for n in files},code_root=str(code));write(b/'source-publication.json',publication)
if v['action']=='run':
 if (b/'summary.json').exists():raise RuntimeError('Existing result must be read via status; no repeated run')
 if (b/'intent.json').exists():raise RuntimeError('Unknown previous outcome: inspect PID/log, do not repeat')
 write(b/'intent.json',dict(source=proof(b/'source-publication.json'),epoch=time.time(),GPU=0))
 target=pathlib.Path(v['target']);pub=read(target/'source-publication.json');assert pub['commit']=='501cf80478b77f68c3e13a96d4659e3da0971b2e'
 env=dict(os.environ,QWEN7B_CODE_ROOT=pub['code_root'],CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
 with (b/'cpu.log').open('ab') as log:r=subprocess.run(['bash',pub['code_root']+'/reproduction/operations/qwen7b_warmup/runtime.sh','-u',str(code/'headlr_fixed_prior_diagnostic.py'),'--config',str(code/'config.json'),'--output',str(b)],env=env,stdout=log,stderr=subprocess.STDOUT,timeout=65)
 write(b/'exit.json',dict(returncode=r.returncode,epoch=time.time()))
 if r.returncode:raise RuntimeError((b/'cpu.log').read_text()[-2200:])
result=dict(source=proof(b/'source-publication.json'),exit=read(b/'exit.json')if (b/'exit.json').exists()else None)
if (b/'summary.json').exists():result.update(summary=read(b/'summary.json'),receipt=proof(b/'summary.json'),actual_progress=[json.loads(s)for s in (b/'cpu.log').read_text().splitlines()if s.startswith('{')])
print(json.dumps(result))
'''

def main():
 p=argparse.ArgumentParser();p.add_argument('--commit',required=True);p.add_argument('--action',choices=('run','status'),required=True);v=p.parse_args();root=pathlib.Path(__file__).resolve().parents[3];control=root.parents[1];payload={}
 for name in (P+'headlr_fixed_prior_diagnostic.py',CONFIG):
  raw=subprocess.check_output(['git','show',v.commit+':'+name],cwd=root)
  if raw!=(root/name).read_bytes():raise ValueError('Current source/config differs from published commit')
  payload[pathlib.Path(name).name]=base64.b64encode(raw).decode()
 remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/codex/qwen7b-probe-warmup-20261019'],cwd=root).decode().split()[0]
 if remote!=v.commit:raise ValueError('Publication has not been remotely verified')
 raw=json.dumps(payload,sort_keys=True).encode();values=dict(base=BASE,target=BASE.rsplit('/',1)[0]+'/direct-hidden-count30016-train-v2',commit=v.commit,action=v.action,payload=base64.b64encode(gzip.compress(raw,mtime=0)).decode(),payload_sha=hashlib.sha256(raw).hexdigest())
 code='v='+repr(values)+'\n'+BODY;compile(code,'remote-worker','exec');command='python3 -c '+shlex.quote(code)
 if len(command.encode())>=16000:raise ValueError('Code-only command exceeds16KB')
 spec=importlib.util.spec_from_file_location('headlr_diagnostic_private_control',control/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl)
 r=ctl.run(command,account='mou176',source_address=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0']).decode().strip(),timeout=80)
 if r.returncode:raise RuntimeError(r.stderr[-2400:])
 result=json.loads(r.stdout.splitlines()[-1]);out=root/'reproduction/reports/qwen7b-algorithm-improvement-20261021/headlr-fixed-prior-diagnostic'/('actual-'+v.action+'.json');out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
