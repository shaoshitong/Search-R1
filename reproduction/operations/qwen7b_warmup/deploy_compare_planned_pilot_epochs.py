"""Run published CPU-only all-planned-epoch comparison after seven acceptances."""
import argparse,base64,hashlib,importlib.util,json,shlex,subprocess,zlib
from pathlib import Path
R=Path(__file__).resolve().parents[3];CONTROL=R.parents[1];P='reproduction/operations/qwen7b_warmup/'
BRANCH='codex/q7b-planned-pilot-all-epochs-viewed2048-20261022'
BASE='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/planned-pilot-all-epochs-viewed2048-20261022-v1'
COMMON='''import pathlib,json,hashlib,base64,subprocess,os
b=pathlib.Path(v['base']);b.mkdir(exist_ok=True);code=b/('code-'+v['commit'][:12]);code.mkdir(exist_ok=True)
def proof(p):
 p=pathlib.Path(p);raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
'''
STORE='''raw=base64.b64decode(v['source']);assert hashlib.sha256(raw).hexdigest()==v['sha'];path=code/v['filename']
if path.exists():assert path.read_bytes()==raw
else:path.write_bytes(raw);path.chmod(0o444)
compile(raw,str(path),'exec');print(json.dumps(dict(stored=True,file=proof(path))))
'''
RUN='''suite=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/online-mixed-layer-prefix-20261022-2gpu-v1');pub=json.loads((suite/'source-publication.json').read_bytes());runtime=pathlib.Path(pub['code_root'])/'reproduction/operations/qwen7b_warmup/runtime.sh';env=dict(os.environ,QWEN7B_CODE_ROOT=pub['code_root'],CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
for filename,sha in v['sources'].items():assert proof(code/filename)['sha256']==sha
r=subprocess.run(['bash',str(runtime),'-u',str(code/'compare_planned_pilot_epochs.py'),'--helper',str(code/'compare_viewed2048_pilot_depth.py')],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=50)
if r.returncode:raise RuntimeError(r.stdout[-1500:]+r.stderr[-2500:])
print(r.stdout.strip())
'''


def main():
 p=argparse.ArgumentParser();p.add_argument('--commit',required=True);a=p.parse_args()
 head=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+BRANCH],cwd=R,text=True).split()[0];subprocess.check_call(['git','merge-base','--is-ancestor',a.commit,head],cwd=R)
 contents={n:subprocess.check_output(['git','show',a.commit+':'+P+n],cwd=R)for n in('compare_planned_pilot_epochs.py','compare_viewed2048_pilot_depth.py')}
 spec=importlib.util.spec_from_file_location('planned_epoch_private_control',CONTROL/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl);ip=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0'],text=True).strip()
 def call(body,**extra):
  v=dict(base=BASE,commit=a.commit,**extra);script='v='+repr(v)+'\n'+COMMON+body;compile(script,'planned_pilot_remote','exec');packed=base64.b64encode(zlib.compress(script.encode())).decode();command='python3 -c '+shlex.quote('import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(packed)+')))');assert len(command.encode())<16000;r=ctl.run(command,account='mou176',source_address=ip,timeout=58)
  if r.returncode:raise RuntimeError(r.stdout[-1600:]+r.stderr[-3000:])
  return json.loads(r.stdout.splitlines()[-1])
 for n,raw in contents.items():call(STORE,filename=n,source=base64.b64encode(raw).decode(),sha=hashlib.sha256(raw).hexdigest())
 result=call(RUN,sources={n:hashlib.sha256(raw).hexdigest()for n,raw in contents.items()});out=R/'reproduction/reports/qwen7b-algorithm-improvement-20261021/planned-pilot-all-epochs-viewed2048';out.mkdir(exist_ok=True,parents=True);(out/'actual-result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(dict(receipt=result['receipt'],ranking=result['ranking'],seconds=result['seconds'])))


for body in(COMMON,STORE,RUN):compile(body,'planned_pilot_static','exec')
if __name__=='__main__':main()
