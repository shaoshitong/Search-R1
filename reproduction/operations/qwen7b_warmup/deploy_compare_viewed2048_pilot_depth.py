"""Run the published CPU-only comparison of existing viewed2048 predictions."""
import argparse,base64,hashlib,importlib.util,json,shlex,subprocess,zlib
from pathlib import Path
R=Path(__file__).resolve().parents[3];CONTROL=R.parents[1];P='reproduction/operations/qwen7b_warmup/'
BRANCH='codex/q7b-viewed2048-pilot-depth-comparison-20261022'
BASE='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/viewed2048-pilot-depth-comparison-20261022-v1'
REMOTE='''import pathlib,json,hashlib,base64,subprocess,os
b=pathlib.Path(v['base']);b.mkdir(exist_ok=True);raw=base64.b64decode(v['source']);assert hashlib.sha256(raw).hexdigest()==v['sha256'];code=b/('code-'+v['commit'][:12]);code.mkdir(exist_ok=True);path=code/'compare_viewed2048_pilot_depth.py'
if path.exists():assert path.read_bytes()==raw
else:path.write_bytes(raw);path.chmod(0o444)
compile(raw,str(path),'exec');suite=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/online-mixed-layer-prefix-20261022-2gpu-v1');pub=json.loads((suite/'source-publication.json').read_bytes());runtime=pathlib.Path(pub['code_root'])/'reproduction/operations/qwen7b_warmup/runtime.sh';env=dict(os.environ,QWEN7B_CODE_ROOT=pub['code_root'],CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
r=subprocess.run(['bash',str(runtime),'-u',str(path)],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=45)
if r.returncode:raise RuntimeError(r.stdout[-1000:]+r.stderr[-3000:])
print(r.stdout.strip())
'''


def main():
 p=argparse.ArgumentParser();p.add_argument('--commit',required=True);a=p.parse_args()
 remote_sha=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+BRANCH],cwd=R,text=True).split()[0];subprocess.check_call(['git','merge-base','--is-ancestor',a.commit,remote_sha],cwd=R)
 raw=subprocess.check_output(['git','show',a.commit+':'+P+'compare_viewed2048_pilot_depth.py'],cwd=R)
 spec=importlib.util.spec_from_file_location('viewed_pilot_private_control',CONTROL/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl);ip=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0'],text=True).strip()
 v=dict(base=BASE,commit=a.commit,source=base64.b64encode(raw).decode(),sha256=hashlib.sha256(raw).hexdigest());script='v='+repr(v)+'\n'+REMOTE;compile(script,'viewed_pilot_remote','exec');packed=base64.b64encode(zlib.compress(script.encode())).decode();command='python3 -c '+shlex.quote('import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(packed)+')))');assert len(command.encode())<16000;r=ctl.run(command,account='mou176',source_address=ip,timeout=55)
 if r.returncode:raise RuntimeError(r.stdout[-1600:]+r.stderr[-3000:])
 result=json.loads(r.stdout.splitlines()[-1]);out=R/'reproduction/reports/qwen7b-algorithm-improvement-20261021/viewed2048-pilot-depth-comparison';out.mkdir(exist_ok=True,parents=True);(out/'actual-result.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


compile(REMOTE,'viewed_pilot_remote_static','exec')
if __name__=='__main__':main()
