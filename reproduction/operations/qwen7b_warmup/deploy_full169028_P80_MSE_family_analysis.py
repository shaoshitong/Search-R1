"""Stage/run a published CPU-only selector over already-viewed scores."""
import argparse,base64,hashlib,importlib.util,json,shlex,subprocess,zlib
from pathlib import Path
R=Path(__file__).resolve().parents[3];CONTROL=R.parents[1]
P='reproduction/operations/qwen7b_warmup/'
BRANCH='codex/q7b-native-joint-root-20261021'
FILES=('select_compare_full169028_P80_MSE.py','compare_full169028_packed_variance.py')
REMOTE='''import pathlib,json,base64,hashlib,subprocess,os,time
A=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021');assert A.is_dir()
b=A/'full169028-P80-MSE-actual-family-selection-20261023-v1';b.mkdir(exist_ok=True);code=b/('code-'+v['commit'][:12]);code.mkdir(exist_ok=True)
def proof(p):
 raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
for name,ref in v['files'].items():
 raw=base64.b64decode(ref['data']);assert len(raw)==ref['bytes']and hashlib.sha256(raw).hexdigest()==ref['sha256'];p=code/name;compile(raw,name,'exec')
 if p.exists():assert p.read_bytes()==raw
 else:p.write_bytes(raw);p.chmod(0o444)
pub=b/('source-publication-'+v['commit'][:12]+'.json');document=dict(commit=v['commit'],branch=v['branch'],code_root=str(code),files={n:proof(code/n)for n in v['files']})
raw=(json.dumps(document,sort_keys=True)+chr(10)).encode()
if pub.exists():assert pub.read_bytes()==raw
else:pub.write_bytes(raw)
training=A/'full169028-L15P80-MSE-ownG5-physical-seed9137-plan2026102227-20261023-v3';runtime=pathlib.Path(json.loads((training/'source-publication.json').read_bytes())['code_root'])/'reproduction/operations/qwen7b_warmup/runtime.sh'
entry=code/'analyze.sh';text='#!/usr/bin/env bash'+chr(10)+'set -euo pipefail'+chr(10)+'export CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1'+chr(10)+'exec bash '+str(runtime)+' -u '+str(code/'select_compare_full169028_P80_MSE.py')+chr(10)
if entry.exists():assert entry.read_text()==text
else:entry.write_text(text);entry.chmod(0o444)
subprocess.check_call(['bash','-n',str(entry)])
out=dict(status='FULL_MSE_FAMILY_SELECTOR_STAGED_NOT_SELECTED',source_publication=proof(pub),entry=proof(entry),selection_rule='seed1 Top512 maximum then earliest epoch then EMA; same epoch/state seed2',final4096_read=False,checkpoint_read=False,Slurm_mutated=False,time=time.time())
if v['action']=='run':
 result=subprocess.run(['bash',str(entry)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=50)
 if result.returncode:raise RuntimeError(result.stdout[-1000:]+result.stderr[-2000:])
 parsed=json.loads(result.stdout.strip().splitlines()[-1]);out.update(status=parsed['result']['status'],result=parsed)
print(json.dumps(out))
'''

def main():
 p=argparse.ArgumentParser();p.add_argument('--commit',required=True);p.add_argument('--action',choices=('stage','run'),required=True);a=p.parse_args()
 remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+BRANCH],cwd=R,text=True).split()[0];subprocess.check_call(['git','merge-base','--is-ancestor',a.commit,remote],cwd=R)
 sources={n:subprocess.check_output(['git','show',a.commit+':'+P+n],cwd=R)for n in FILES};payload={n:dict(data=base64.b64encode(raw).decode(),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())for n,raw in sources.items()}
 spec=importlib.util.spec_from_file_location('private_control',CONTROL/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl)
 script='v='+repr(dict(commit=a.commit,branch=BRANCH,files=payload,action=a.action))+'\n'+REMOTE;packed=base64.b64encode(zlib.compress(script.encode())).decode();command='python3 -c '+shlex.quote('import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(packed)+')))');assert len(command.encode())<16000
 address=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0'],text=True).strip();reply=ctl.run(command,account='mou176',source_address=address,timeout=60)
 if reply.returncode:raise RuntimeError(reply.stdout[-1000:]+reply.stderr[-2000:])
 result=json.loads(reply.stdout.strip().splitlines()[-1]);folder=R/'reproduction/reports/qwen7b-algorithm-improvement-20261021/full169028-P80-MSE-family-selection';folder.mkdir(exist_ok=True);(folder/('actual-'+a.action+'.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
