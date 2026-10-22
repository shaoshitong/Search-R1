"""Stage corrected window2018-v2 explicit-seed2 terminal CPU entry only."""
import argparse,base64,hashlib,importlib.util,json,shlex,subprocess,zlib
from pathlib import Path
R=Path(__file__).resolve().parents[3];ROOT=R.parents[1];P='reproduction/operations/qwen7b_warmup/'
BRANCH='codex/q7b-window2018-seed2-v2-CPU-acceptance-20261023'
REMOTE='''import pathlib,json,zlib,base64,hashlib,subprocess
A=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021');b=A/'seed2-window2018-v2-terminal-CPU-acceptance-20261023-v1';assert A.is_dir();b.mkdir(exist_ok=True);code=b/('code-'+v['commit'][:12]);code.mkdir(exist_ok=True)
def proof(p):
 raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
for name,ref in v['files'].items():
 raw=zlib.decompress(base64.b64decode(ref['payload']));assert len(raw)==ref['bytes']and hashlib.sha256(raw).hexdigest()==ref['sha256'];p=code/name
 if p.exists():assert p.read_bytes()==raw
 else:p.write_bytes(raw);p.chmod(0o444)
 compile(raw,name,'exec')
entries=[]
for suite,cell in [('online-packed-window2018-seed19137-plan2026102302-L15P80-8192-20261023-v2','window2018')]:
 base=A/suite/cell;pub=json.loads((base.parent/'source-publication.json').read_bytes());root=pathlib.Path(pub['code_root']);runtime=root/'reproduction/operations/qwen7b_warmup/runtime.sh';assert runtime.is_file();entry=code/(cell+'.sh')
 text=chr(10).join(['#!/usr/bin/env bash','set -euo pipefail','export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 QWEN7B_CODE_ROOT='+repr(str(root)),'exec bash '+repr(str(runtime))+' -u '+repr(str(code/'accept_seeded_window2018_v2_checkpoint.py'))+' --base '+repr(str(base))+' --source-sha '+v['files']['accept_seeded_window2018_v2_checkpoint.py']['sha256']+' "$@"',''])
 if entry.exists():assert entry.read_text()==text
 else:entry.write_text(text)
 subprocess.check_call(['bash','-n',str(entry)]);entries.append(dict(cell=cell,base=str(base),entry=proof(entry),training_source_publication=proof(base.parent/'source-publication.json')))
result=dict(status='EXPLICIT_SEED2_MATCHED_WINDOW_TERMINAL_ONLY_CPU_ACCEPTOR_STAGED_NOT_STARTED',source_commit=v['commit'],files={n:proof(code/n)for n in v['files']},entries=entries,shared_global_CPU_lock=str(A/'online-mixed-P80-handoff-CPU-acceptance-20261023-v1/acceptance.lock'),checkpoints_opened=False,GPU_or_Slurm_action=False,final4096_read=False)
path=code/'staged.json'
if path.exists():assert json.loads(path.read_bytes())==result
else:path.write_text(json.dumps(result,sort_keys=True)+chr(10))
print(json.dumps(result))
'''
def main():
 p=argparse.ArgumentParser();p.add_argument('--commit',required=True);a=p.parse_args();remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+BRANCH],cwd=R,text=True).split()[0];subprocess.check_call(['git','merge-base','--is-ancestor',a.commit,remote],cwd=R);files={}
 for n in ('accept_seeded_window2018_v2_checkpoint.py','detached_score_window.py'):
  raw=subprocess.check_output(['git','show',a.commit+':'+P+n],cwd=R);files[n]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),payload=base64.b64encode(zlib.compress(raw)).decode())
 script='v='+repr(dict(commit=a.commit,files=files))+'\n'+REMOTE;compile(script,'remote_CPU_acceptor_stage','exec');command='python3 -c '+shlex.quote('import zlib,base64;exec(zlib.decompress(base64.b64decode('+repr(base64.b64encode(zlib.compress(script.encode())).decode())+')))');assert len(command.encode())<16000
 spec=importlib.util.spec_from_file_location('private_control',ROOT/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl);address=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0'],text=True).strip();r=ctl.run(command,account='mou176',source_address=address,timeout=35)
 if r.returncode:raise RuntimeError(repr(r.stdout[-1500:])+repr(r.stderr[-1500:]))
 result=json.loads(r.stdout);out=R/'reproduction/reports/qwen7b-algorithm-improvement-20261021/seed2-window2018-v2-CPU-acceptance';out.mkdir(exist_ok=True,parents=True);(out/'actual-stage.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
compile(REMOTE,'remote_CPU_acceptor_stage','exec')
if __name__=='__main__':main()
