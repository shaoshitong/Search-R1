"""Stage independent full169028 Z second-seed terminal CPU entry, never launch."""
import argparse,base64,hashlib,importlib.util,json,shlex,subprocess,zlib
from pathlib import Path
R=Path(__file__).resolve().parents[3];CONTROL=R.parents[1];P='reproduction/operations/qwen7b_warmup/';BRANCH='codex/q7b-full169028-Z-seed2-CPU-acceptance-20261023';NAME='accept_full169028_Z_seed2_checkpoint.py'
REMOTE='''import pathlib,json,base64,zlib,hashlib,subprocess
A=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021');assert A.is_dir();b=A/'full169028-Z-seed2-terminal-CPU-acceptance-20261023-v1';b.mkdir(exist_ok=True);code=b/('code-'+v['commit'][:12]);code.mkdir(exist_ok=True)
def proof(p):
 raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
raw=zlib.decompress(base64.b64decode(v['source']));assert hashlib.sha256(raw).hexdigest()==v['sha256'];p=code/v['name'];compile(raw,str(p),'exec')
if p.exists():assert p.read_bytes()==raw
else:p.write_bytes(raw);p.chmod(0o444)
entries=[]
for seed,plan in [(19137,2026102302)]:
 base=A/'online-packed-full169028-Z-seed19137-plan2026102302-P20-20261023-v1';assert(base/'source-publication.json').is_file(),'Actual full run source must already be staged';pub=json.loads((base/'source-publication.json').read_bytes());assert pub['commit']=='e5e5678ec6f42669509b493f08fab0dbea4994a8';root=pathlib.Path(pub['code_root']);module=root/'reproduction/operations/qwen7b_warmup/online_packed_variance_seeded.py';assert module.is_file()and proof(module)==dict(path=str(module),**pub['files']['reproduction/operations/qwen7b_warmup/online_packed_variance_seeded.py']);runtime=root/'reproduction/operations/qwen7b_warmup/runtime.sh';assert runtime.is_file()and proof(runtime)==dict(path=str(runtime),**pub['files']['reproduction/operations/qwen7b_warmup/runtime.sh']);entry=code/('seed%d.sh'%seed);text=chr(10).join(['#!/usr/bin/env bash','set -euo pipefail','export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 QWEN7B_CODE_ROOT='+repr(str(root)),'exec bash '+repr(str(runtime))+' -u '+repr(str(p))+' --base '+repr(str(base))+' --source-sha '+v['sha256']+' "$@"',''])
 if entry.exists():assert entry.read_text()==text
 else:entry.write_text(text)
 subprocess.check_call(['bash','-n',str(entry)]);entries.append(dict(init_seed=seed,train_plan_seed=plan,base=str(base),entry=proof(entry),actual_training_source_publication=proof(base/'source-publication.json'),actual_module=proof(module)))
result=dict(status='FULL169028_Z_SEED2_TERMINAL_CPU_ACCEPTOR_STAGED_NOT_STARTED',source_commit=v['commit'],validator=proof(p),entries=entries,shared_global_CPU_lock=str(A/'online-mixed-P80-handoff-CPU-acceptance-20261023-v1/acceptance.lock'),actual_fixture_complete_required_before_acceptance=True,actual_seeded_module_and_all10_explicit_plan_SHA_required=True,legacy_shared_P20_fixture_not_new_P80_physical_fixture=True,model_checkpoint_or_tensor_opened=False,GPU_or_Slurm_action=False,final4096_read=False);s=code/'staged.json'
if s.exists():assert json.loads(s.read_bytes())==result
else:s.write_text(json.dumps(result,sort_keys=True)+chr(10))
print(json.dumps(result))
'''
def main():
 p=argparse.ArgumentParser();p.add_argument('--commit',required=True);a=p.parse_args();remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+BRANCH],cwd=R,text=True).split()[0];subprocess.check_call(['git','merge-base','--is-ancestor',a.commit,remote],cwd=R);raw=subprocess.check_output(['git','show',a.commit+':'+P+NAME],cwd=R);v=dict(commit=a.commit,name=NAME,source=base64.b64encode(zlib.compress(raw)).decode(),sha256=hashlib.sha256(raw).hexdigest());script='v='+repr(v)+'\n'+REMOTE;compile(script,'full_CPU_stage','exec');command='python3 -c '+shlex.quote('import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(base64.b64encode(zlib.compress(script.encode())).decode())+')))');assert len(command.encode())<16000
 spec=importlib.util.spec_from_file_location('control',CONTROL/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl);ip=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0'],text=True).strip();reply=ctl.run(command,account='mou176',source_address=ip,timeout=35)
 if reply.returncode:raise RuntimeError(reply.stdout[-1000:]+reply.stderr[-1800:])
 result=json.loads(reply.stdout);out=R/'reproduction/reports/qwen7b-algorithm-improvement-20261021/full169028-Z-seed2-CPU-acceptance';out.mkdir(exist_ok=True,parents=True);(out/'actual-stage.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
compile(REMOTE,'full_CPU_stage','exec')
if __name__=='__main__':main()
