"""Stage immutable two-GPU online layer/prefix pilot; never submits Slurm jobs."""
import argparse,base64,hashlib,importlib.util,json,shlex,subprocess,zlib
from pathlib import Path
R=Path(__file__).resolve().parents[3];CONTROL=R.parents[1];P='reproduction/operations/qwen7b_warmup/'
BRANCH='codex/q7b-online-mixed-layer-prefix-2gpu-20261022'
FILES=['online_mixed_prefix_explore.py','deploy_online_mixed_prefix_explore.py','peer_long_prefix_diagnostic.py','native_mixed_group_classifier.py','runtime.sh']
BASE='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/online-mixed-layer-prefix-20261022-2gpu-v1'
VARIANTS=[(27,20),(21,20),(27,80)]
SBATCH='''#!/usr/bin/env bash
#SBATCH --job-name=q7b-online2-L@@L@@P@@P@@
#SBATCH --partition=acd_ue
#SBATCH --qos=normal
#SBATCH --account=mou176
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH --time=03:00:00
#SBATCH --output=@@BASE@@/layer@@L@@-prefix@@P@@/train-%j.log
set -euo pipefail
export QWEN7B_CODE_ROOT=@@CODE@@
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6
export PYTHONDONTWRITEBYTECODE=1 PYTHONFAULTHANDLER=1
export NCCL_ASYNC_ERROR_HANDLING=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
exec 9>@@BASE@@/layer@@L@@-prefix@@P@@/training.lock
flock -n 9
srun --kill-on-bad-exit=1 bash "${QWEN7B_CODE_ROOT}/reproduction/operations/qwen7b_warmup/runtime.sh" -u -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 -m reproduction.operations.qwen7b_warmup.online_mixed_prefix_explore --action train --layer @@L@@ --prefix @@P@@ --epochs 3 --lr 0.000005
'''
COMMON='''import pathlib,json,hashlib,base64,zlib,subprocess,os,time,fcntl
b=pathlib.Path(v['base']);b.mkdir(parents=True,exist_ok=True);code=b/('code-'+v['commit'][:12]);code.mkdir(exist_ok=True)
def atomic(p,x):
 p=pathlib.Path(p);q=p.with_name(p.name+'.tmp');q.write_text(json.dumps(x,sort_keys=True)+'\\n');q.replace(p)
def proof(p):
 p=pathlib.Path(p);r=p.read_bytes();return dict(path=str(p),bytes=len(r),sha256=hashlib.sha256(r).hexdigest())
'''
PART='''p=code/(v['name']+'.part-%04d'%v['index']);p.parent.mkdir(exist_ok=True,parents=True);raw=base64.b64decode(v['data']);assert hashlib.sha256(raw).hexdigest()==v['sha256']
if p.exists():assert p.read_bytes()==raw
else:p.write_bytes(raw)
print(json.dumps(dict(stored=True)))
'''
SEAL='''for name,record in v['files'].items():
 p=code/name;p.parent.mkdir(parents=True,exist_ok=True);parts=sorted(code.glob(name+'.part-*'));raw=b''.join(x.read_bytes()for x in parts)if parts else p.read_bytes();assert len(raw)==record['bytes']and hashlib.sha256(raw).hexdigest()==record['sha256']
 if p.exists():assert p.read_bytes()==raw
 else:p.write_bytes(raw);p.chmod(0o444)
 if name.endswith('.py'):compile(raw,name,'exec')
 for x in parts:x.unlink()
manifest=dict(commit=v['commit'],branch=v['branch'],code_root=str(code),files=v['files']);path=b/'source-publication.json'
replace_entries=False
if path.exists()and json.loads(path.read_text())!=manifest:
 old=json.loads(path.read_text());active=subprocess.check_output(['squeue','-h','-u','mou176','-o','%j'],universal_newlines=True);assert not any(n.startswith('q7b-online2-L')for n in active.splitlines()),'Existing pilot GPU job active; never alter its entry'
 assert not list(b.glob('layer*-prefix*/progress-rank-*.json')),'Existing training attempt must retain entry'
 archive=b/('pre-submit-source-'+old['commit'][:12]+'.json');atomic(archive,old);atomic(path,manifest);replace_entries=True
elif path.exists():assert json.loads(path.read_text())==manifest
else:atomic(path,manifest)
entries=[]
for layer,prefix in v['variants']:
 dest=b/('layer%d-prefix%d'%(layer,prefix));dest.mkdir(exist_ok=True);script=v['sbatch'].replace('@@BASE@@',str(b)).replace('@@CODE@@',str(code)).replace('@@L@@',str(layer)).replace('@@P@@',str(prefix));entry=dest/'training.sbatch'
 if entry.exists()and entry.read_text()!=script:
  assert replace_entries;archive=entry.with_name('pre-submit-'+old['commit'][:12]+'.sbatch');archive.write_text(entry.read_text());entry.write_text(script)
 elif entry.exists():assert entry.read_text()==script
 else:entry.write_text(script)
 subprocess.check_call(['bash','-n',str(entry)]);entries.append(dict(layer=layer,prefix=prefix,entry=proof(entry)))
print(json.dumps(dict(status='ONLINE_LAYER_PREFIX_THREE_2GPU_ENTRIES_STAGED_NOT_SUBMITTED',publication=proof(path),entries=entries)))
'''
PREPARE='''pub=json.loads((b/'source-publication.json').read_text());assert pub['commit']==v['commit']
f=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/online-mixed-layer-prefix-20261022-v1');fixture=json.loads((f/'fixture-complete.json').read_text());assert fixture['status']=='ONLINE_MIXED_MATCHED8192_G10_FIXED2048_TOKENS_READY'and fixture['training_questions']==8192 and fixture['validation_questions']==2048 and len(fixture['token_shards'])==4
receipt=dict(status='EXISTING_CAUSAL_TOKEN_FIXTURE_REUSED_NO_REEXTRACTION',fixture=proof(f/'fixture-complete.json'),source_inputs=str(f/'inputs'),training_questions=8192,validation_questions=2048,new_trajectory_or_feature_generation=False);atomic(b/'fixture-reuse.json',receipt);print(json.dumps(receipt))
'''
STATUS='''result=dict(time=time.time())
for name in('fixture-reuse.json',):
 if(b/name).exists():result[name]=json.loads((b/name).read_text())
if'prepare-process.json'in result:
 pid=result['prepare-process.json']['pid'];p=pathlib.Path('/proc/%d/cmdline'%pid);result['actual_prepare_identity_live']=p.exists()and b'online_mixed_prefix_explore'in p.read_bytes()
result['log_tail']=(b/'prepare.log').read_text()[-1500:]if(b/'prepare.log').exists()else''
print(json.dumps(result))
'''

def main():
 p=argparse.ArgumentParser();p.add_argument('--action',choices=['stage','prepare','status'],required=True);p.add_argument('--commit',required=True);a=p.parse_args();remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+BRANCH],cwd=R,text=True).split()[0]if a.action=='stage'else a.commit;assert remote==a.commit
 files={};payload={}
 for name in FILES:
  rel=P+name;raw=subprocess.check_output(['git','show',a.commit+':'+rel],cwd=R);payload[rel]=raw;files[rel]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
 spec=importlib.util.spec_from_file_location('online_private_control',CONTROL/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl);ip=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0'],text=True).strip()
 def call(source,**more):
  v=dict(base=BASE,branch=BRANCH,commit=a.commit,files=files,variants=VARIANTS,sbatch=SBATCH,**more);text='v='+repr(v)+'\n'+COMMON+source;packed=base64.b64encode(zlib.compress(text.encode())).decode();cmd='python3 -c '+shlex.quote('import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(packed)+')))');assert len(cmd.encode())<16000;r=ctl.run(cmd,account='mou176',source_address=ip,timeout=55)
  if r.returncode:raise RuntimeError(r.stdout[-2000:]+r.stderr[-2000:])
  return json.loads(r.stdout.splitlines()[-1])
 if a.action=='stage':
  for name,raw in payload.items():
   for i,start in enumerate(range(0,len(raw),3072)):
    z=raw[start:start+3072];call(PART,name=name,index=i,data=base64.b64encode(z).decode(),sha256=hashlib.sha256(z).hexdigest())
  result=call(SEAL)
 else:result=call(PREPARE if a.action=='prepare'else STATUS)
 out=R/'reproduction/reports/qwen7b-algorithm-improvement-20261021/online-layer-prefix-2gpu';out.mkdir(exist_ok=True,parents=True);(out/('actual-'+a.action+'.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
if __name__=='__main__':main()
