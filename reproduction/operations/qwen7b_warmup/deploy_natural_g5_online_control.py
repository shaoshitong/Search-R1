"""Immutable natural-ownG5 paired control; stage/prepare/status never submit.

Only root explicitly invokes submit-once. The existing8192 token shards and
fixed evaluation metadata stay in place; new plans and outputs are isolated.
"""
import argparse,base64,hashlib,importlib.util,json,shlex,subprocess,zlib
from pathlib import Path
R=Path(__file__).resolve().parents[3];CONTROL=R.parents[1];P='reproduction/operations/qwen7b_warmup/'
BRANCH='codex/q7b-online-natural-g5-8192-20261022'
BASE='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/online-natural-g5-8192-20261022-v1'
SOURCE_BASE='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/online-mixed-layer-prefix-20261022-2gpu-v1'
SOURCE_COMMIT='5a3cfa1588c883d1c816d77ff95191acaa71eb35'
FILES=['online_mixed_prefix_explore.py','native_mixed_group_classifier.py','runtime.sh','natural_g5_online_control.py','deploy_natural_g5_online_control.py']
SBATCH='''#!/usr/bin/env bash
#SBATCH --job-name=q7b-natural2-L@@L@@P20
#SBATCH --partition=acd_u
#SBATCH --qos=normal
#SBATCH --account=mou176
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH --time=00:30:00
#SBATCH --output=@@BASE@@/layer@@L@@-prefix20/train-%j.log
set -euo pipefail
export QWEN7B_CODE_ROOT=@@CODE@@
export OMP_NUM_THREADS=6 MKL_NUM_THREADS=6 OPENBLAS_NUM_THREADS=6
export PYTHONDONTWRITEBYTECODE=1 PYTHONFAULTHANDLER=1
export NCCL_ASYNC_ERROR_HANDLING=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
exec 9>@@BASE@@/layer@@L@@-prefix20/training.lock
flock -n 9
srun --kill-on-bad-exit=1 bash "$QWEN7B_CODE_ROOT/reproduction/operations/qwen7b_warmup/runtime.sh" -u -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 -m reproduction.operations.qwen7b_warmup.natural_g5_online_control --action train --layer @@L@@ --prefix 20 --epochs 3 --lr 0.000005
'''
COMMON=r'''import pathlib,json,hashlib,base64,subprocess,os,time,fcntl
b=pathlib.Path(v['base']);b.mkdir(parents=True,exist_ok=True);code=b/('code-'+v['commit'][:12]);code.mkdir(exist_ok=True)
def read(p):return json.loads(pathlib.Path(p).read_text())
def proof(p):
 p=pathlib.Path(p);raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def atomic(p,x):
 p=pathlib.Path(p);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(x,sort_keys=True)+'\n');tmp.replace(p)
def active(layer):return subprocess.check_output(['squeue','-h','-u','mou176','-n','q7b-natural2-L%dP20'%layer,'-o','%i|%T|%j|%N|%R'],universal_newlines=True).strip().splitlines()
'''
PART=r'''p=code/(v['name']+'.part-%04d'%v['index']);p.parent.mkdir(parents=True,exist_ok=True);raw=base64.b64decode(v['data']);assert hashlib.sha256(raw).hexdigest()==v['sha256']
if p.exists():assert p.read_bytes()==raw
else:p.write_bytes(raw)
print(json.dumps(dict(stored=True)))
'''
SEAL=r'''source_path=pathlib.Path(v['source_base'])/'source-publication.json';source=read(source_path);assert source['commit']==v['source_commit']
for name,record in v['files'].items():
 p=code/name;p.parent.mkdir(parents=True,exist_ok=True);parts=sorted(code.glob(name+'.part-*'));raw=b''.join(x.read_bytes()for x in parts)if parts else p.read_bytes();assert len(raw)==record['bytes']and hashlib.sha256(raw).hexdigest()==record['sha256']
 if p.exists():assert p.read_bytes()==raw
 else:p.write_bytes(raw);p.chmod(0o444)
 if name.endswith('.py'):compile(raw,name,'exec')
 for q in parts:q.unlink()
for name in('online_mixed_prefix_explore.py','native_mixed_group_classifier.py','runtime.sh'):
 key='reproduction/operations/qwen7b_warmup/'+name;assert v['files'][key]==source['files'][key]
 assert proof(pathlib.Path(source['code_root'])/key)['sha256']==v['files'][key]['sha256']
publication=dict(commit=v['commit'],branch=v['branch'],code_root=str(code),files=v['files'],unchanged_online_math_source=proof(source_path));pub=b/'source-publication.json'
if pub.exists():assert read(pub)==publication
else:atomic(pub,publication)
entries=[]
for layer in(21,27):
 dest=b/('layer%d-prefix20'%layer);dest.mkdir(exist_ok=True);text=v['sbatch'].replace('@@BASE@@',str(b)).replace('@@CODE@@',str(code)).replace('@@L@@',str(layer));entry=dest/'training.sbatch'
 if entry.exists():assert entry.read_text()==text
 else:entry.write_text(text)
 subprocess.check_call(['bash','-n',str(entry)]);entries.append(dict(layer=layer,entry=proof(entry),job_name='q7b-natural2-L%dP20'%layer,active_same_name_jobs=active(layer)))
print(json.dumps(dict(status='NATURAL_G5_TWO_LAYER_ENTRIES_STAGED_NOT_SUBMITTED',publication=proof(pub),entries=entries,original_online_math_files_identical=True,Slurm_submitted=False)))
'''
PREPARE=r'''pub=read(b/'source-publication.json');assert pub['commit']==v['commit']
env=dict(os.environ,QWEN7B_CODE_ROOT=str(code),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
r=subprocess.run(['bash',str(code/'reproduction/operations/qwen7b_warmup/runtime.sh'),'-u','-m','reproduction.operations.qwen7b_warmup.natural_g5_online_control','--action','prepare'],env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=40)
if r.returncode:raise RuntimeError(r.stderr[-2000:]+r.stdout[-2000:])
value=json.loads([line for line in r.stdout.splitlines()if line.startswith('{')][-1]);print(json.dumps(value))
'''
STATUS=r'''result=dict(status='NATURAL_G5_MATCHED_CONTROL_READ_ONLY_STATUS',time=time.time(),variants=[])
for name in('source-publication.json','prepared.json'):
 if(b/name).exists():result[name]=dict(reference=proof(b/name),value=read(b/name))
for layer in(21,27):
 d=b/('layer%d-prefix20'%layer);row=dict(layer=layer,active_same_name_jobs=active(layer),files={})
 if(d/'job.id').exists():
  row['job_id']=(d/'job.id').read_text().strip();row['sacct']=subprocess.check_output(['sacct','-n','-P','-j',row['job_id'],'-o','JobIDRaw,State,ExitCode,Elapsed,NodeList'],universal_newlines=True).strip().splitlines()
 for name in('controller.json','progress-rank-0.json','progress-rank-1.json','latest.json','complete.json','failure-rank-0.json','failure-rank-1.json'):
  if(d/name).exists():row['files'][name]=read(d/name)
 row['evaluations']={p.name:read(p)for p in sorted(d.glob('evaluation-epoch-*.json'))}
 result['variants'].append(row)
print(json.dumps(result))
'''
SUBMIT=r'''pub=read(b/'source-publication.json');assert pub['commit']==v['commit'];prepared=read(b/'prepared.json');assert prepared['Slurm_submitted']is False
layer=v['layer'];d=b/('layer%d-prefix20'%layer);entry=d/'training.sbatch';guard=(b/'root-submit.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not(d/'job.id').exists(),'Prior owned job ID exists: inspect its real state, no automatic duplicate'
assert not(d/'submission-intent.json').exists(),'Prior intent exists: reconcile unknown outcome before another submit'
assert not active(layer),'Same-name live job exists'
intent=dict(layer=layer,entry=proof(entry),source_publication=proof(b/'source-publication.json'),preparation=proof(b/'prepared.json'),account='mou176',partition=v['partition'],qos=v['qos'],time_limit=v['time_limit'],gpus=2,no_dependency=True,epoch=time.time());atomic(d/'submission-intent.json',intent)
command=['sbatch','--parsable','--account=mou176','--partition='+v['partition'],'--qos='+v['qos'],'--time='+v['time_limit'],str(entry)]
r=subprocess.run(command,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True);atomic(d/'submission-result.json',dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,command=command,epoch=time.time()))
if r.returncode:raise RuntimeError('Submit rejected; inspect retained intent/result and reconcile, do not blindly retry')
job=r.stdout.strip().split(';')[0];assert job.isdigit();(d/'job.id').write_text(job+'\n');print(json.dumps(dict(status='ROOT_SUBMITTED_NATURAL_G5_CONTROL_ONCE',layer=layer,job_id=job,intent=intent)))
'''


def main():
 p=argparse.ArgumentParser();p.add_argument('--action',choices=('stage','prepare','status','submit-once'),required=True);p.add_argument('--commit',required=True)
 p.add_argument('--layer',type=int,choices=(21,27),default=21);p.add_argument('--partition',default='acd_u');p.add_argument('--qos',default='normal');p.add_argument('--time-limit',default='00:30:00');args=p.parse_args()
 if args.action=='stage':assert subprocess.check_output(['git','ls-remote','origin','refs/heads/'+BRANCH],cwd=R,text=True).split()[0]==args.commit
 payload={};files={}
 for name in FILES:
  rel=P+name;raw=subprocess.check_output(['git','show',args.commit+':'+rel],cwd=R);payload[rel]=raw;files[rel]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
 spec=importlib.util.spec_from_file_location('natural_g5_private_control',CONTROL/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl);ip=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0'],text=True).strip()
 def call(body,**more):
  v=dict(base=BASE,branch=BRANCH,commit=args.commit,source_base=SOURCE_BASE,source_commit=SOURCE_COMMIT,files=files,sbatch=SBATCH,layer=args.layer,partition=args.partition,qos=args.qos,time_limit=args.time_limit,**more);text='v='+repr(v)+'\n'+COMMON+body;packed=base64.b64encode(zlib.compress(text.encode())).decode();command='python3 -c '+shlex.quote('import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(packed)+')))');assert len(command.encode())<16000;result=ctl.run(command,account='mou176',source_address=ip,timeout=50)
  if result.returncode:raise RuntimeError(result.stdout[-2500:]+result.stderr[-2500:])
  return json.loads([line for line in result.stdout.splitlines()if line.strip().startswith(b'{')][-1])
 if args.action=='stage':
  for name,raw in payload.items():
   for index,start in enumerate(range(0,len(raw),3072)):
    chunk=raw[start:start+3072];call(PART,name=name,index=index,data=base64.b64encode(chunk).decode(),sha256=hashlib.sha256(chunk).hexdigest())
  result=call(SEAL)
 else:result=call({'prepare':PREPARE,'status':STATUS,'submit-once':SUBMIT}[args.action])
 report=R/'reproduction/reports/qwen7b-algorithm-improvement-20261021/natural-g5-online-8192';report.mkdir(parents=True,exist_ok=True);(report/('actual-'+args.action+('-L%d'%args.layer if args.action=='submit-once'else'')+'.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))


if __name__=='__main__':main()
