"""Minimal published stage/freeze/single-submit control for current-CP scoring."""
import argparse,base64,hashlib,importlib.util,json,subprocess
from pathlib import Path
R=Path(__file__).resolve().parents[3];PKG='reproduction/operations/qwen7b_warmup/'
A='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021'
BASE=A+'/joint124194-current-snapshot-eval-v1';MAIN=A+'/joint-balanced-native-E4-124194-v1';JOB='q7b-joint124194-snapshot'
OUT=R/'reproduction/reports/qwen7b-algorithm-improvement-20261021/joint124194-current-snapshot'
COMMON="""import pathlib,json,hashlib,os,time,fcntl,subprocess,re
b=pathlib.Path(v['base']);m=pathlib.Path(v['main']);code=b/('code-'+v['commit'][:12])
def rd(p):return json.loads(pathlib.Path(p).read_bytes())
def proof(p):
 p=pathlib.Path(p);h=hashlib.sha256()
 with p.open('rb')as f:
  for x in iter(lambda:f.read(8*1024**2),b''):h.update(x)
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())
def once(p,value):
 raw=value if isinstance(value,bytes)else(json.dumps(value,sort_keys=True,indent=2)+'\\n').encode();p=pathlib.Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 if p.exists():assert p.read_bytes()==raw,'Immutable asset differs: '+str(p)
 else:
  with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
def active():
 raw=subprocess.check_output(['squeue','-h','-u','mou176','-o','%i|%j|%T'],universal_newlines=True)
 return [x for x in raw.splitlines()if x.split('|')[1]==v['job']]
"""
PREPARE="""b.mkdir(parents=True,exist_ok=True)
lock=(b/'freeze.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not active()and not(b/'submission-intent.json').exists(),'Reconcile existing job/intent before preparation'
pub=rd(b/'source-publication.json');assert pub['commit']==v['commit']
for n,r in pub['files'].items():assert proof(code/n)==r
source_config=proof(m/'config.json');assert source_config['sha256']=='5aaa63c9348f410357be92ce491ba6d18286d618ab86dae6e63f193f2b4367b6'
qualified_publication=proof(m/'source-publication.json');oldpub=rd(qualified_publication['path']);assert oldpub['commit']=='3b56e583efe2469ac9108873c31de7144a5201e6'
for name,r in oldpub['files'].items():assert proof(pathlib.Path(oldpub['code_root'])/name)['sha256']==r['sha256']
actual_main=subprocess.check_output(['squeue','-h','-j','745501','-o','%i|%j|%u|%T|%N'],universal_newlines=True);assert '745501|q7b-native-joint124194|mou176|RUNNING|'in actual_main
fixed=b/'inputs/snapshot.json'
if fixed.exists():snapshot=rd(fixed);assert proof(snapshot['pinned_checkpoint']['path'])==snapshot['pinned_checkpoint']
else:
 for attempt in range(4):
  latest=rd(m/'latest.json');assert latest['config_sha256']==source_config['sha256']and latest['steps']==14418+latest['cursor']and latest['full_CPU_six_state_reload_passed']and latest['sampler_full_state_and_cursor_reload_passed']
  src=pathlib.Path(latest['path']);assert str(src).startswith(str(m/'checkpoints')+'/');pin=b/'inputs'/('snapshot-cursor-%06d.pt'%latest['cursor']);pin.parent.mkdir(parents=True,exist_ok=True)
  try:os.link(str(src),str(pin))
  except FileNotFoundError:continue
  except FileExistsError:assert pin.stat().st_ino==src.stat().st_ino
  actual=proof(pin);assert actual['bytes']==latest['bytes']and actual['sha256']==latest['sha256']
  now=rd(m/'latest.json')
  if now['cursor']>latest['cursor']:pin.unlink();continue
  assert now==latest
  snapshot=dict(checkpoint=latest,pinned_checkpoint=actual,hardlink_same_inode=True,actual_main=actual_main,epoch=time.time(),read_only_source=True);once(fixed,snapshot);break
 else:raise RuntimeError('Rolling checkpoint advanced/disappeared; no snapshot published; safely repeat prepare')
cfg=dict(role='accepted_current_checkpoint_read_only_snapshot',base=str(b),source_config=source_config,qualified_publication=qualified_publication,qualified_code_root=oldpub['code_root'],snapshot=snapshot,worker_sha256=pub['files']['joint124194_snapshot_eval.py']['sha256'],actor_updates=0,probe_updates=0,final4096_scored=False)
once(b/'config.json',cfg);entry=v['entry'].replace('@@BASE@@',str(b)).replace('@@CODE@@',str(code)).replace('@@RUNTIME@@',oldpub['code_root']).replace('@@SHA@@',proof(b/'config.json')['sha256']);once(b/'evaluate.sbatch',entry.encode());subprocess.check_call(['bash','-n',str(b/'evaluate.sbatch')]);(b/'logs').mkdir(exist_ok=True)
ready=dict(status='CURRENT_ACCEPTED_CP_HARDLINK_FULL_SHA_VERIFIED_SNAPSHOT_READY',snapshot_cursor=snapshot['checkpoint']['cursor'],snapshot=proof(fixed),config=proof(b/'config.json'),entry=proof(b/'evaluate.sbatch'),publication=proof(b/'source-publication.json'),actor_updates=0,probe_updates=0,final4096_scored=False);once(b/'prepared.json',ready);print(json.dumps(ready))
"""
ENTRY='''#!/usr/bin/env bash
#SBATCH --job-name=q7b-joint124194-snapshot
#SBATCH --partition=acd_u
#SBATCH --account=mou176
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=@@BASE@@/logs/evaluate-%j.log
#SBATCH --chdir=@@BASE@@
set -euo pipefail
export QWEN7B_CODE_ROOT=@@RUNTIME@@
export FLASH_ATTENTION_DETERMINISTIC=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
srun --kill-on-bad-exit=1 bash "$QWEN7B_CODE_ROOT/reproduction/operations/qwen7b_warmup/runtime.sh" -u @@CODE@@/joint124194_snapshot_eval.py --config @@BASE@@/config.json --config-sha256 @@SHA@@
'''
SUBMIT="""assert not active()and not(b/'submission-intent.json').exists(),'Unknown/existing submission must be reconciled, never retried'
lock=(b/'evaluation.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);lock.close();ready=rd(b/'prepared.json')
for key in('config','entry','publication','snapshot'):assert proof(ready[key]['path'])==ready[key]
once(b/'submission-intent.json',dict(commit=v['commit'],ready=ready,epoch=time.time()))
r=subprocess.run(['sbatch','--hold','--parsable',str(b/'evaluate.sbatch')],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=25)
if r.returncode or not re.fullmatch(r'[0-9]+(?:;[^ ]+)?',r.stdout.strip()):raise RuntimeError('Unknown sbatch: reconcile same-name job and intent before any retry: '+r.stdout+r.stderr)
job=r.stdout.strip().split(';')[0];actual=subprocess.check_output(['scontrol','show','job','-o',job],universal_newlines=True)
for token in('JobName='+v['job'],'Account=mou176','QOS=normal','Partition=acd_u','TimeLimit=01:00:00','NumCPUs=8','MinMemoryNode=128G','Dependency=(null)','Command='+str(b/'evaluate.sbatch'),'WorkDir='+str(b)):assert token in actual,token
assert re.search(r'(?:^|[ ,])gres/gpu=1(?:[ ,]|$)',actual)or re.search(r'TresPerNode=gres/gpu:1(?:[ ,]|$)',actual)
once(b/'submission-identity.json',dict(job_id=job,actual=actual,epoch=time.time(),commit=v['commit']));subprocess.check_call(['scontrol','release',job]);result=dict(status='CURRENT_SNAPSHOT_EVALUATION_SINGLE_JOB_RELEASED',job_id=job,snapshot_cursor=ready['snapshot_cursor'],identity=proof(b/'submission-identity.json'),config=ready['config']);once(b/'submission-result.json',result);print(json.dumps(result))
"""
STATUS="""out=dict(epoch=time.time(),base=str(b),active=active())
for n in('prepared.json','submission-result.json','controller.json','progress.json','restoration-acceptance.json','failure.json','complete.json'):
 try:
  if(b/n).exists():out[n[:-5]]=rd(b/n)
 except FileNotFoundError:pass
if(b/'submission-identity.json').exists():
 job=rd(b/'submission-identity.json')['job_id'];out['sacct']=subprocess.check_output(['sacct','-n','-X','-j',job,'-o','JobID,JobName,State,ExitCode,Elapsed,NodeList','-P'],universal_newlines=True)
 if(b/'logs'/('evaluate-'+job+'.log')).exists():out['log_tail']=(b/'logs'/('evaluate-'+job+'.log')).read_text()[-1800:]
print(json.dumps(out))
"""

def main():
    p=argparse.ArgumentParser();p.add_argument('--action',choices=['stage','prepare','submit','status'],required=True);p.add_argument('--commit');p.add_argument('--submit-authorized',action='store_true');a=p.parse_args()
    if a.action=='submit' and not a.submit_authorized:raise ValueError('Explicit authorized submit flag required')
    commit=a.commit or subprocess.check_output(['git','rev-parse','HEAD'],cwd=R).decode().strip()
    remote_head=subprocess.check_output(['git','ls-remote','origin','refs/heads/codex/qwen7b-probe-warmup-20261019'],cwd=R).decode().split()[0]
    assert not subprocess.run(['git','merge-base','--is-ancestor',commit,remote_head],cwd=R).returncode
    own=str(Path(__file__).relative_to(R));assert subprocess.check_output(['git','show',commit+':'+own],cwd=R)==Path(__file__).read_bytes()
    spec=importlib.util.spec_from_file_location('snapshot_transport',R/(PKG+'deploy_hpc3_tail_g5.py'));transport=importlib.util.module_from_spec(spec);spec.loader.exec_module(transport)
    common='v='+repr(dict(base=BASE,main=MAIN,commit=commit,job=JOB,entry=ENTRY))+'\n'+COMMON
    if a.action=='stage':
        refs={}
        for name in('joint124194_snapshot_eval.py','deploy_joint124194_snapshot_eval.py'):
            raw=subprocess.check_output(['git','show',commit+':'+PKG+name],cwd=R);ref=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest());refs[name]=dict(path=BASE+'/code-'+commit[:12]+'/'+name,**ref)
            for i in range(0,len(raw),4096):
                data=base64.b64encode(raw[i:i+4096]).decode();transport.remote(common+'once(code/'+repr(name+'.part-%04d'%(i//4096))+',__import__("base64").b64decode('+repr(data)+'));print(json.dumps(dict(staged='+repr(name)+',part='+str(i//4096)+')))')
            transport.remote(common+'raw=b"".join((code/('+repr(name+'.part-')+'+("%04d"%i))).read_bytes()for i in range('+str((len(raw)+4095)//4096)+'));assert len(raw)=='+str(len(raw))+'and hashlib.sha256(raw).hexdigest()=='+repr(ref['sha256'])+';once(code/'+repr(name)+',raw);print(json.dumps(proof(code/'+repr(name)+')))')
        result=transport.remote(common+'once(b/"source-publication.json",'+repr(dict(commit=commit,files=refs,qualified_runtime_commit='3b56e583efe2469ac9108873c31de7144a5201e6'))+');print(json.dumps(dict(status="SNAPSHOT_CODE_STAGED",publication=proof(b/"source-publication.json"))))')
    else:result=transport.remote(common+dict(prepare=PREPARE,submit=SUBMIT,status=STATUS)[a.action],timeout=115 if a.action=='prepare' else 55)
    OUT.mkdir(exist_ok=True,parents=True);(OUT/('actual-'+a.action+'.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
