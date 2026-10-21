"""Separate screen, fixed3h entry and guarded root-authorized single submission."""
import argparse,base64,hashlib,importlib.util,json,pathlib,shlex,subprocess
R=pathlib.Path(__file__).resolve().parents[4];C=R.parents[1];HERE=pathlib.Path(__file__).parent
NAME='reproduction/operations/qwen7b_warmup/direct_count_large_screen.py'
BASE='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/direct-count-frozenE1-fit30016-hold5120-v1'
CSHA='0be60cc806a177790dce39fa28269303087a1b43ede90a439cce8a63f9e3a0fc'
COMMON='''import pathlib,json,hashlib,fcntl,os,base64,subprocess,time
b=pathlib.Path(v['base']);root=b/'screen-auditor';root.mkdir(exist_ok=True);guard=(root/'operator.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
def read(p):return json.loads(pathlib.Path(p).read_bytes())
def proof(p):
 p=pathlib.Path(p);raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
'''
def main():
 p=argparse.ArgumentParser();p.add_argument('--action',choices=('stage','prepare','submit','status'),required=True);p.add_argument('--submit-authorized',action='store_true');a=p.parse_args()
 if a.action=='submit'and not a.submit_authorized:p.error('Explicit root GO plus --submit-authorized required')
 s=importlib.util.spec_from_file_location('largecap_screen_entry_private',C/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(s);s.loader.exec_module(ctl);ip=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0']).decode().strip()
 def call(body,**values):
  values.update(base=BASE,csha=CSHA);cmd='python3 -c '+shlex.quote('v='+repr(values)+'\n'+COMMON+body)
  if len(cmd.encode())>=16000:raise ValueError('Control commands below16KB required')
  x=ctl.run(cmd,account='mou176',source_address=ip,timeout=55)
  if x.returncode:raise RuntimeError(x.stderr.decode()[-2500:]+x.stdout.decode()[-2000:])
  return json.loads(next(r for r in reversed(x.stdout.decode().splitlines())if r.startswith('{')))
 if a.action=='stage':
  commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R).decode().strip();remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/codex/qwen7b-probe-warmup-20261019'],cwd=R).decode().split()[0]
  if commit!=remote:raise ValueError('Publish screen rules/source first')
  raw=subprocess.check_output(['git','show',commit+':'+NAME],cwd=R)
  if raw!=(R/NAME).read_bytes():raise ValueError('Unpublished screen source refused')
  for i,start in enumerate(range(0,len(raw),4096)):
   part=raw[start:start+4096];call("code=root/('code-'+v['commit'][:12]);code.mkdir(exist_ok=True);p=code/('part-%04d'%v['i']);raw=base64.b64decode(v['data']);assert hashlib.sha256(raw).hexdigest()==v['sha']\nif p.exists():assert p.read_bytes()==raw\nelse:p.write_bytes(raw)\nprint(json.dumps(dict(part=v['i'])))",commit=commit,i=i,data=base64.b64encode(part).decode(),sha=hashlib.sha256(part).hexdigest())
  result=call("code=root/('code-'+v['commit'][:12]);raw=b''.join((code/('part-%04d'%i)).read_bytes()for i in range(v['parts']));assert hashlib.sha256(raw).hexdigest()==v['sha'];source=code/'screen.py';compile(raw,str(source),'exec')\nif source.exists():assert source.read_bytes()==raw\nelse:source.write_bytes(raw);source.chmod(0o444)\npub=dict(commit=v['commit'],remote_sha=v['commit'],source=proof(source));p=root/'source-publication.json'\nif p.exists():assert read(p)==pub\nelse:p.write_text(json.dumps(pub)+chr(10))\nprint(json.dumps(dict(publication=proof(p),screen_source=pub['source'])))",commit=commit,parts=(len(raw)+4095)//4096,sha=hashlib.sha256(raw).hexdigest())
 elif a.action=='prepare':
  result=call("""assert proof(b/'config.json')['sha256']==v['csha'];pub=read(b/'source-publication.json');code=pub['code_root'];screen=read(root/'source-publication.json');assert proof(screen['source']['path'])==screen['source']
env=dict(os.environ,QWEN7B_CODE_ROOT=code,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
r=subprocess.run(['bash',code+'/reproduction/operations/qwen7b_warmup/runtime.sh',screen['source']['path'],'--action','freeze','--base',str(b),'--config-sha256',v['csha']],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,env=env,timeout=35);assert r.returncode==0,r.stderr;frozen=json.loads(r.stdout.splitlines()[-1]);assert frozen['candidate_nodes']==12
script='''#!/usr/bin/env bash
#SBATCH --job-name=q7b-frozenE1-head30016
#SBATCH --partition=acd_u
#SBATCH --account=mou176
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --signal=USR1@300
#SBATCH --output=@@BASE@@/logs/cache-head-%j.log
#SBATCH --chdir=@@BASE@@
set -euo pipefail
export QWEN7B_CODE_ROOT=@@CODE@@
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 PYTHONDONTWRITEBYTECODE=1
export FLASH_ATTENTION_DETERMINISTIC=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
test -f @@BASE@@/screen-protocol/frozen-NQ-halves.json
srun --kill-on-bad-exit=1 bash "$QWEN7B_CODE_ROOT/reproduction/operations/qwen7b_warmup/runtime.sh" -u -m reproduction.operations.qwen7b_warmup.direct_count_large_capacity --action train --config-sha256 @@SHA@@
'''.replace('@@BASE@@',str(b)).replace('@@CODE@@',code).replace('@@SHA@@',v['csha'])
p=b/'capacity.sbatch'
if p.exists():assert p.read_text()==script
else:p.write_text(script)
(b/'logs').mkdir(exist_ok=True)
result=dict(status='LARGE_CAPACITY_SCREEN_FROZEN_AND_NORMAL3H_ENTRY_PREPARED_NO_SUBMISSION',config=proof(b/'config.json'),sbatch=proof(p),source_publication=proof(b/'source-publication.json'),source_commit=pub['commit'],code_root=code,training_implementation=read(b/'config.json')['implementation'],screen_publication=proof(root/'source-publication.json'),screen_source=screen['source'],frozen_halves=frozen,resources=dict(GPU=1,CPU=8,memory='128G',walltime='03:00:00',partition='acd_u',QoS='normal',dependency=None),GPU_submitted=False)
p=root/'entry-preparation.json'
if p.exists():assert read(p)==result
else:p.write_text(json.dumps(result,sort_keys=True)+chr(10))
print(json.dumps(result))""")
 elif a.action=='submit':
  result=call("""prep=read(root/'entry-preparation.json');assert proof(b/'config.json')==prep['config'] and prep['config']['sha256']==v['csha'];assert proof(b/'capacity.sbatch')==prep['sbatch'];assert prep['source_commit']=='20ac6243d9847ad1deab1e5d264f6dbb289a2b1d';assert proof(b/'source-publication.json')==prep['source_publication'];assert proof(prep['frozen_halves']['file']['path'])==prep['frozen_halves']['file']
assert prep['resources']==dict(GPU=1,CPU=8,memory='128G',walltime='03:00:00',partition='acd_u',QoS='normal',dependency=None)
guard=(b/'head-only-training.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
if(b/'job.id').exists():print(json.dumps(dict(status='EXISTING_LARGE_CAPACITY_JOB_PRESERVED',job_id=(b/'job.id').read_text().strip())))
else:
 assert not(b/'submit-intent.json').exists(),'Unknown submission outcome: reconcile actual named job first, no blind retry'
 q=subprocess.run(['squeue','-h','-u','mou176','-n','q7b-frozenE1-head30016','-o','%i|%j|%T|%Z'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True);assert q.returncode==0 and not q.stdout.strip(),q.stdout
 with(b/'submit-intent.json').open('x')as f:json.dump(dict(config=prep['config'],sbatch=prep['sbatch'],root_GO=True,epoch=time.time()),f);f.flush();os.fsync(f.fileno())
 r=subprocess.run(['sbatch','--parsable',str(b/'capacity.sbatch')],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True);(b/'submit-response.json').write_text(json.dumps(dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr))+chr(10));assert r.returncode==0,r.stderr
 jid=r.stdout.strip().split(';')[0];assert jid.isdigit(),r.stdout;(b/'job.id').write_text(jid+chr(10));result=dict(status='ONE_NORMAL_LARGE_FROZEN_E1_HEAD_CAPACITY_JOB_SUBMITTED',job_id=jid,config=prep['config'],sbatch=prep['sbatch'],resources=prep['resources'],source_commit=prep['source_commit'],screen_source=prep['screen_source'],epoch=time.time());(b/'submission.json').write_text(json.dumps(result)+chr(10));print(json.dumps(result))""")
 else:
  result=call("""out=dict(epoch=time.time())
for name in ('submission.json','progress.json','failure.json','interrupted.json','first-update-replay.json','latest.json','complete.json'):
 if(b/name).exists():out[name[:-5]]=read(b/name)
for name in ('progress.json','complete.json','same-api-golden.json'):
 if(b/'cache'/name).exists():out['cache-'+name[:-5]]=read(b/'cache'/name)
if(b/'job.id').exists():
 jid=(b/'job.id').read_text().strip();assert jid.isdigit();out['job_id']=jid
 for name,args in [('squeue',['squeue','-h','-j',jid,'-o','%i|%j|%T|%M|%R|%N']),('sacct',['sacct','-j',jid,'-n','-P','--format=JobIDRaw,State,ExitCode,Elapsed,NodeList'])]:
  r=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True);out[name]=dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
print(json.dumps(out))""")
 (HERE/('actual-screen-'+a.action+'.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
