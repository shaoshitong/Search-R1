"""One authorized all252 G10 experiment: code-only stage/CPU gates/guarded job.

Never duplicates source assets or submits on import. Holds/reconciles unknown
outcomes; no dependencies, no priority escalation, no GPU before explicitsubmit.
"""
import argparse,ast,base64,gzip,hashlib,importlib.util,io,json,shlex,subprocess,tarfile
from pathlib import Path
BASE='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/g10-exact-uniform252-v1'
JOB='q7b-g10-all252'
P='reproduction/operations/qwen7b_warmup/'
SEEDS=tuple(P+n for n in ('deploy_exact_uniform_g5.py','prepare_exact_uniform_g5.py','probe_exact_uniform_g5_train.py','test_exact_uniform_g5_loss.py','test_probe_exact_uniform_g5_train.py','runtime.sh'))


def closure(repo):
    pending=list(SEEDS);seen=set()
    while pending:
        name=pending.pop()
        if name in seen:continue
        path=repo/name
        if not path.is_file():raise ValueError('Missing source dependency '+name)
        seen.add(name)
        if path.suffix!='.py':continue
        package=name[:-3].split('/')[:-1]
        for n in ast.walk(ast.parse(path.read_text())):
            modules=[]
            if isinstance(n,ast.ImportFrom):
                if n.level:
                    prefix=package[:len(package)-n.level+1];modules=['.'.join(prefix+[n.module])] if n.module else ['.'.join(prefix+[a.name]) for a in n.names]
                elif n.module and n.module.startswith('reproduction.'):modules=[n.module]
            elif isinstance(n,ast.Import):modules=[a.name for a in n.names if a.name.startswith('reproduction.')]
            for module in modules:
                child=module.replace('.','/')+'.py'
                if not (repo/child).is_file():raise ValueError('Unresolved runtime dependency '+module)
                pending.append(child)
    return sorted(seen)


def script(code):
    return '''#!/usr/bin/env bash
#SBATCH --job-name=q7b-g10-all252
#SBATCH --account=mou176
#SBATCH --partition=acd_u
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --signal=USR1@180
#SBATCH --output=@@BASE@@/logs/train-%j.log
set -euo pipefail
export QWEN7B_CODE_ROOT=@@CODEQ@@
export FLASH_ATTENTION_DETERMINISTIC=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1
srun --kill-on-bad-exit=1 bash "$QWEN7B_CODE_ROOT/reproduction/operations/qwen7b_warmup/runtime.sh" -u -m reproduction.operations.qwen7b_warmup.probe_exact_uniform_g5_train --action train --base @@BASEQ@@
'''.replace('@@BASE@@',BASE).replace('@@BASEQ@@',shlex.quote(BASE)).replace('@@CODEQ@@',shlex.quote(code))


COMMON='''import base64,fcntl,gzip,hashlib,io,json,os,pathlib,subprocess,tarfile,time
b=pathlib.Path(%r)
def read(p):return json.loads(pathlib.Path(p).read_text())
def fp(p):
 p=pathlib.Path(p);h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(8*1024**2),b''):h.update(x)
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())
def atomic(p,v):
 p=pathlib.Path(p);t=p.with_name(p.name+'.partial');t.write_text(json.dumps(v,sort_keys=True));t.rename(p)
def command(cmd):
 r=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=20);return dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
def lock():
 b.mkdir(parents=True,exist_ok=True);g=(b/'operator.lock').open('a');fcntl.flock(g,fcntl.LOCK_EX|fcntl.LOCK_NB);return g
def published():
 p=read(b/'source-publication.json');root=pathlib.Path(p['code_root'])
 for n,v in p['files'].items():
  f=root/n;assert f.stat().st_size==v['bytes'] and fp(f)['sha256']==v['sha256'],n
 return p,root
'''%BASE


def main():
    p=argparse.ArgumentParser();p.add_argument('--action',choices=('prepare','submit','status'),required=True);p.add_argument('--control-root',type=Path,required=True);p.add_argument('--submit-authorized',action='store_true');a=p.parse_args();repo=Path(__file__).resolve().parents[3]
    spec=importlib.util.spec_from_file_location('exact_all252_control',a.control_root/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl);ip=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0']).decode().strip()
    def call(s):
        command='python3 -c '+shlex.quote(COMMON+s)
        if len(command.encode())>=16000:raise ValueError('Code-control command exceeds16KB')
        r=ctl.run(command,account='mou176',source_address=ip,timeout=55);rows=[json.loads(x) for x in r.stdout.decode(errors='replace').splitlines() if x.startswith('{')]
        if r.returncode or not rows:raise RuntimeError('Exact-all252 action unconfirmed: '+r.stderr.decode(errors='replace')[-1500:]+' '+r.stdout.decode(errors='replace')[-2500:])
        return rows[-1]
    if a.action=='prepare':
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo).decode().strip();remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/codex/qwen7b-probe-warmup-20261019'],cwd=repo).split()[0].decode();assert commit==remote
        files={};buffer=io.BytesIO()
        with gzip.GzipFile(fileobj=buffer,mode='wb',mtime=0,filename='') as gz:
            with tarfile.open(fileobj=gz,mode='w') as tar:
                for n in closure(repo):
                    raw=(repo/n).read_bytes();assert raw==subprocess.check_output(['git','show',commit+':'+n],cwd=repo),n;files[n]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest());m=tarfile.TarInfo(n);m.size=len(raw);m.mode=0o444;tar.addfile(m,io.BytesIO(raw))
        data=buffer.getvalue();pub=dict(commit=commit,remote_sha=commit,code_root=BASE+'/code-'+commit[:12],files=files,bundle_bytes=len(data),bundle_sha256=hashlib.sha256(data).hexdigest());metadata=json.dumps(pub,sort_keys=True).encode()
        for kind,payload in (('publication',metadata),('code',data)):
            for index,start in enumerate(range(0,len(payload),4096)):
                part=payload[start:start+4096];s="guard=lock();p=b/%r;data=base64.b64decode(%r);assert hashlib.sha256(data).hexdigest()==%r\nif p.exists():assert p.read_bytes()==data\nelse:p.write_bytes(data)\nprint(json.dumps(dict(status='CODE_CHUNK_STAGED',kind=%r,index=%d)))"%('%s-%s-%05d.chunk'%(kind,commit[:12],index),base64.b64encode(part).decode(),hashlib.sha256(part).hexdigest(),kind,index);call(s)
        s="guard=lock();commit=%r;metadata=b''.join(p.read_bytes() for p in sorted(b.glob('publication-'+commit[:12]+'-*.chunk')));assert len(metadata)==%d and hashlib.sha256(metadata).hexdigest()==%r;pub=json.loads(metadata);data=b''.join(p.read_bytes() for p in sorted(b.glob('code-'+commit[:12]+'-*.chunk')));assert len(data)==pub['bundle_bytes'] and hashlib.sha256(data).hexdigest()==pub['bundle_sha256']\n"%(commit,len(metadata),hashlib.sha256(metadata).hexdigest())+'''
root=pathlib.Path(pub['code_root']);stage=root.with_name(root.name+'.staging')
if not root.exists():
 stage.mkdir(exist_ok=True)
 with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as tar:
  assert {m.name for m in tar.getmembers()}==set(pub['files'])
  for m in tar.getmembers():
   n=pathlib.Path(m.name);assert m.isfile() and not n.is_absolute() and '..' not in n.parts;raw=tar.extractfile(m).read();v=pub['files'][m.name];assert len(raw)==v['bytes'] and hashlib.sha256(raw).hexdigest()==v['sha256'];p=stage/n;p.parent.mkdir(parents=True,exist_ok=True)
   if p.exists():assert p.read_bytes()==raw
   else:p.write_bytes(raw)
   p.chmod(0o444)
 stage.rename(root)
if (b/'source-publication.json').exists():assert read(b/'source-publication.json')==pub
else:atomic(b/'source-publication.json',pub)
published();(b/'logs').mkdir(exist_ok=True);print(json.dumps(dict(status='ALL252_IMMUTABLE_COMPLETE_CLOSURE_STAGED',source_publication=fp(b/'source-publication.json'),files=len(pub['files']))))
''';call(s)
        s="guard=lock();pub,root=published();script="+repr(script(pub['code_root']))+'''
p=b/'fit.sbatch'
if p.exists():assert p.read_text()==script
else:p.write_text(script)
if (b/'preparation.json').exists():print(json.dumps(dict(status='EXISTING_TYPED_PREPARATION_PRESERVED',preparation=read(b/'preparation.json'))))
elif (b/'cpu-prepare-launch.json').exists():print(json.dumps(dict(status='EXISTING_CPU_PREPARATION_LAUNCH_PRESERVED',launch=read(b/'cpu-prepare-launch.json'))))
else:
 assert not (b/'cpu-prepare-intent.json').exists(),'Unknown previous preparation; inspect log, never repeat'
 atomic(b/'cpu-prepare-intent.json',dict(source_publication=fp(b/'source-publication.json'),time=time.time()))
 env=dict(os.environ,QWEN7B_CODE_ROOT=str(root),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
 # One actual immutable subprocess runs the ten focused tests before typed CPU
 # preparation. No GPU, package installation, or checkpoint tensor load.
 args=['bash','-c','set -e; bash "$1" -m unittest reproduction.operations.qwen7b_warmup.test_exact_uniform_g5_loss reproduction.operations.qwen7b_warmup.test_probe_exact_uniform_g5_train reproduction.operations.qwen7b_warmup.test_probe_g10_train.ProbeG10TrainTests.test_accepted_fullstate_save_load_next_update_is_equal -q; exec bash "$1" -u -m reproduction.operations.qwen7b_warmup.prepare_exact_uniform_g5 --base "$2"','all252-cpu',str(root/'reproduction/operations/qwen7b_warmup/runtime.sh'),str(b)]
 with (b/'cpu-prepare.log').open('ab') as log:child=subprocess.Popen(args,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 ticks=pathlib.Path('/proc/%d/stat'%child.pid).read_text().rsplit(')',1)[1].split()[19];value=dict(status='ACTUAL_ALL252_10_TESTS_AND_TYPED_CPU_PREPARATION_STARTED',pid=child.pid,start_ticks=ticks,source_publication=fp(b/'source-publication.json'),time=time.time());atomic(b/'cpu-prepare-launch.json',value);print(json.dumps(value))
''';result=call(s)
    elif a.action=='submit':
        if not a.submit_authorized:p.error('Explicit --submit-authorized required for one new GPU job')
        s='''guard=lock();pub,root=published();done=read(b/'preparation.json');assert done['status']=='ACTUAL_ALL252_E1_SAME_G10_B48X136_TYPED_INPUTS_PREPARED_NOT_SUBMITTED';assert fp(done['config']['path'])==done['config'];c=read(done['config']['path']);assert c['explicit_launch_authorized'] is True and c['global_questions']==48 and c['update_budget']==136 and c['recipe']['loss_kwargs']==dict(trajectory_weight=1.,three_event_weight=1.)
runlock=(b/'exact-g5-train.lock').open('a');fcntl.flock(runlock,fcntl.LOCK_EX|fcntl.LOCK_NB)
if (b/'job.id').exists():print(json.dumps(dict(status='EXISTING_ALL252_JOB_PRESERVED',job_id=(b/'job.id').read_text().strip())))
else:
 assert not (b/'submission-intent.json').exists(),'Unknown prior submit outcome; inspect same-name queue/account before retry'
 assert not any((b/n).exists() for n in ('progress.json','latest.json','update-intent.json')),'Existing actual training state preserved'
 q=command(['squeue','-h','-u','mou176','-n','q7b-g10-all252','-o','%i|%j|%T|%R']);assert q['returncode']==0 and not q['stdout'].strip(),q
 intent=dict(status='ALL252_SUBMIT_INTENT',script=fp(b/'fit.sbatch'),config=done['config'],source_publication=done['source_publication'],resources=dict(partition='acd_u',qos='normal',GPU=1,CPU=8,memory='64G',time='02:00:00'),dependency=None,time=time.time());atomic(b/'submission-intent.json',intent)
 try:r=subprocess.run(['sbatch','--hold','--parsable',str(b/'fit.sbatch')],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=20)
 except subprocess.TimeoutExpired:
  atomic(b/'submission-unknown.json',dict(status='UNKNOWN_DO_NOT_REPEAT',time=time.time()));raise
 atomic(b/'actual-submit-response.json',dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr));assert r.returncode==0,r.stderr[-1500:];job=r.stdout.strip().split(';')[0];assert job.isdigit(),'Unknown sbatch response; reconcile'
 with (b/'job.id').open('x') as f:f.write(job+'\\n');f.flush();os.fsync(f.fileno())
 state=command(['squeue','-h','-j',job,'-o','%i|%j|%T|%R']);assert state['returncode']==0 and 'JobHeldUser' in state['stdout'],state;release=command(['scontrol','release',job]);assert release['returncode']==0,release
 value=dict(intent,status='ALL252_ONE_JOB_RELEASED_NOT_PROGRESS_VERIFIED',job_id=job,queue=command(['squeue','-h','-j',job,'-o','%i|%j|%T|%R|%N']));atomic(b/'submission.json',value);print(json.dumps(value))
''';result=call(s)
    else:
        s='''r=dict(status='ACTUAL_READ_ONLY_ALL252_OBSERVATION',time=time.time(),base=str(b))
for name in ('cpu-prepare-launch.json','cpu-prepare-failure.json','progress.json','next-update-replay.json','latest.json','failure.json','interrupted.json','gpu-memory.json','controller.json'):
 p=b/name
 if p.exists():
  value=read(p)
  if name=='controller.json':value={k:value[k] for k in ('pid','job_id','status','source_steps','global_original_questions') if k in value}
  r[name]=value
if (b/'preparation.json').exists():r['preparation']=read(b/'preparation.json')
if (b/'complete.json').exists():r['complete_ref']=fp(b/'complete.json');r['complete']=read(b/'complete.json')
if (b/'job.id').exists():
 job=(b/'job.id').read_text().strip();r['job_id']=job;r['squeue']=command(['squeue','-h','-j',job,'-o','%i|%j|%T|%R|%N']);r['sacct']=command(['sacct','-X','-n','-P','-j',job,'--format=JobID,State,ExitCode,Elapsed,NodeList'])
else:
 p=b/'cpu-prepare.log'
 if p.exists():r['cpu_log_tail']=p.read_text(errors='replace')[-2500:]
if (b/'failure.json').exists():r['GPU_log_tail']={p.name:p.read_text(errors='replace')[-3000:] for p in (b/'logs').glob('*.log')}
print(json.dumps(r))
''';result=call(s)
    print(json.dumps(result),flush=True)

if __name__=='__main__':main()
