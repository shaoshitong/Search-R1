"""Immutable code-only staging and one bounded CPU2 calibration after real scoring."""
import argparse
import base64
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import shlex
import subprocess
import tarfile

BASE='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/platt-g10-calibration-v1'
SCORES='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/peer8192-frozen-calibration-scores-v2'
P='reproduction/operations/qwen7b_warmup/'
FILES=tuple(P+n for n in ('platt_g10_calibration.py','test_platt_g10_calibration.py',
    'g5_event_scores_dp.py','group_score_ablation.py','probe_ablation_loss.py',
    'deploy_platt_g10_calibration.py','runtime.sh'))
BRANCH='refs/heads/codex/qwen7b-probe-warmup-20261019'


def script(base,code):
    return '''#!/usr/bin/env bash
set -euo pipefail
export QWEN7B_CODE_ROOT=@@CODE@@ CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
export PYTHONDONTWRITEBYTECODE=1
exec 9>@@BASE@@/allocation.lock
flock -n 9
python3 - @@CODE@@ @@BASE@@ <<'VERIFY'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]);b=pathlib.Path(sys.argv[2]);pub=json.loads((b/'source-publication.json').read_text())
assert pub['code_root']==str(root)
for name,v in pub['files'].items():
 p=root/name;assert p.stat().st_size==v['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==v['sha256']
VERIFY
set +e
bash "$QWEN7B_CODE_ROOT/reproduction/operations/qwen7b_warmup/runtime.sh" -u -m reproduction.operations.qwen7b_warmup.platt_g10_calibration @@BASE@@/config.json @@BASE@@/output
result=$?
python3 - @@BASE@@ "$result" <<'EXIT'
import json,pathlib,sys,time
b=pathlib.Path(sys.argv[1]);p=b/'exit.json';t=p.with_suffix('.tmp');t.write_text(json.dumps(dict(returncode=int(sys.argv[2]),epoch=time.time())));t.replace(p)
EXIT
exit "$result"
'''.replace('@@BASE@@',shlex.quote(base)).replace('@@CODE@@',shlex.quote(code))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--action',choices=('prepare','launch','status'),required=True)
    ap.add_argument('--control-root',type=Path,required=True);args=ap.parse_args();repo=Path(__file__).resolve().parents[3]
    spec=importlib.util.spec_from_file_location('private_calibration_control',args.control_root/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py')
    ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl)
    ip=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0']).decode().strip()
    common='''import base64,fcntl,gzip,hashlib,io,json,os,pathlib,subprocess,tarfile,time
b=pathlib.Path(%r);sc=pathlib.Path(%r)
def read(p):return json.loads(pathlib.Path(p).read_text())
def fp(p):
 p=pathlib.Path(p);data=p.read_bytes();return dict(path=str(p),bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
def checked(v):
 p=pathlib.Path(v['path']);assert fp(p)==v;return p
def atomic(p,v):
 p=pathlib.Path(p);t=p.with_suffix('.tmp');t.write_text(json.dumps(v,sort_keys=True)+'\\n');t.replace(p)
def guard():
 b.mkdir(parents=True,exist_ok=True);h=(b/'operator.lock').open('a');fcntl.flock(h,fcntl.LOCK_EX|fcntl.LOCK_NB);return h
'''%(BASE,SCORES)

    def call(source):
        cmd='python3 -c '+shlex.quote(common+source)
        if len(cmd.encode())>=16000:raise ValueError('Control command must remain below16KB')
        r=ctl.run(cmd,account='mou176',source_address=ip,timeout=55)
        if r.returncode:raise RuntimeError(r.stderr.decode(errors='replace')[-1500:]+' '+r.stdout.decode(errors='replace')[-1500:])
        lines=[json.loads(s) for s in r.stdout.decode().splitlines() if s.startswith('{')]
        return lines[-1] if lines else {}

    if args.action=='prepare':
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo).decode().strip()
        remote=subprocess.check_output(['git','ls-remote','origin',BRANCH],cwd=repo).split()[0].decode();assert commit==remote
        stream=io.BytesIO();files={}
        with gzip.GzipFile(fileobj=stream,mode='wb',mtime=0,filename='') as gz:
            with tarfile.open(fileobj=gz,mode='w') as tar:
                for name in FILES:
                    data=(repo/name).read_bytes();assert data==subprocess.check_output(['git','show',commit+':'+name],cwd=repo)
                    files[name]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
                    item=tarfile.TarInfo(name);item.size=len(data);tar.addfile(item,io.BytesIO(data))
        data=stream.getvalue();code=BASE+'/code-'+commit[:12]
        pub=dict(commit=commit,remote_sha=commit,code_root=code,files=files,bundle_bytes=len(data),bundle_sha256=hashlib.sha256(data).hexdigest())
        for i,start in enumerate(range(0,len(data),4096)):
            part=data[start:start+4096]
            call('lock=guard()\np=b/%r;data=base64.b64decode(%r)\nif p.exists():assert p.read_bytes()==data\nelse:p.write_bytes(data)\n'%('code-%s-%05d.chunk'%(commit[:12],i),base64.b64encode(part).decode()))
        source='lock=guard()\npub='+repr(pub)+'\nrunner='+repr(script(BASE,code))+'''
parts=sorted(b.glob('code-'+pub['commit'][:12]+'-*.chunk'));data=b''.join(p.read_bytes() for p in parts)
assert len(data)==pub['bundle_bytes'] and hashlib.sha256(data).hexdigest()==pub['bundle_sha256']
root=pathlib.Path(pub['code_root']);stage=root.with_name(root.name+'.staging')
if not root.exists():
 stage.mkdir(exist_ok=True)
 with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as tar:
  assert {m.name for m in tar.getmembers()}==set(pub['files'])
  for m in tar.getmembers():
   n=pathlib.Path(m.name);assert m.isfile() and not n.is_absolute() and '..' not in n.parts
   raw=tar.extractfile(m).read();v=pub['files'][m.name];assert len(raw)==v['bytes'] and hashlib.sha256(raw).hexdigest()==v['sha256']
   p=stage/n;p.parent.mkdir(parents=True,exist_ok=True)
   if p.exists():assert p.read_bytes()==raw
   else:p.write_bytes(raw)
   p.chmod(0o444)
 stage.rename(root)
for n,v in pub['files'].items():assert fp(root/n)==dict(path=str(root/n),**v)
for name,value in [('source-publication.json',json.dumps(pub,sort_keys=True)+'\\n'),('analysis.sh',runner)]:
 p=b/name
 if p.exists():assert p.read_text()==value
 else:p.write_text(value)
env=dict(os.environ,QWEN7B_CODE_ROOT=str(root),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
r=subprocess.run(['bash',str(root/'reproduction/operations/qwen7b_warmup/runtime.sh'),'-m','unittest','reproduction.operations.qwen7b_warmup.test_platt_g10_calibration'],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True,timeout=50)
(b/'actual-cpu-tests.log').write_text(r.stdout);assert r.returncode==0,r.stdout[-2000:]
r=subprocess.run(['bash','-n',str(b/'analysis.sh')],stdout=subprocess.PIPE,stderr=subprocess.PIPE);assert r.returncode==0
result=dict(status='CALIBRATION_IMMUTABLE_SOURCE_9_CPU_TESTS_PASSED_INPUTS_NOT_FIT',source_publication=fp(b/'source-publication.json'),script=fp(b/'analysis.sh'),CPU_tests=fp(b/'actual-cpu-tests.log'),training_fitted=False,GPUs=0)
atomic(b/'preparation.json',result);print(json.dumps(result))
'''
        print(json.dumps(call(source)))
    elif args.action=='launch':
        source='''lock=guard()
p=read(b/'preparation.json');checked(p['source_publication']);checked(p['script']);pub=read(b/'source-publication.json')
for n,v in pub['files'].items():checked(dict(path=str(pathlib.Path(pub['code_root'])/n),**v))
inputs=sc/'CPU-compatible-inputs.json'
if not inputs.exists():
 r=subprocess.run(['squeue','-h','-j','743416','-o','%i|%T|%R'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True)
 print(json.dumps(dict(status='WAITING_ACTUAL_SCORER_COMPLETE_NO_CPU_START',scoring_job='743416',squeue=r.stdout.strip(),error=r.stderr.strip(),CPU_started=False)))
elif (b/'launch.json').exists():print(json.dumps(dict(status='EXISTING_CALIBRATION_LAUNCH_PRESERVED',launch=read(b/'launch.json'))))
else:
 assert not (b/'launch-intent.json').exists(),'Unknown launch outcome; inspect existing PID before retry'
 config=read(inputs);assert set(s['name'] for s in config['sources'])=={'E4','peer256'}
 checked(config['groups']);complete=read(checked(config['scoring_complete']))
 assert complete['status']=='FROZEN_E4_AND_PEER256_REAL_G10_CALIBRATION_SCORES_COMPLETE'
 assert complete['normal_exit'] and complete['actor_updates']==0 and complete['probe_updates']==0
 for s in config['sources']:checked(s['predictions'])
 # External references are not opened before the module freezes a candidate.
 atomic(b/'config.json',config);atomic(b/'launch-intent.json',dict(epoch=time.time(),config=fp(b/'config.json'),script=p['script']))
 with (b/'analysis.log').open('ab') as log:child=subprocess.Popen(['bash',str(b/'analysis.sh')],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 birth=pathlib.Path('/proc/%d/stat'%child.pid).read_text().rsplit(')',1)[1].split()[19]
 value=dict(status='CALIBRATION_CPU2_LAUNCHED_NOT_PROGRESS_VERIFIED',pid=child.pid,start_ticks=birth,GPUs=0,epoch=time.time(),config=fp(b/'config.json'),script=p['script'])
 atomic(b/'launch.json',value);print(json.dumps(value))
'''
        print(json.dumps(call(source)))
    else:
        source='''r=dict(epoch=time.time(),status='ACTUAL_CALIBRATION_READ_ONLY_STATUS')
for name in ('launch.json','exit.json','output/controller.json','output/progress.json','output/failure.json','output/isolation.json','output/candidate-freeze.json'):
 p=b/name
 if p.exists():r[name]=read(p)
for name in ('launch.json','output/controller.json'):
 if name in r:
  v=r[name];p=pathlib.Path('/proc/%s/stat'%v['pid']);r[name+'_identity_alive']=p.exists() and p.read_text().rsplit(')',1)[1].split()[19]==str(v['start_ticks'])
p=b/'output/complete.json'
if p.exists():r['complete_ref']=fp(p);r['complete']=read(p)
if (b/'analysis.log').exists():r['log_tail']=(b/'analysis.log').read_text()[-1800:]
print(json.dumps(r))
'''
        print(json.dumps(call(source)))


if __name__=='__main__':main()
