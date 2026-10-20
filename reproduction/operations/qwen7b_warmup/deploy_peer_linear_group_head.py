"""Code-only immutable staging and guarded debug CPU0GPU linear-head job."""
import argparse,base64,gzip,hashlib,importlib.util,io,json,shlex,subprocess,tarfile
from pathlib import Path
BASE='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/peer8192-linear-group-head-20261021'
JOB='q7b-peer8192-linear-cpu'
P='reproduction/operations/qwen7b_warmup/'
FILES=tuple(P+n for n in ('peer_linear_group_head.py','test_peer_linear_group_head.py','deploy_peer_linear_group_head.py','replay_feature_reader.py','replay_probe_layers.py','peer_g10_learnability.py','g10_grouping.py','evaluate_frozen_validation.py','runtime.sh'))
BRANCH='refs/heads/codex/qwen7b-probe-warmup-20261019'


def script(base,code):
    return '''#!/usr/bin/env bash
#SBATCH --job-name=@@JOB@@
#SBATCH --account=mou176
#SBATCH --partition=debug
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --gres=gpu:0
#SBATCH --time=00:30:00
#SBATCH --output=@@BASE@@/logs/analysis-%j.log
set -euo pipefail
export QWEN7B_CODE_ROOT=@@CODEQ@@
export CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=12 MKL_NUM_THREADS=12 OPENBLAS_NUM_THREADS=12
export PYTHONDONTWRITEBYTECODE=1
exec 9>@@BASEQ@@/allocation.lock
flock -n 9
python3 - @@CODEQ@@ <<'VERIFY'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]);pub=json.loads((root.parent/'source-publication.json').read_text());assert pub['code_root']==str(root)
for n,v in pub['files'].items():
 p=root/n;assert p.stat().st_size==v['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==v['sha256']
VERIFY
srun --kill-on-bad-exit=1 --gres=gpu:0 bash "$QWEN7B_CODE_ROOT/reproduction/operations/qwen7b_warmup/runtime.sh" -u -m reproduction.operations.qwen7b_warmup.peer_linear_group_head --config @@BASEQ@@/config.json --output @@BASEQ@@/output
'''.replace('@@JOB@@',JOB).replace('@@BASE@@',base).replace('@@BASEQ@@',shlex.quote(base)).replace('@@CODEQ@@',shlex.quote(code))


def main():
    a=argparse.ArgumentParser();a.add_argument('--action',choices=('prepare','submit','status'),required=True);a.add_argument('--control-root',required=True);args=a.parse_args();repo=Path(__file__).resolve().parents[3];control=Path(args.control_root)
    spec=importlib.util.spec_from_file_location('peer_linear_cpu_private',control/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl);ip=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0']).decode().strip()
    common='import base64,fcntl,gzip,hashlib,io,json,os,pathlib,subprocess,tarfile,time\nb=pathlib.Path(%r)\ndef fp(p):\n p=pathlib.Path(p);data=p.read_bytes();return dict(path=str(p),bytes=len(data),sha256=hashlib.sha256(data).hexdigest())\ndef rd(p):return json.loads(pathlib.Path(p).read_text())\n'%BASE
    def call(s):
        cmd='python3 -c '+shlex.quote(common+s)
        if len(cmd.encode())>=16000:raise ValueError('Code-control command exceeds16KB')
        r=ctl.run(cmd,account='mou176',source_address=ip,timeout=55)
        if r.returncode:raise RuntimeError('Remote CPU operation failed: '+r.stderr.decode(errors='replace')[-1500:]+' '+r.stdout.decode(errors='replace')[-1500:])
        values=[json.loads(l) for l in r.stdout.decode().splitlines() if l.startswith('{')];return values[-1] if values else {}
    lock="b.mkdir(parents=True,exist_ok=True)\ng=(b/'operator.lock').open('a');fcntl.flock(g,fcntl.LOCK_EX|fcntl.LOCK_NB)\n"
    if args.action=='prepare':
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=repo).decode().strip();remote=subprocess.check_output(['git','ls-remote','origin',BRANCH],cwd=repo).split()[0].decode();assert commit==remote
        stream=io.BytesIO();files={}
        with gzip.GzipFile(fileobj=stream,mode='wb',mtime=0,filename='') as gz:
            with tarfile.open(fileobj=gz,mode='w') as tar:
                for name in FILES:
                    data=(repo/name).read_bytes();assert data==subprocess.check_output(['git','show',commit+':'+name],cwd=repo);files[name]=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest());item=tarfile.TarInfo(name);item.size=len(data);tar.addfile(item,io.BytesIO(data))
        data=stream.getvalue();code=BASE+'/code-'+commit[:12];pub=dict(commit=commit,remote_sha=commit,code_root=code,files=files,bundle_bytes=len(data),bundle_sha256=hashlib.sha256(data).hexdigest())
        for index,start in enumerate(range(0,len(data),4096)):
            part=data[start:start+4096];call(lock+"p=b/%r;data=base64.b64decode(%r);assert hashlib.sha256(data).hexdigest()==%r\nif p.exists():assert p.read_bytes()==data\nelse:p.write_bytes(data)\n"%('code-%s-%05d.chunk'%(commit[:12],index),base64.b64encode(part).decode(),hashlib.sha256(part).hexdigest()))
        bindings=json.loads((control/'outputs/qwen7b-warmup-20261019/reports/peer8192-group-head-input-refs.json').read_text());config=dict(training_input=bindings['training_input'],training_plan=bindings['plan'],validation_complete=dict(path='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/validation-fixedG5-hf-replay-20261021/capture/complete.json',bytes=23003,sha256='54453790ae4f2ed2c1cdd0b87a17b383fe6737d5ada3dba640b3ee78b7605056'),validation_reader_audit=bindings['validation_reader_audit'],E4_baseline_complete=bindings['E4_baseline_child'],E4_predictions=bindings['E4_predictions'])
        s=lock+'pub='+repr(pub)+'\nconfig='+repr(config)+'\nscript='+repr(script(BASE,code))+'''
parts=sorted(b.glob('code-'+pub['commit'][:12]+'-*.chunk'));data=b''.join(p.read_bytes() for p in parts);assert len(data)==pub['bundle_bytes'] and hashlib.sha256(data).hexdigest()==pub['bundle_sha256'];root=pathlib.Path(pub['code_root']);stage=root.with_name(root.name+'.staging')
if not root.exists():
 stage.mkdir(exist_ok=True)
 with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as tar:
  assert {x.name for x in tar.getmembers()}==set(pub['files'])
  for m in tar.getmembers():
   n=pathlib.Path(m.name);assert m.isfile() and not n.is_absolute() and '..' not in n.parts;raw=tar.extractfile(m).read();v=pub['files'][m.name];assert len(raw)==v['bytes'] and hashlib.sha256(raw).hexdigest()==v['sha256'];p=stage/n;p.parent.mkdir(parents=True,exist_ok=True)
   if p.exists():assert p.read_bytes()==raw
   else:p.write_bytes(raw)
   p.chmod(0o444)
 stage.rename(root)
for n,v in pub['files'].items():
 p=root/n;assert p.stat().st_size==v['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==v['sha256']
for name,value in [('source-publication.json',json.dumps(pub,sort_keys=True)+'\\n'),('config.json',json.dumps(config,sort_keys=True)+'\\n'),('analysis.sbatch',script)]:
 p=b/name
 if p.exists():assert p.read_text()==value
 else:p.write_text(value)
(b/'logs').mkdir(exist_ok=True)
env=dict(os.environ,QWEN7B_CODE_ROOT=str(root),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2');r=subprocess.run(['bash',str(root/'reproduction/operations/qwen7b_warmup/runtime.sh'),'-m','unittest','reproduction.operations.qwen7b_warmup.test_peer_linear_group_head'],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True,timeout=50);(b/'actual-cpu-tests.log').write_text(r.stdout);assert r.returncode==0,r.stdout[-1500:]
test=subprocess.run(['sbatch','--test-only',str(b/'analysis.sbatch')],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True,timeout=20);(b/'actual-debug-cpu-plugin-test.log').write_text(test.stdout);assert test.returncode==0,test.stdout[-1500:]
result=dict(status='IMMUTABLE_DIRECT_MIXED_CPU_HEAD_PREPARED_TESTS_AND_CPU0GPU_PLUGIN_PASSED',source_publication=fp(b/'source-publication.json'),config=fp(b/'config.json'),sbatch=fp(b/'analysis.sbatch'),CPU_test=fp(b/'actual-cpu-tests.log'),plugin_test=fp(b/'actual-debug-cpu-plugin-test.log'),GPU_requested=0)
(b/'preparation.json').write_text(json.dumps(result));print(json.dumps(result))
''';print(json.dumps(call(s)))
    elif args.action=='submit':
        s=lock+'''
assert (b/'preparation.json').is_file();p=rd(b/'preparation.json');assert fp(b/'analysis.sbatch')==p['sbatch'] and fp(b/'config.json')==p['config'];name=%r
if (b/'job.id').exists():print(json.dumps(dict(status='EXISTING_CPU_JOB_PRESERVED',job=(b/'job.id').read_text().strip())))
else:
 assert not (b/'submit-intent.json').exists(),'Unknown previous submission outcome; inspect account before retry'
 queue=subprocess.run(['squeue','-h','-u','mou176','-n',name,'-o','%%i|%%j|%%T'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=15);assert queue.returncode==0 and not queue.stdout.strip(),'Existing same-name job prevents duplicate'
 (b/'submit-intent.json').write_text(json.dumps(dict(epoch=time.time(),sbatch=p['sbatch'],GPU_requested=0)))
 result=subprocess.run(['sbatch','--parsable',str(b/'analysis.sbatch')],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=20);(b/'actual-submit.json').write_text(json.dumps(dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)));assert result.returncode==0,result.stderr[-1500:];job=result.stdout.strip().split(';')[0];assert job.isdigit();(b/'job.id').write_text(job+'\\n');print(json.dumps(dict(status='CPU0GPU_DIAGNOSTIC_SUBMITTED_NOT_PROGRESS_VERIFIED',job_id=job,partition='debug',CPUs=12,memory_GB=32,GPU_requested=0)))
'''%JOB;print(json.dumps(call(s)))
    else:
        s='''
r=dict(status='ACTUAL_READ_ONLY_CPU_GROUP_HEAD_DIAGNOSTIC_OBSERVATION',time=time.time())
for n in ('preparation.json','output/progress.json','output/failure.json','output/frozen-before-external.json'):
 p=b/n
 if p.exists():r[n]=rd(p)
p=b/'output/complete.json'
if p.exists():r['complete_ref']=fp(p);r['complete']=rd(p)
if (b/'job.id').exists():
 job=(b/'job.id').read_text().strip();r['job_id']=job
 for name,cmd in [('squeue',['squeue','-h','-j',job,'-o','%i|%T|%R|%N']),('sacct',['sacct','-X','-n','-P','-j',job,'--format=JobID,State,ExitCode,Elapsed,AllocCPUS,ReqMem,AllocTRES'])]:
  v=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=15);r[name]=dict(returncode=v.returncode,stdout=v.stdout,stderr=v.stderr)
 if (b/'output/failure.json').exists():r['log_tail']={p.name:p.read_text(errors='replace')[-2000:] for p in (b/'logs').glob('*.log')}
print(json.dumps(r))
''';print(json.dumps(call(s)))
if __name__=='__main__':main()
