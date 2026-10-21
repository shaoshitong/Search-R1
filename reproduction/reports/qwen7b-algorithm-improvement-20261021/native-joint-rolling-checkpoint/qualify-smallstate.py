"""Only four synthetic CPU tests inside already-owned745177; never real tensors."""
import base64,hashlib,importlib.util,json,pathlib,shlex,subprocess,zlib
R=pathlib.Path(__file__).resolve().parents[4];C=R.parents[1];HERE=pathlib.Path(__file__).parent;P='reproduction/operations/qwen7b_warmup/'
pub=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip();files={n:(R/P/n).read_text()for n in('native_joint_rolling_checkpoint.py','test_native_joint_rolling_checkpoint.py')}
for n,raw in files.items():
 if subprocess.check_output(['git','show',pub+':'+P+n],cwd=R)!=raw.encode():raise ValueError('Publish exact smallstate sources first')
if subprocess.check_output(['git','ls-remote','origin','refs/heads/codex/qwen7b-probe-warmup-20261019'],cwd=R,text=True).split()[0]!=pub:raise ValueError('Remote publication differs')
s=importlib.util.spec_from_file_location('ctl',C/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(s);s.loader.exec_module(ctl);ip=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0']).decode().strip()
def call(cmd,timeout=55):
 if len(cmd.encode())>=16000:raise ValueError('Oversized command')
 r=ctl.run(cmd,account='mou176',source_address=ip,timeout=timeout)
 if r.returncode:raise RuntimeError(r.stderr.decode(errors='replace')[-2000:]+r.stdout.decode(errors='replace')[-1000:])
 return r.stdout.decode()
RUNNER="import sys,pathlib,importlib.util,unittest,resource,json,os,time\nold='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/joint-balanced-native-E4-qualification-v3/code-919a277f94f1'\nsys.path.insert(0,old)\nimport reproduction.operations.qwen7b_warmup as pkg\nroot=pathlib.Path(__file__).parent\nfor name in ['native_joint_rolling_checkpoint','test_native_joint_rolling_checkpoint']:\n spec=importlib.util.spec_from_file_location('reproduction.operations.qwen7b_warmup.'+name,root/(name+'.py'));m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;setattr(pkg,name,m);spec.loader.exec_module(m)\nassert os.environ.get('CUDA_VISIBLE_DEVICES')==''\nstarted=time.time()\nresult=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(m))\nrss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024\nassert rss<3*1024**3\nsummary=dict(status='SYNTHETIC_CPU_FOUR_TESTS_PASSED'if result.wasSuccessful()else'FAILED',tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),max_RSS_bytes=rss,seconds=time.time()-started,allocation_job_id=os.environ['SLURM_JOB_ID'],CUDA_initialized=False,real_model_or_checkpoint_loaded=False)\n(root/'actual-result.json').write_text(json.dumps(summary))\nprint(json.dumps(summary))\nraise SystemExit(0 if result.wasSuccessful()else 1)\n"
blob=json.dumps(dict(files=files,publication=pub,runner=RUNNER)).encode();encoded=base64.b64encode(zlib.compress(blob)).decode();identity=hashlib.sha256(blob).hexdigest();root='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/native-joint-rolling-smallstate-cpu-v1/code-'+identity[:12]
for index in range(0,len(encoded),4096):
 chunk=encoded[index:index+4096];worker="import pathlib; r=pathlib.Path(%r);r.mkdir(parents=True,exist_ok=True);p=r/%r;v=%r.encode();assert not p.exists()or p.read_bytes()==v;p.write_bytes(v)"%(root,'chunk-%04d'%(index//4096),chunk);call('python3 -c '+shlex.quote(worker))
worker='''import pathlib,json,base64,zlib,hashlib,subprocess,os,time,re
r=pathlib.Path(@@ROOT@@);blob=zlib.decompress(base64.b64decode(b''.join(p.read_bytes()for p in sorted(r.glob('chunk-*')))));assert hashlib.sha256(blob).hexdigest()==@@IDENTITY@@;p=json.loads(blob)
for n,raw in p['files'].items():
 f=r/n
 if f.exists():assert f.read_bytes()==raw.encode()
 else:f.write_text(raw)
old=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/joint-balanced-native-E4-qualification-v3/code-919a277f94f1');runtime=old/'reproduction/operations/qwen7b_warmup/runtime.sh'
runner=r/'run-toy.py';runner.write_text(p['runner'])
# runtime chooses a shared TMPDIR; the final Python wrapper explicitly resets it
# before importing tempfile, so all synthetic temporary tensors stay remote/tmp.
bootstrap="import os,runpy;os.environ['TMPDIR']='/tmp';runpy.run_path("+repr(str(runner))+",run_name='__main__')"
import shlex
name='q7b-rolling-smallcpu-'+@@IDENTITY@@[:8];intent=r/'CPU-submission-intent.json'
if intent.exists():
 print(json.dumps(dict(status='EXISTING_CPU_SUBMISSION_PRESERVED',root=str(r),intent=json.loads(intent.read_bytes()))))
else:
 q=subprocess.run(['squeue','-h','-u','mou176','-n',name,'-o','%i|%T|%j'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True);assert not q.stdout.strip(),'Unknown existing same-name CPU allocation; reconcile'
 receipt=dict(status='CPU_ONLY_SUBMISSION_INTENT',root=str(r),source_publication=p['publication'],job_name=name,cpus=2,memory='8G',GPUs=0,time_limit='00:05:00',epoch=time.time());intent.write_text(json.dumps(receipt))
 wrap=' '.join(shlex.quote(arg)for arg in ['env','CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=2','MKL_NUM_THREADS=2','OPENBLAS_NUM_THREADS=2','QWEN7B_CODE_ROOT='+str(old),'bash',str(runtime),'-c',bootstrap])
 cmd=['sbatch','--parsable','--hold','--partition=acd_u','--qos=normal','--nodes=1','--ntasks=1','--cpus-per-task=2','--gres=gpu:0','--mem=8G','--time=00:05:00','--job-name='+name,'--output='+str(r/'CPU-%j.log'),'--wrap='+wrap]
 x=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=30);assert x.returncode==0,x.stderr;job=x.stdout.strip().split(';')[0];assert job.isdigit();receipt.update(status='CPU_ONLY_JOB_HELD',job_id=job);(r/'CPU-submission.json').write_text(json.dumps(receipt))
 detail=subprocess.check_output(['scontrol','show','job','-o',job],universal_newlines=True);assert('JobName='+name)in detail and'UserId=mou176('in detail and'NumCPUs=2'in detail and'Partition=acd_u'in detail and'JobState=PENDING'in detail and'TimeLimit=00:05:00'in detail and all(int(v)==0 for v in re.findall(r'gres/gpu[=:]([0-9]+)',detail)),detail
 subprocess.check_call(['scontrol','release',job]);receipt.update(status='CPU_ONLY_JOB_RELEASED',job_identity=detail);(r/'CPU-submission.json').write_text(json.dumps(receipt));print(json.dumps(receipt))
'''.replace('@@ROOT@@',repr(root)).replace('@@IDENTITY@@',repr(identity))
result=call('python3 -c '+shlex.quote(worker),timeout=150);v=json.loads(result.splitlines()[-1]);(HERE/'actual-smallstate-CPU-submission.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v))
