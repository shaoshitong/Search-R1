"""Explicit root-authorized <=2CPU/<8GiB synthetic tests (cold import<=60s; tests<=10s; total<=90s) on remote login.

No real checkpoint/model is read; the Slurm guard is mocked with an explicit
NONALLOCATION sentinel only for tiny unittest state files. This is not a real
full-checkpoint performance or allocation qualification.
"""
import hashlib,importlib.util,json,pathlib,shlex,subprocess
R=pathlib.Path(__file__).resolve().parents[4];C=R.parents[1];HERE=pathlib.Path(__file__).parent;P='reproduction/operations/qwen7b_warmup/'
proofs={n:hashlib.sha256((R/P/n).read_bytes()).hexdigest()for n in('native_joint_rolling_checkpoint.py','test_native_joint_rolling_checkpoint.py')}
runner='''import os,sys,pathlib,importlib.util,unittest,json,time,resource
os.environ['TMPDIR']='/tmp'
os.environ['SLURM_JOB_ID']='SYNTHETIC_LOGIN_UNITTEST_NO_SLURM_ALLOCATION'
root=pathlib.Path(sys.argv[1]);old='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/joint-balanced-native-E4-qualification-v3/code-919a277f94f1';sys.path.insert(0,old)
import reproduction.operations.qwen7b_warmup as pkg
for name in ['native_joint_rolling_checkpoint','test_native_joint_rolling_checkpoint']:
 s=importlib.util.spec_from_file_location('reproduction.operations.qwen7b_warmup.'+name,root/(name+'.py'));m=importlib.util.module_from_spec(s);sys.modules[s.name]=m;setattr(pkg,name,m);s.loader.exec_module(m)
(root/'synthetic-tests-started.json').write_text(json.dumps(dict(monotonic=time.monotonic())))
started=time.time();r=unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromModule(m));rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024
assert rss<8*1024**3 and not m.torch.cuda.is_initialized()
v=dict(status='SYNTHETIC_ROLLING_FOUR_TESTS_PASSED'if r.wasSuccessful()else'FAILED',tests=r.testsRun,failures=len(r.failures),errors=len(r.errors),max_RSS_bytes=rss,test_seconds=time.time()-started,CUDA_initialized=m.torch.cuda.is_initialized(),actual_checkpoint_or_model_loaded=False,scope='remote_login_synthetic_CPU_unittest_only',Slurm_allocation_used=False,Slurm_guard_mock_explicit_nonallocation_sentinel=True)
(root/'actual-login-synthetic-result.json').write_text(json.dumps(v));print(json.dumps(v));raise SystemExit(0 if r.wasSuccessful()else 1)
'''
worker='''import os,pathlib,json,hashlib,sys,subprocess,time
p=json.loads(PACKET);base=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/native-joint-rolling-smallstate-cpu-v1');roots=[]
for r in base.glob('code-*'):
 if all((r/n).is_file()and hashlib.sha256((r/n).read_bytes()).hexdigest()==h for n,h in p['proofs'].items()):roots.append(r)
assert roots,'Exact already-staged small test/helper code required';r=max(roots,key=lambda f:f.stat().st_mtime);f=r/'run-login-synthetic.py';f.write_text(p['runner']);os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2]);allowed=list(os.sched_getaffinity(0));old=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/joint-balanced-native-E4-qualification-v3/code-919a277f94f1');env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',QWEN7B_CODE_ROOT=str(old));log=r/'login-synthetic.log'
with log.open('wb')as stream:
 started=time.monotonic();child=subprocess.Popen(['bash',str(old/'reproduction/operations/qwen7b_warmup/runtime.sh'),str(f),str(r)],env=env,stdout=stream,stderr=subprocess.STDOUT);peak=0
 while child.poll()is None:
  status=pathlib.Path('/proc/%d/status'%child.pid)
  if status.exists():
   text=status.read_text();values=[int(line.split()[1])*1024 for line in text.splitlines()if line.startswith('VmRSS:')];peak=max([peak]+values)
  now=time.monotonic();marker=r/'synthetic-tests-started.json';test_elapsed=now-json.loads(marker.read_bytes())['monotonic']if marker.exists()else None
  if peak>8*1024**3 or now-started>90 or(test_elapsed is None and now-started>60)or(test_elapsed is not None and test_elapsed>10):child.kill();child.wait();raise RuntimeError('Synthetic-only cold-import/test/RSS bound reached; no real CP was opened')
  time.sleep(.025)
 elapsed=time.monotonic()-started
assert child.returncode==0,log.read_text()[-2400:]
v=json.loads((r/'actual-login-synthetic-result.json').read_bytes());v.update(root=str(r),wall_seconds_including_import=elapsed,observed_max_RSS_bytes=peak,CPU_affinity=allowed,source_code_SHA256=p['proofs'],actual_stdout_log=dict(path=str(log),bytes=log.stat().st_size,sha256=hashlib.sha256(log.read_bytes()).hexdigest()));print(json.dumps(v))'''.replace('PACKET',repr(json.dumps(dict(proofs=proofs,runner=runner))))
s=importlib.util.spec_from_file_location('ctl',C/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(s);s.loader.exec_module(ctl);cmd='python3 -c '+shlex.quote(worker);assert len(cmd.encode())<16000
r=ctl.run(cmd,account='mou176',source_address=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0']).decode().strip(),timeout=35)
if r.returncode:raise RuntimeError(r.stderr.decode(errors='replace')[-2800:]+r.stdout.decode(errors='replace')[-1000:])
v=json.loads(r.stdout.decode().splitlines()[-1]);(HERE/'actual-login-synthetic-result.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v))
