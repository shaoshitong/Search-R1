import os,pathlib,json,subprocess,time,fcntl,signal,hashlib
base=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/coarse-event-cached-prepare-code-v1')
code=base/'code-149ab8a7399c'
uid=os.getuid();current={os.getpid(),os.getppid()};found=[]
for item in pathlib.Path('/proc').iterdir():
 if not item.name.isdigit()or int(item.name)in current:continue
 try:
  if item.stat().st_uid!=uid:continue
  args=(item/'cmdline').read_bytes().split(b'\0')
  if not args or b'python'not in pathlib.Path(args[0].decode(errors='replace')).name.encode():continue
  command=b' '.join(args)
  if b'test_coarse_event_cached_train'not in command or b'unittest'not in command:continue
  own_root=next((x.split(b'=',1)[1]for x in (item/'environ').read_bytes().split(b'\0')if x.startswith(b'QWEN7B_CODE_ROOT=')),None)
  if own_root!=str(code).encode():continue
  stat=(item/'stat').read_text().split();found.append(dict(pid=int(item.name),state=stat[2],birth=stat[21]))
 except (OSError,StopIteration):pass
if found:
 print(json.dumps(dict(status='SKIPPED_MATCHING_PRIOR_SYNTHETIC_WORKER_STILL_EXISTS',workers=found)));raise SystemExit(0)
producer_lock=(base/'operator.lock').open('r')
try:fcntl.flock(producer_lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
except BlockingIOError:
 print(json.dumps(dict(status='SKIPPED_PRODUCER_CODE_QUALIFIER_LOCK_BUSY')));raise SystemExit(0)
fcntl.flock(producer_lock,fcntl.LOCK_UN);producer_lock.close()
lock=(base/'first-synthetic-stack-physical3G.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
log=base/'first-synthetic-stack-physical3G.log';intent=base/'first-synthetic-stack-physical3G-intent.json';result_path=base/'first-synthetic-stack-physical3G-complete.json'
if intent.exists():
 print(json.dumps(dict(status='EXISTING_SINGLE_DEBUG_ATTEMPT_PRESERVED',intent=str(intent),log=str(log),result=str(result_path),log_tail=log.read_text()[-7000:]if log.exists()else None)));raise SystemExit(0)
job=subprocess.check_output(['scontrol','show','job','-o','743978'],universal_newlines=True)
assert 'JobState=RUNNING' in job and 'UserId=mou176(21930)' in job and 'JobName=q7b-frozenE1-head30016' in job
fixture="""import os,resource,faulthandler,unittest,json,time
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2]);resource.setrlimit(resource.RLIMIT_CPU,(60,60))
faulthandler.enable();faulthandler.dump_traceback_later(10,repeat=True)
started=time.time()
from reproduction.operations.qwen7b_warmup.test_coarse_event_cached_train import Tests
try:
 r=unittest.TextTestRunner(verbosity=2).run(unittest.TestSuite([Tests('test_actual_small_CP0_restore_replay_save_reload_next_update')]))
 print(json.dumps(dict(status='ONE_SYNTHETIC_TEST_FINISHED',tests=r.testsRun,success=r.wasSuccessful(),seconds=time.time()-started)),flush=True)
finally:faulthandler.cancel_dump_traceback_later()
"""
env=dict(os.environ,QWEN7B_CODE_ROOT=str(code),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
started=time.time();intent.write_text(json.dumps(dict(epoch=started,code=str(code),test='test_actual_small_CP0_restore_replay_save_reload_next_update',CPU=2,address_space_limit_not_applied=True,physical_step_memory_bytes=3*1024**3,wall_timeout_seconds=33,no_matching_prior_worker=True,producer_lock_was_free=True))+'\n')
with log.open('w')as stream:
 child=subprocess.Popen(['srun','--jobid=743978','--overlap','--gres=gpu:0','--ntasks=1','--cpus-per-task=2','--mem=3G','bash',str(code/'reproduction/operations/qwen7b_warmup/runtime.sh'),'-u','-c',fixture],stdout=stream,stderr=subprocess.STDOUT,env=env,start_new_session=True)
 timed_out=False
 try:returncode=child.wait(timeout=33)
 except subprocess.TimeoutExpired:
  timed_out=True;os.killpg(child.pid,signal.SIGTERM)
  try:returncode=child.wait(timeout=1)
  except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);returncode=child.wait(timeout=1)
text=log.read_text();offset=text.find('Timeout (0:00:10)!');first=text[offset:]if offset>=0 else text
if offset>=0:
 stop=first.find('Timeout (0:00:10)!',1)
 if stop>=0:first=first[:stop]
out=dict(status='BOUNDED_ONE_SYNTHETIC_STACK_DIAGNOSIS_FINISHED',seconds=time.time()-started,timed_out=timed_out,returncode=returncode,log=dict(path=str(log),bytes=log.stat().st_size,sha256=hashlib.sha256(log.read_bytes()).hexdigest()),first_actual_stack=first[:9000],CPU=2,GPU=0,real_data_or_model_loaded=False,source_modified=False)
result_path.write_text(json.dumps(out,sort_keys=True)+'\n');print(json.dumps(out))
