import json,subprocess,time,shlex
from pathlib import Path
base=Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fastpath-probe-20261022/diagnostics/candidate-retrieval-pair-r1-20261023/attempt-r1')
code=r'''
import importlib.metadata,json,os,resource,shutil,time
from pathlib import Path
base=Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fastpath-probe-20261022/diagnostics/candidate-retrieval-pair-r1-20261023/attempt-r1')
target=base/'failure-evidence';target.mkdir(exist_ok=True)
roots=list(Path('/tmp').glob('e7bc-750398-*'));files=[];matches=[]
for root in roots:
 logs=root/'session_latest/logs'
 paths=list(logs.glob('*.err'))+list(logs.glob('raylet.out'))+list(logs.glob('runtime_env_agent.log'))+list(logs.glob('runtime_env_setup*.log'))
 for p in paths:
  if not p.is_file() or p.stat().st_size==0:continue
  raw=p.read_bytes()[-65536:]
  if b'pthread_create failed' in raw:matches.append(str(p))
  (target/p.name).write_bytes(raw);files.append(dict(name=p.name,bytes=len(raw),source_bytes=p.stat().st_size))
record=dict(checked_unix=time.time(),job='750398',failure='Ray worker and runtime-env pthread_create failed; no candidate engine or generation completed',error_log_count=len(matches),error_logs=matches,files=files,ray_version=importlib.metadata.version('ray'),nproc_limit=resource.getrlimit(resource.RLIMIT_NPROC),cpus=len(os.sched_getaffinity(0)),optimizer_writes=0,production_untouched=['750029','745198'])
(target/'summary.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
'''
argv=['srun','--jobid=750398','--overlap','--exact','--ntasks=1','--cpus-per-task=1','--mem=256M','--gres=none','env','CUDA_VISIBLE_DEVICES=','/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fa3-leanmetrics-r1-20261023/env/bin/python','-B','-c',code]
done=subprocess.run(argv,capture_output=True,text=True,timeout=35)
print(json.dumps(dict(phase='PRESERVED_ACTUAL_FAILED_STARTUP',returncode=done.returncode,stdout=done.stdout,stderr=done.stderr)))
if done.returncode:raise SystemExit(done.returncode)
actual=json.loads(done.stdout)
if not actual['error_log_count']:raise RuntimeError('No actual startup failure observed; do not stop a healthy job')
intent=dict(checked_unix=time.time(),action='cancel_only_failed_diagnostic750398',reason=actual['failure'],evidence=str(base/'failure-evidence/summary.json'),no_training_or_service_cancellation=True)
(base/'failure-evidence/cancel-intent.json').write_text(json.dumps(intent,indent=2)+'\n')
r=subprocess.run(['scancel','750398'],capture_output=True,text=True,timeout=10)
(base/'failure-evidence/cancel-result.json').write_text(json.dumps(dict(intent,returncode=r.returncode,stdout=r.stdout,stderr=r.stderr),indent=2)+'\n')
print(json.dumps(dict(phase='FAILED_DIAGNOSTIC_CANCEL_REQUEST',returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)))
