"""Optional guarded CPU audit while the immutable four-GPU job is pending.

The unavailable debug audit never started. Reuse its separate proof path, not
the GPU continuation's canonical proof. All payloads stay on HPC3. The auditor
uses existing mmap/chunked finite validation; an owned supervisor aborts only
its direct child if RSS exceeds 7 GiB or the audit exceeds twenty minutes.
This guard never affects training or changes its acceptance rules.
"""
import base64
import hashlib
import json
from pathlib import Path
import shlex
import subprocess

from reproduction.hpc3_prefix5.control import BASE, OUTPUT, PYTHON
from reproduction.hpc3_prefix5.direct_control import run


SUPERVISOR = r'''import fcntl,hashlib,json,os,pathlib,subprocess,time
b=pathlib.Path(BASE+'-r2');p=b/'preparation';source=pathlib.Path(BASE)
lock=(p/'cp9-login-cpu.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
proof=source/'preparation/cp9-independent-cpu-prequalification.json'
assert not proof.exists() and not proof.with_name(proof.name+'.intent.json').exists()
spec=json.loads((p/'resume-config-template.json').read_text())['resume_joint']
assert spec['completed_outer']==9
program="""import json,pathlib
from reproduction.prefix_selection200.i1_i6_checkpoint_cpu import qualify
s=json.loads(pathlib.Path(TEMPLATE).read_text())['resume_joint']
v=qualify(pathlib.Path(SOURCE),9,pathlib.Path(s['terminal_proof']),s['terminal_proof_sha256'],pathlib.Path(PROOF))
print(json.dumps({k:v.get(k) for k in ('status','completed_outer','actual_adam_steps_per_rank','probe_adam_steps_per_rank','actor_rank_optimizer_rng_scheduler_verified','payload_full_sha_read_once','actual_cpu_deserialization','cuda_restore_verified','completed_unix')}),flush=True)
""".replace('TEMPLATE',repr(str(p/'resume-config-template.json'))).replace('SOURCE',repr(str(source))).replace('PROOF',repr(str(proof)))
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',HF_HOME=str(b/'cache/hf'),XDG_CACHE_HOME=str(b/'cache'),TMPDIR=str(b/'tmp'))
log=b/'logs/cp9-login-cpu.log';controller=p/'cp9-login-cpu-controller.json';progress=p/'cp9-login-cpu-progress.json';done=p/'cp9-login-cpu-result.json'
def write(path,value):
 tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(value));os.replace(tmp,path)
started=time.time();peak=0;cause=None
with log.open('xb') as stream:
 child=subprocess.Popen([str(b/'env/bin/python'),'-u','-c',program],cwd=b/'repo',env=env,stdout=stream,stderr=subprocess.STDOUT)
 write(controller,dict(supervisor_pid=os.getpid(),child_pid=child.pid,started_unix=started,source_outer=9,proof=str(proof),rss_limit_bytes=7*1024**3,timeout_seconds=1200,cuda_visible_devices='',gpu_job_unchanged=729624))
 while child.poll() is None:
  try:
   status=pathlib.Path('/proc/%d/status'%child.pid).read_text();rss=int(next(x.split()[1] for x in status.splitlines() if x.startswith('VmRSS:')))*1024
  except (FileNotFoundError,StopIteration):rss=0
  peak=max(peak,rss)
  write(progress,dict(epoch=time.time(),child_pid=child.pid,rss_bytes=rss,peak_rss_bytes=peak,elapsed_seconds=time.time()-started,log_bytes=log.stat().st_size,proof_exists=proof.exists()))
  if rss>7*1024**3 or time.time()-started>1200:
   cause='OWN_CPU_CHILD_RSS_LIMIT' if rss>7*1024**3 else 'OWN_CPU_CHILD_TIME_LIMIT'
   child.terminate()
   try:child.wait(timeout=10)
   except subprocess.TimeoutExpired:child.kill();child.wait()
   break
  time.sleep(1)
 result=dict(completed_unix=time.time(),elapsed_seconds=time.time()-started,returncode=child.wait(),termination_cause=cause,peak_rss_bytes=peak,proof_exists=proof.exists(),gpu_job_unchanged=729624)
 if proof.exists():
  v=json.loads(proof.read_text());result['qualification']={k:v.get(k) for k in ('status','completed_outer','actual_adam_steps_per_rank','probe_adam_steps_per_rank','actor_rank_optimizer_rng_scheduler_verified','payload_full_sha_read_once','actual_cpu_deserialization','cuda_restore_verified')};result['proof_sha256']=hashlib.sha256(proof.read_bytes()).hexdigest()
 write(done,result)
'''


def main():
    source = SUPERVISOR.replace('BASE', repr(BASE)).encode()
    digest = hashlib.sha256(source).hexdigest()
    code = '''import base64,hashlib,json,os,pathlib,subprocess,time
b=pathlib.Path(TARGET);p=b/'preparation';intent=p/'cp9-login-cpu-start-intent.json'
assert not intent.exists(),'Reconcile existing audit before another launch'
q=subprocess.check_output(['squeue','-h','-j','729624','-o','%i|%T'],text=True).strip();assert q=='729624|PENDING',q
assert 'CANCELLED' in subprocess.check_output(['sacct','-nP','-j','729652','--format=State'],text=True)
mem=pathlib.Path('/sys/fs/cgroup/memory/users/mou176');limit=int((mem/'memory.limit_in_bytes').read_text());usage=int((mem/'memory.usage_in_bytes').read_text());assert limit>=8*1024**3 and usage<512*1024**2,(limit,usage)
script=p/'cp9-login-cpu-supervisor.py';payload=base64.b64decode(PAYLOAD);assert hashlib.sha256(payload).hexdigest()==DIGEST
with script.open('xb') as f:f.write(payload)
with intent.open('x') as f:json.dump(dict(created_unix=time.time(),source_sha256=DIGEST,memory_limit=limit,memory_usage_before=usage),f)
with (b/'logs/cp9-login-cpu-supervisor.log').open('xb') as log:
 child=subprocess.Popen([PYTHON,str(script)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,close_fds=True)
print(json.dumps(dict(supervisor_pid=child.pid,source_sha256=DIGEST,script=str(script),proof=SOURCE+'/preparation/cp9-independent-cpu-prequalification.json',memory_limit=limit,memory_usage_before=usage,gpu_continuation_unchanged=True,started_unix=time.time())))
'''.replace('TARGET', repr(BASE + '-r2')).replace('SOURCE', repr(BASE)).replace('PAYLOAD', repr(base64.b64encode(source).decode())).replace('DIGEST', repr(digest)).replace('PYTHON', repr(PYTHON))
    address = subprocess.check_output(['ipconfig', 'getifaddr', 'en0'], text=True).strip()
    result = run(PYTHON + ' -c ' + shlex.quote(code), account='mou176', source_address=address, timeout=25)
    if result.returncode:
        raise RuntimeError(result.stdout.decode()[-3000:] + result.stderr.decode()[-1000:])
    value = next(json.loads(line) for line in result.stdout.decode().splitlines() if line.strip().startswith('{'))
    (Path(OUTPUT) / 'cp9-login-cpu-start.json').write_text(json.dumps(value, indent=2) + '\n')
    print(json.dumps(value))


if __name__ == '__main__':
    main()
