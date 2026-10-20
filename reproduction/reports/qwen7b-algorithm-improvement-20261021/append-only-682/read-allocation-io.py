"""Read only this job's own live training PID IO inside its existing allocation."""
import importlib.util,json,pathlib,shlex,subprocess,time
ROOT=pathlib.Path(__file__).resolve().parents[4];CONTROL=ROOT.parents[1]
spec=importlib.util.spec_from_file_location('append_allocation_io_control',CONTROL/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl)
code='''import pathlib,json,time,socket
out=[]
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  cmd=(p/'cmdline').read_bytes();env=(p/'environ').read_bytes();exe=(p/'exe').resolve().name
  if 'python' not in exe or b'peer_append_train\\x00--action\\x00train' not in cmd or b'SLURM_JOB_ID=743517\\x00' not in env:continue
  io={k:int(v.strip()) for k,v in [x.split(':',1) for x in (p/'io').read_text().splitlines()]};status=(p/'status').read_text();rss=next(x for x in status.splitlines() if x.startswith('VmRSS:'))
  out.append(dict(pid=int(p.name),rchar=io['rchar'],read_bytes=io['read_bytes'],rss=rss,fd_count=len(list((p/'fd').iterdir()))))
 except (OSError,ValueError):pass
print(json.dumps(dict(job_id='743517',node=socket.gethostname(),own_training_processes=out,time=time.time())))'''
command='srun --jobid=743517 --overlap --gres=gpu:0 --ntasks=1 --cpus-per-task=1 python3 -c '+shlex.quote(code)
r=ctl.run(command,account='mou176',source_address=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0']).decode().strip(),timeout=35)
if r.returncode:raise RuntimeError(r.stderr[-1200:])
v=json.loads(r.stdout);out=pathlib.Path(__file__).parent/'actual-allocation-io.jsonl'
with out.open('a') as f:f.write(json.dumps(v,sort_keys=True)+'\n')
print(json.dumps(v,sort_keys=True))
