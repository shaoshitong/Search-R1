"""One read-only owned-node IO sample; never reads tensor or credentials."""
import importlib.util,json,pathlib,shlex,subprocess
ROOT=pathlib.Path(__file__).resolve().parents[4];CONTROL=ROOT.parents[1];HERE=pathlib.Path(__file__).parent
spec=importlib.util.spec_from_file_location('matched_headlr_owned_process_private',CONTROL/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl)
code='''import pathlib,json,time,socket
out=dict(epoch=time.time(),node=socket.gethostname(),processes=[])
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  argv=(p/'cmdline').read_bytes()
  if 'python'not in(p/'exe').resolve().name or b'reproduction.operations.qwen7b_warmup.direct_count_headlr_train'not in argv or b'--action\\x00train'not in argv or b'SLURM_JOB_ID=743809\\x00'not in(p/'environ').read_bytes():continue
  stats=(p/'stat').read_text().split();io=dict((k,int(v.strip()))for k,v in[x.split(':',1)for x in(p/'io').read_text().splitlines()]);row=dict(pid=int(p.name),rchar=io['rchar'],read_bytes=io['read_bytes'],user_ticks=int(stats[13]),system_ticks=int(stats[14]),RSS=[x for x in(p/'status').read_text().splitlines()if x.startswith('VmRSS:')],owned_open_files=[])
  for fd in(p/'fd').iterdir():
   try:
    name=str(fd.resolve())
    if name.startswith('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/')and len(row['owned_open_files'])<6:row['owned_open_files'].append(name)
   except OSError:pass
  out['processes'].append(row)
 except(OSError,ValueError):pass
print(json.dumps(out))'''
r=ctl.run('srun --jobid=743809 --overlap --gres=gpu:0 --ntasks=1 --cpus-per-task=1 python3 -c '+shlex.quote(code),account='mou176',source_address=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0']).decode().strip(),timeout=30)
if r.returncode:raise RuntimeError('Read-only sample unavailable; do not infer training failure')
v=json.loads(r.stdout.decode())
with(HERE/'actual-initialization-IO.jsonl').open('a')as f:f.write(json.dumps(v)+'\n')
print(json.dumps(v))
