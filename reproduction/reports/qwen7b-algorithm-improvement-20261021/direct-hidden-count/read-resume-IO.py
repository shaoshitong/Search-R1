"""Read only the actual new allocation's owned resume loader CPU/I/O."""
import argparse,importlib.util,json,pathlib,shlex,subprocess
ROOT=pathlib.Path(__file__).resolve().parents[4];CONTROL=ROOT.parents[1];OUT=pathlib.Path(__file__).parent
p=argparse.ArgumentParser();p.add_argument('--job-id',required=True);args=p.parse_args()
if not args.job_id.isdigit() or args.job_id=='743619':p.error('Actual separate continuation job required')
code=r'''import pathlib,json,os,time,socket
base=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/direct-hidden-count30016-train-v2');attempt=base/'continuations/attempt-0001-v2'
assert (attempt/'job.id').read_text().strip()==JOB and os.environ['SLURM_JOB_ID']==JOB
pub=json.loads((base/'source-publication.json').read_text());op=json.loads((attempt/'operator-publication.json').read_text());assert pub['commit']=='501cf80478b77f68c3e13a96d4659e3da0971b2e' and op['commit']=='31e930ce8327078612c3ee9dd98aec2188a78826'
processes=[]
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit() or int(p.name)==os.getpid():continue
 try:
  if p.stat().st_uid!=os.getuid():continue
  cmd=(p/'cmdline').read_bytes();env=(p/'environ').read_bytes()
  if b'external_joint_resume' not in cmd or b'direct_hidden_resume.py' not in cmd:continue
  if b'SLURM_JOB_ID='+JOB.encode()+b'\x00' not in env or b'QWEN7B_CODE_ROOT='+pub['code_root'].encode()+b'\x00' not in env:continue
  io={k:int(v.strip()) for k,v in [line.split(':',1) for line in (p/'io').read_text().splitlines()]};st=(p/'stat').read_text().split();rss=[r for r in (p/'status').read_text().splitlines() if r.startswith('VmRSS:')]
  processes.append(dict(pid=int(p.name),rchar=io['rchar'],read_bytes=io['read_bytes'],user_ticks=int(st[13]),system_ticks=int(st[14]),RSS=rss[0] if rss else None))
 except (OSError,ValueError):pass
progress=json.loads((base/'progress.json').read_text());controller=json.loads((attempt/'controller.json').read_text()) if (attempt/'controller.json').exists() else None
print(json.dumps(dict(job_id=JOB,original_job_id='743619',node=socket.gethostname(),time=time.time(),actual_resume_processes=processes,accepted_cursor=progress['cursor'],resume_controller=controller,failure_exists=(attempt/'failure.json').exists() or (base/'failure.json').exists(),read_only=True)))
'''
spec=importlib.util.spec_from_file_location('joint_actual_resume_IO_control',CONTROL/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl)
command='srun --jobid='+args.job_id+' --overlap --gres=gpu:0 --ntasks=1 --cpus-per-task=1 python3 -c '+shlex.quote('JOB='+repr(args.job_id)+'\n'+code)
r=ctl.run(command,account='mou176',source_address=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0']).decode().strip(),timeout=35)
if r.returncode:raise RuntimeError(r.stderr[-1800:])
value=json.loads(r.stdout)
with (OUT/'actual-resume-input-IO.jsonl').open('a') as f:f.write(json.dumps(value,sort_keys=True)+'\n')
print(json.dumps(value))
