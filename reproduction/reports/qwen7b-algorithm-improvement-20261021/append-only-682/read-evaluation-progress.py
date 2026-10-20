"""Only own milestone output sizes/finished rows and allocation PID read IO."""
import argparse,importlib.util,json,pathlib,shlex,subprocess
ROOT=pathlib.Path(__file__).resolve().parents[4];CONTROL=ROOT.parents[1]
p=argparse.ArgumentParser();p.add_argument('--cursor',type=int,choices=(512,768,938),required=True);a=p.parse_args()
spec=importlib.util.spec_from_file_location('append_evaluation_progress_control',CONTROL/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl)
code='''import pathlib,json,time,socket,gzip
base=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/peer-expanded-append-only-train-v2');out=base/'evaluations'/('cursor-%06d'%CURSOR);files=[]
for p in sorted(out.glob('*')):
 if not p.is_file():continue
 r=dict(name=p.name,bytes=p.stat().st_size,mtime=p.stat().st_mtime)
 if p.name.endswith('.jsonl.gz'):
  with gzip.open(str(p),'rt') as f:r['finished_questions']=sum(1 for line in f if line.strip())
 files.append(r)
processes=[]
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  cmd=(p/'cmdline').read_bytes();env=(p/'environ').read_bytes();exe=(p/'exe').resolve().name
  if 'python' not in exe or b'peer_append_train\\x00--action\\x00train' not in cmd or b'SLURM_JOB_ID=743517\\x00' not in env:continue
  io={k:int(v.strip()) for k,v in [x.split(':',1) for x in (p/'io').read_text().splitlines()]};processes.append(dict(pid=int(p.name),rchar=io['rchar'],read_bytes=io['read_bytes']))
 except (OSError,ValueError):pass
print(json.dumps(dict(job_id='743517',node=socket.gethostname(),cursor=CURSOR,evaluation_output=files,training_process=processes,failure_exists=(base/'failure.json').exists(),time=time.time())))'''
command='srun --jobid=743517 --overlap --gres=gpu:0 --ntasks=1 --cpus-per-task=1 python3 -c '+shlex.quote('CURSOR='+str(a.cursor)+'\n'+code)
r=ctl.run(command,account='mou176',source_address=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0']).decode().strip(),timeout=35)
if r.returncode:raise RuntimeError(r.stderr[-1800:])
v=json.loads(r.stdout)
with (pathlib.Path(__file__).parent/('actual-evaluation-%d-progress.jsonl'%a.cursor)).open('a') as f:f.write(json.dumps(v,sort_keys=True)+'\n')
print(json.dumps(v,sort_keys=True))
