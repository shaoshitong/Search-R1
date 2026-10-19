"""Authorized retained18->management bounded RAM->private HF, no model data on Mac."""
import sys,json,pathlib,subprocess,shlex,base64,hashlib,time
sys.path.insert(0,'/Users/shaoshitong/Documents/remote_codex_control');import control
D=pathlib.Path('outputs/grounding-eval-20261019');repo=pathlib.Path('work/grounding-eval-strict-20261019');commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','ad09b983'],text=True).strip();remote=subprocess.check_output(['git','-C',str(repo),'ls-remote','origin','refs/heads/codex/grounding-eval-strict-20261019'],text=True).split()[0];subprocess.run(['git','-C',str(repo),'merge-base','--is-ancestor',commit,remote],check=True)
source=subprocess.check_output(['git','-C',str(repo),'show',commit+':reproduction/grounding_eval/upload_h100_hf_stream.py']);plan=json.loads((D/'easy-hf-stream-plan.json').read_text());payload={'source_commit':commit,'source':base64.b64encode(source).decode(),'source_sha256':hashlib.sha256(source).hexdigest(),'plan':plan}
code=r'''import sys,json,base64,hashlib,pathlib,os,subprocess,time,fcntl
p=json.load(sys.stdin);base=pathlib.Path('/mnt/lustre/shaoshitong/grounding-eval-20261019');root=base/'easy-hf-upload';root.mkdir(parents=True,exist_ok=True);runtime=base/('easy-upload-runtime-'+p['source_commit'][:8]);runtime.mkdir(mode=0o700,exist_ok=True)
with(root/'upload.lock').open('a+')as lock:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
if(root/'status.json').exists():
 s=json.loads((root/'status.json').read_text());pid=s.get('pid');cmd=(pathlib.Path('/proc')/str(pid)/'cmdline')
 if cmd.exists():raise RuntimeError('Prior task PID still exists; inspect command before retry')
def put(path,raw):
 if path.exists():assert path.read_bytes()==raw,'Immutable deployment differs'
 else:
  with path.open('xb')as f:os.fchmod(f.fileno(),0o600);f.write(raw);f.flush();os.fsync(f.fileno())
raw=base64.b64decode(p['source']);assert hashlib.sha256(raw).hexdigest()==p['source_sha256'];entry=runtime/'upload_h100_hf_stream.py';put(entry,raw);plan=runtime/'plan.json';put(plan,json.dumps(p['plan'],sort_keys=True,indent=2).encode()+b'\n')
args=[sys.executable,'-u',str(entry),'--plan',str(plan),'--task-dir',str(root),'--route-json','/home/lichen/.claude/projects/-home-lichen/memory/reference_h100_retained18_route_20261008.public.json','--token-file','/mnt/lustre/shaoshitong/hf-checkpoint-upload-20261005/task-token','--upload-threads','3','--execute']
# Check actual installed dependency and exact source before any mutation.
subprocess.run([sys.executable,'-m','py_compile',str(entry)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
log=(root/'worker.log').open('ab',buffering=0);child=subprocess.Popen(args,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close();time.sleep(.5);assert child.poll()is None,'Worker failed on start; inspect sanitized log'
x={'at':time.time(),'host':os.uname().nodename,'pid':child.pid,'source_commit':p['source_commit'],'source_sha256':hashlib.sha256(entry.read_bytes()).hexdigest(),'plan_sha256':hashlib.sha256(plan.read_bytes()).hexdigest(),'entrypoint':str(entry),'task_root':str(root),'repo':'Sutoonq/TMP_MODEL','prefix':p['plan']['prefix'],'source':p['plan']['source_root'],'route':'retained18 -> mgmt bounded RAM -> private HF','model_disk_staging':False,'Mac_model_bytes':0,'upload_threads':3};put(root/'launch.json',json.dumps(x,indent=2).encode()+b'\n');print(json.dumps(x))'''
a=control.route('mgmt-host','python3 -B -c '+shlex.quote(code));a[1:1]=['-o','ServerAliveInterval=5','-o','ServerAliveCountMax=6'];r=subprocess.run(a,input=json.dumps(payload),capture_output=True,text=True,timeout=30);assert r.returncode==0,'Upload launch unavailable; inspect task metadata';x=json.loads(r.stdout);(D/'easy-hf-upload-launch.json').write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(x))
