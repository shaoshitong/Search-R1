"""Controller: only code/config/credential crosses Mac, no model body."""
import sys,subprocess,json,pathlib,shlex,hashlib,importlib.util,base64,time
D=pathlib.Path('outputs/grounding-eval-20261019');repo=pathlib.Path('work/grounding-eval-strict-20261019');commit=subprocess.check_output(['git','-C',str(repo),'rev-parse','7bb6f705'],text=True).strip();origin=subprocess.check_output(['git','-C',str(repo),'ls-remote','origin','refs/heads/codex/grounding-eval-strict-20261019'],text=True).split()[0];subprocess.run(['git','-C',str(repo),'merge-base','--is-ancestor',commit,origin],check=True)
source=subprocess.check_output(['git','-C',str(repo),'show',commit+':reproduction/grounding_eval/asset_fetch.py']);plan=json.loads((D/'easy-hf-stream-plan.json').read_text());uploaded=json.loads((D/'easy-hf-upload-status-latest.json').read_text())['status'];assert uploaded['state']=='HF_FILES_VERIFIED','Upload pinned metadata verification not complete';revision=uploaded['final_revision'];assert len(uploaded['receipts'])==13;job={'repo':plan['repo'],'revision':revision,'prefix':plan['prefix'],'output':'/data/shaoshitong/grounding-eval-20261019/assets/easy-selection-iter200','files':{r['name']:{'size':r['bytes'],'sha256':r['sha256']}for r in plan['files']},'route':'retained18 -> mgmt RAM -> private HF -> A800 direct HTTPS; per-request HF DNS binding; Mac control metadata only','source_receipt_sha256':hashlib.sha256((D/'easy-hf-upload-status-latest.json').read_bytes()).hexdigest(),'concurrency':3,'receipt_name':'asset-migration-complete.json','resolve':['huggingface.co:443:3.167.112.25']}
(D/'easy-a800-asset-job.json').write_text(json.dumps(job,sort_keys=True,indent=2)+'\n');spec=importlib.util.spec_from_file_location('ctl','/Users/shaoshitong/Documents/remote_codex_control/control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl)
tc="from pathlib import Path;import sys;sys.stdout.write(Path('/mnt/lustre/shaoshitong/hf-checkpoint-upload-20261005/task-token').read_text().strip())";r=subprocess.run(ctl.route('mgmt-host','python3 -B -c '+shlex.quote(tc)),capture_output=True,timeout=25);assert r.returncode==0;token=r.stdout.decode().strip()
code=r'''import sys,json,pathlib,base64,hashlib,os,fcntl,subprocess,time
p=json.load(sys.stdin);base=pathlib.Path('/data/shaoshitong/grounding-eval-20261019');rt=base/('asset-runtime-'+p['commit'][:8]);rt.mkdir(mode=0o700,exist_ok=True);target=pathlib.Path(p['job']['output']);target.mkdir(parents=True,exist_ok=True)
with(target/'.asset-migration.lock').open('a+')as lock:fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
if(target/'asset-status.json').exists():
 old=json.loads((target/'asset-status.json').read_text());pid=old['pid']
 try:os.kill(pid,0)
 except ProcessLookupError:pass
 else:raise RuntimeError('Existing migration PID remains alive; inspect identity before resuming')
def put(q,b):
 if q.exists():assert q.read_bytes()==b,'Immutable deployment differs'
 else:
  with q.open('xb')as f:os.fchmod(f.fileno(),0o600);f.write(b);f.flush();os.fsync(f.fileno())
raw=base64.b64decode(p['source']);assert hashlib.sha256(raw).hexdigest()==p['source_sha256'];entry=rt/'asset_fetch.py';put(entry,raw);put(rt/'asset-job.json',json.dumps(p['job'],sort_keys=True,indent=2).encode()+b'\n')
subprocess.run([sys.executable,'-m','py_compile',str(entry)],check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
log=(target/'asset-fetch.log').open('ab',buffering=0);process=subprocess.Popen([sys.executable,'-u',str(entry),'--job',str(rt/'asset-job.json')],stdin=subprocess.PIPE,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);process.stdin.write(p['token'].encode()+b'\n');process.stdin.close();log.close();time.sleep(.5)
assert process.poll()is None,'Asset process exited on launch; sanitized log must be inspected'
x={'host':os.uname().nodename,'at':time.time(),'pid':process.pid,'entrypoint':str(entry),'source_commit':p['commit'],'source_sha256':hashlib.sha256(entry.read_bytes()).hexdigest(),'job_path':str(rt/'asset-job.json'),'job_sha256':hashlib.sha256((rt/'asset-job.json').read_bytes()).hexdigest(),'target':str(target),'gpu_jobs_submitted':0,'token_transport':'stdin/RAM only','weight_bytes_through_Mac':0};print(json.dumps(x))'''
payload={'commit':commit,'job':job,'source':base64.b64encode(source).decode(),'source_sha256':hashlib.sha256(source).hexdigest(),'token':token};r=subprocess.run(ctl.route('baai-a800','python3 -B -c '+shlex.quote(code)),input=json.dumps(payload),capture_output=True,text=True,timeout=30)
if r.returncode:raise RuntimeError('Asset launch failed; no credentials retained; inspect remote task log')
x=json.loads(r.stdout);(D/'easy-a800-asset-launch.json').write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(x))
