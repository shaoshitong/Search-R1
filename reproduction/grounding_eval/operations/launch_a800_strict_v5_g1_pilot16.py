"""Deploy/publish a detached G1 strict-v5 pilot16 only, on root-authorized GPU5."""
import argparse,base64,hashlib,importlib.util,json,pathlib,shlex,subprocess
p=argparse.ArgumentParser(__doc__);p.add_argument('--execute',action='store_true');args=p.parse_args()
repo=pathlib.Path(__file__).resolve().parents[3];controller=(pathlib.Path(__file__).parent/'cf_only_pilot_controller.py').read_bytes()
b='/data/shaoshitong/grounding-eval-20261019';normal=b+'/search-g1-normal';job=dict(mode='pilot16',module='reproduction.grounding_eval.strict_v5',method='Search-G1',run_id='db474b6abf654ad9892b02ae67ab6f86',source_commit='16c36aa777f684f245b1d38cb2939ef175131831',python='/data/search-g1/env/bin/python',runtime=b+'/runtime-16c36aa7',source=normal+'/normal.jsonl',checkpoint='/data/shaoshitong/Search-G1-a800-fresh200-a20f0475ec824c69ba1554f39e9ff512/checkpoints/search-g1-common8/actor/global_step_200',normal_config=normal+'/normal-config.json',control_output=b+'/search-g1-cf-strict-v5-pilot16-control',cf_output=b+'/search-g1-cf-strict-v5',gpu_uuid='GPU-21508d9d-bd43-ff94-7bee-1227ec7b007b')
payload=dict(job=job,controller=base64.b64encode(controller).decode(),controller_sha256=hashlib.sha256(controller).hexdigest())
code=r'''
import pathlib,json,sys,base64,hashlib,os,subprocess,time,fcntl
p=json.load(sys.stdin);job=p['job'];runtime=pathlib.Path(job['runtime']);b=runtime.parent;assert b==pathlib.Path('/data/shaoshitong/grounding-eval-20261019')
receipt=json.loads((runtime/'deployment-receipt.json').read_text());assert receipt['source_commit']==job['source_commit']
for rel,v in receipt['files'].items():assert hashlib.sha256((runtime/rel).read_bytes()).hexdigest()==v['sha256']
control=pathlib.Path(job['control_output']);control.mkdir(exist_ok=True);lock=open(control/'launch.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
if(control/'launch.json').exists():print(json.dumps({'state':'EXISTING_INSPECT_ONLY','launch':json.loads((control/'launch.json').read_text())}));raise SystemExit(0)
sys.path.insert(0,str(runtime));from reproduction.grounding_eval.normal_a800 import gpu_lease
fd,index=gpu_lease(job['gpu_uuid']);assert index==5
raw=base64.b64decode(p['controller']);assert hashlib.sha256(raw).hexdigest()==p['controller_sha256'];script=b/('cf-only-pilot-controller-'+p['controller_sha256'][:12]+'.py')
if script.exists():assert script.read_bytes()==raw
else:
 with script.open('xb')as f:os.fchmod(f.fileno(),0o600);f.write(raw);f.flush();os.fsync(f.fileno())
jobpath=b/'search-g1-cf-strict-v5-pilot16.job.json';raw=json.dumps(job,sort_keys=True,indent=2).encode()+b'\n'
if jobpath.exists():assert jobpath.read_bytes()==raw
else:
 with jobpath.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
command=[job['python'],'-B',str(script),'--job',str(jobpath)];os.close(fd)
with(control/'controller.log').open('xb')as log:child=subprocess.Popen(command,cwd=b,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1'),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
launch=dict(pid=child.pid,birth_ticks=(pathlib.Path('/proc')/str(child.pid)/'stat').read_text().split()[21],command=command,job_sha256=hashlib.sha256(raw).hexdigest(),controller_sha256=p['controller_sha256'],source_commit=job['source_commit'],gpu_uuid=job['gpu_uuid'],launched_at=time.time(),mode='pilot16',full_run_authorized=False)
with(control/'launch.json').open('x')as f:json.dump(launch,f);f.flush();os.fsync(f.fileno())
print(json.dumps(launch))
'''
compile(code,'remote-strict-v5-pilot16-launch','exec')
if not args.execute:print(json.dumps(dict(state='PREPARED_PILOT16_ONLY',job=job,controller_sha256=payload['controller_sha256'])));raise SystemExit(0)
s=importlib.util.spec_from_file_location('ctl','/Users/shaoshitong/Documents/remote_codex_control/control.py');ctl=importlib.util.module_from_spec(s);s.loader.exec_module(ctl)
r=subprocess.run(ctl.route('baai-a800','/usr/bin/python3 -c '+shlex.quote(code)),input=json.dumps(payload),capture_output=True,text=True,timeout=40);audit=dict(transport_exit=r.returncode,stderr=r.stderr[-1500:],launch=json.loads(r.stdout.strip().splitlines()[-1])if not r.returncode else r.stdout[-2500:]);pathlib.Path('/Users/shaoshitong/Documents/Codex/2026-10-03/xia/outputs/grounding-eval-20261019/a800-strict-v5-g1-pilot16-launch.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit));raise SystemExit(r.returncode)
