"""Explicit OPSD diagnostic-v4 CF on a fresh idle UUID; never normal/train."""
import argparse,importlib.util,json,pathlib,shlex,subprocess
p=argparse.ArgumentParser(__doc__);p.add_argument('--execute',action='store_true');p.add_argument('--recovery1',action='store_true',help='Preserve failed GPU0 launch and use explicitly authorized reclaimed GPU5');args=p.parse_args()
code=r'''
import pathlib,json,subprocess,os,time,sys,hashlib,fcntl
b=pathlib.Path('/data/shaoshitong/grounding-eval-20261019');runtime=b/'runtime-21b9ccf4';normal=b/'opsd3304-normal-v4';out=b/('opsd3304-cf-pipeline-v4-recovery1'if RECOVERY else'opsd3304-cf-pipeline-v4');cf=b/'opsd3304-cf-full-v4'
receipt=json.loads((runtime/'deployment-receipt.json').read_text())
for rel,v in receipt['files'].items():assert hashlib.sha256((runtime/rel).read_bytes()).hexdigest()==v['sha256']
assert json.loads((normal/'normal-independent-verification.json').read_text())['state']=='VERIFIED'
assert json.loads((normal/'complete.json').read_text())['completed']==3610
assert json.loads((normal/'control/exit.json').read_text())['exit_code']==0
out.mkdir(exist_ok=True);fd=open(out/'launch.lock','a+');fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
if (out/'launch.json').exists():print(json.dumps({'state':'EXISTING_INSPECT_ONLY','launch':json.loads((out/'launch.json').read_text())}));raise SystemExit(0)
sys.path.insert(0,str(runtime));from reproduction.grounding_eval.normal_a800 import gpu_lease
gpu='GPU-21508d9d-bd43-ff94-7bee-1227ec7b007b'if RECOVERY else'GPU-9b8498ae-93e5-dee7-545a-25669080cf83';lease,index=gpu_lease(gpu);assert index==(5 if RECOVERY else 0)
ancestors=set();a=os.getpid()
while a>1:
 ancestors.add(str(a))
 try:a=int((pathlib.Path('/proc')/str(a)/'stat').read_text().split()[3])
 except (OSError,ValueError):break
# Reject a controller preparing a new native training launch, beyond current NVML.
for proc in pathlib.Path('/proc').glob('[0-9]*'):
 try:cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ').decode()
 except (OSError,UnicodeDecodeError):continue
 if proc.name in ancestors:continue
 if not RECOVERY and any(x in cmd for x in ('search_g1_native_actor_controller.py','search-g1-common8-train','run_prefix_probe_training.py')):raise RuntimeError('Training bootstrap requires manual resource review')
job=dict(python='/data/search-g1/env/bin/python',runtime=str(runtime),method='OPSD3304',run_id='H100 native training job3304; formal-3304; actor/global_step_200',seed=None,source_commit=receipt['source_commit'],normal_root=str(normal),source=str(normal/'normal.jsonl'),checkpoint=str(b/'assets/opsd3304-iter200'),normal_config=str(normal/'normal-binding.json'),pipeline_output=str(out),cf_output=str(cf),gpu_uuid=gpu,cf_max_batch=256,cf_max_memory_utilization=.9,export_a800=False,normal_command=['/usr/bin/python3','-c',"import json;assert json.load(open("+repr(str(normal/'normal-independent-verification.json'))+"))['state']=='VERIFIED'"],scientific_scope='diagnostic typed-v4 CF; not corrected strict or full official reproduction')
path=b/('opsd3304-cf-full-v4-recovery1.job.json'if RECOVERY else'opsd3304-cf-full-v4.job.json');raw=json.dumps(job,sort_keys=True,indent=2).encode()+b'\n'
if path.exists():assert path.read_bytes()==raw
else:
 with path.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
command=[job['python'],'-m','reproduction.grounding_eval.pipeline','--job',str(path)];env=dict(os.environ,PYTHONPATH=str(runtime),PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='8')
os.close(lease)
with (out/'driver.log').open('xb')as f:p=subprocess.Popen(command,cwd=runtime,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
value=dict(pid=p.pid,start_ticks=(pathlib.Path('/proc')/str(p.pid)/'stat').read_text().split()[21],command=command,job_path=str(path),job_sha256=hashlib.sha256(raw).hexdigest(),gpu_uuid=gpu,launched_at=time.time(),source_commit=receipt['source_commit'],normal_only_reused=True)
with (out/'launch.json').open('x')as f:json.dump(value,f);f.flush();os.fsync(f.fileno())
print(json.dumps(value))
'''
code=code.replace('RECOVERY',repr(args.recovery1))
compile(code,'remote-opsd-v4-launch','exec')
if not args.execute:print('PREPARED; requires --execute');raise SystemExit(0)
s=importlib.util.spec_from_file_location('ctl','/Users/shaoshitong/Documents/remote_codex_control/control.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
r=subprocess.run(m.route('baai-a800','/usr/bin/python3 -c '+shlex.quote(code)),capture_output=True,text=True,timeout=40)
d=dict(transport_exit=r.returncode,stderr=r.stderr[-1000:]);d['launch'if not r.returncode else'failure']=json.loads(r.stdout.strip().splitlines()[-1])if not r.returncode else r.stdout[-3000:]
(pathlib.Path('/Users/shaoshitong/Documents/Codex/2026-10-03/xia/outputs/grounding-eval-20261019')/('opsd-cf-v4-recovery1-launch.json'if args.recovery1 else'opsd-cf-v4-launch.json')).write_text(json.dumps(d,indent=2));print(json.dumps(d));raise SystemExit(r.returncode)
