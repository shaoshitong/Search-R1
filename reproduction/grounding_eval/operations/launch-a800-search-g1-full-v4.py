import importlib.util,json,pathlib,shlex,subprocess,hashlib
jobraw=pathlib.Path('outputs/grounding-eval-20261019/a800-search-g1-full-v4.job.json').read_bytes()
code=r'''import pathlib,json,subprocess,os,time,sys,hashlib
payload=json.load(sys.stdin);job=payload['job'];b=pathlib.Path('/data/shaoshitong/grounding-eval-20261019');r=pathlib.Path(job['runtime']);out=pathlib.Path(job['pipeline_output']);out.mkdir(mode=0o700,exist_ok=True)
assert job['source_commit']=='21b9ccf4939803c41ed3351415cd90616709735c'
receipt=json.loads((r/'deployment-receipt.json').read_text());assert receipt['source_commit']==job['source_commit']
for rel,expected in receipt['files'].items():assert hashlib.sha256((r/rel).read_bytes()).hexdigest()==expected['sha256']
assert not pathlib.Path('/proc/926694').exists(),'CF pilot worker still present'
assert len(list((b/'search-g1-normal/records').glob('*.json')))==128
pilot=json.loads((b/'search-g1-cf-pilot128-v4/worker-exit-926694.json').read_text());assert pilot['exit_code']==0
launch=out/'launch.json'
if launch.exists():print(json.dumps(dict(status='PIPELINE_ALREADY_LAUNCHED_INSPECT_ONLY',launch=json.loads(launch.read_text()))));raise SystemExit(0)
gpu=subprocess.check_output(['nvidia-smi','-i','4','--query-gpu=index,uuid,memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True);assert gpu.strip().endswith(', 4, 0'),gpu
path=b/'search-g1-full-v4.job.json';raw=json.dumps(job,sort_keys=True,indent=2).encode()+b'\n'
if path.exists():assert path.read_bytes()==raw
else:
 with path.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
command=[job['python'],'-m','reproduction.grounding_eval.pipeline','--job',str(path)];env=dict(os.environ,PYTHONPATH=str(r),PYTHONUNBUFFERED='1',PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='8')
log=out/'driver.log'
with log.open('xb')as f:p=subprocess.Popen(command,cwd=r,env=env,stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
stat=(pathlib.Path('/proc')/str(p.pid)/'stat').read_text();birth=stat[stat.rfind(')')+2:].split()[19]
value=dict(pid=p.pid,start_ticks=int(birth),command=command,source_commit=job['source_commit'],deployment_receipt_sha256=hashlib.sha256((r/'deployment-receipt.json').read_bytes()).hexdigest(),job_path=str(path),job_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),log=str(log),launched_at=time.time(),gpu_inventory_before=gpu)
with launch.open('x')as f:json.dump(value,f);f.flush();os.fsync(f.fileno())
print(json.dumps(value))
'''
s=importlib.util.spec_from_file_location('ctl','/Users/shaoshitong/Documents/remote_codex_control/control.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
r=subprocess.run(m.route('baai-a800','/usr/bin/python3 -c '+shlex.quote(code)),input=json.dumps({'job':json.loads(jobraw)}),capture_output=True,text=True,timeout=25);v=dict(transport_exit=r.returncode,stderr=r.stderr,launch=json.loads(r.stdout)if not r.returncode else r.stdout);pathlib.Path('outputs/grounding-eval-20261019/a800-search-g1-full-v4-launch.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v))
