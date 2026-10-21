"""Publish-checked code via stdin RAM; writes only one NAS metadata plan."""
import hashlib,importlib.util,json,pathlib,shlex,subprocess,time
HERE=pathlib.Path(__file__).resolve().parent;REPO=HERE.parents[3]
NAMES=['peer_newseed_delta.py','peer_newseed_remaining_inventory.py','peer_newseed_freeze_audited100.py'];P='reproduction/operations/qwen7b_warmup/'
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO).decode().strip();remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/codex/qwen7b-probe-warmup-20261019'],cwd=REPO).decode().split()[0]
if commit!=remote:raise ValueError('Publish source before metadata freeze')
sources=[]
for name in NAMES:
 raw=(REPO/P/name).read_bytes()
 if raw!=subprocess.check_output(['git','show',commit+':'+P+name],cwd=REPO):raise ValueError('Unpublished worker refused')
 sources.append(dict(source=raw.decode(),sha256=hashlib.sha256(raw).hexdigest()))
selfname=str(pathlib.Path(__file__).relative_to(REPO))
if pathlib.Path(__file__).read_bytes()!=subprocess.check_output(['git','show',commit+':'+selfname],cwd=REPO):raise ValueError('Unpublished operator refused')
packet=dict(sources=sources,old=json.loads((REPO/'reproduction/reports/qwen7b-algorithm-improvement-20261021/peer-newseed-delta/imported-snapshot-fresh-binding.json').read_bytes()),
 new=json.loads((REPO/'reproduction/reports/qwen7b-algorithm-improvement-20261021/peer-audited122-delta-import/FINAL.json').read_bytes()),reference=json.loads((HERE/'source-observation.json').read_bytes()))
program=r'''
import hashlib,json,os,resource,sys,types
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2]);resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3));resource.setrlimit(resource.RLIMIT_CPU,(120,120))
p=json.loads(sys.stdin.buffer.read());modules=[]
for i,item in enumerate(p['sources']):
 assert hashlib.sha256(item['source'].encode()).hexdigest()==item['sha256'];ns={'__name__':'metadata_freeze_worker_%d'%i};exec(compile(item['source'],'<published_metadata_freeze>','exec'),ns);modules.append(types.SimpleNamespace(**ns))
r=modules[2].freeze(p['old'],p['new'],p['reference'],base_module=modules[0],remaining_module=modules[1]);print(json.dumps(r,sort_keys=True))
'''
command='python3 -c '+shlex.quote(program)
if len(command.encode())>=16000:raise ValueError('Bounded control command required')
spec=importlib.util.spec_from_file_location('metadata_freeze_private_route','/Users/shaoshitong/Documents/remote_codex_control/control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl)
started=time.time();result=subprocess.run(ctl.route('h100-10',command),input=json.dumps(packet).encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=150)
if result.returncode:raise RuntimeError('Source metadata freeze rejected; no payload operation: '+result.stderr.decode(errors='replace')[-1800:])
if len(result.stdout)>1024**2:raise ValueError('Metadata-only plan exceeds1MiB')
r=json.loads(result.stdout.decode().splitlines()[-1]);plan=r['metadata_plan'];raw=json.dumps(plan,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()+b'\n'
if len(raw)!=r['plan']['bytes']or hashlib.sha256(raw).hexdigest()!=r['plan']['sha256']:raise ValueError('Local metadata copy differs from authoritative NAS plan')
(HERE/'frozen-audited100-plan.json').write_bytes(raw)
r.pop('metadata_plan');r.update(source_commit=commit,remote_sha=remote,worker_source_hashes=[v['sha256']for v in sources],operator_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),operator_elapsed_seconds=time.time()-started)
(HERE/'freeze-receipt.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
