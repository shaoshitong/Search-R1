"""Finite read-only source metadata inventory, original strict H100 control."""
import hashlib,importlib.util,json,pathlib,shlex,subprocess,time
HERE=pathlib.Path(__file__).resolve().parent;REPO=HERE.parents[3]
FILES=['reproduction/operations/qwen7b_warmup/peer_newseed_delta.py','reproduction/operations/qwen7b_warmup/peer_newseed_remaining_inventory.py']
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO).decode().strip();remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/codex/qwen7b-probe-warmup-20261019'],cwd=REPO).decode().split()[0]
if commit!=remote:raise ValueError('Publish source before source-node read')
out=HERE/'source-observation.json'
if out.exists():raise ValueError('Preserve existing finite source observation')
sources=[]
for name in FILES:
 raw=(REPO/name).read_bytes()
 if raw!=subprocess.check_output(['git','show',commit+':'+name],cwd=REPO):raise ValueError('Unpublished worker refused')
 sources.append(dict(source=raw.decode(),sha256=hashlib.sha256(raw).hexdigest()))
selfname=str(pathlib.Path(__file__).relative_to(REPO))
if pathlib.Path(__file__).read_bytes()!=subprocess.check_output(['git','show',commit+':'+selfname],cwd=REPO):raise ValueError('Unpublished observer refused')
bind0=REPO/'reproduction/reports/qwen7b-algorithm-improvement-20261021/peer-newseed-delta/imported-snapshot-fresh-binding.json'
bind1=REPO/'reproduction/reports/qwen7b-algorithm-improvement-20261021/peer-audited122-delta-import/FINAL.json'
packet=dict(sources=sources,old=json.loads(bind0.read_bytes()),new=json.loads(bind1.read_bytes()))
program=r'''
import hashlib,json,os,resource,sys,types
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2]);resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3));resource.setrlimit(resource.RLIMIT_CPU,(120,120))
p=json.loads(sys.stdin.buffer.read());namespaces=[]
for i,part in enumerate(p['sources']):
 assert hashlib.sha256(part['source'].encode()).hexdigest()==part['sha256'];ns={'__name__':'readonly_source_inventory_%d'%i};exec(compile(part['source'],'<published_readonly_source_inventory>','exec'),ns);namespaces.append(ns)
v=namespaces[1]['inventory'](p['old'],p['new'],base_module=types.SimpleNamespace(**namespaces[0]));print(json.dumps(v,sort_keys=True))
'''
spec=importlib.util.spec_from_file_location('source_after366_safe_control','/Users/shaoshitong/Documents/remote_codex_control/control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl)
command='python3 -c '+shlex.quote(program)
if len(command.encode())>=16000:raise ValueError('Bounded remote control command required')
started=time.time();r=subprocess.run(ctl.route('h100-10',command),input=json.dumps(packet).encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=150)
if r.returncode:raise RuntimeError('Read-only source observation rejected: '+r.stderr.decode(errors='replace')[-1800:])
if len(r.stdout)>128*1024:raise ValueError('Only bounded aggregate metadata may return')
v=json.loads(r.stdout.decode().splitlines()[-1]);v.update(source_commit=commit,remote_sha=remote,worker_source_hashes=[x['sha256']for x in sources],operator_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest(),operator_elapsed_seconds=time.time()-started)
out.write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v),flush=True)
