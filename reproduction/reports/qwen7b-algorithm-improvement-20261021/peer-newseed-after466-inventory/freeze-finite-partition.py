"""Publish-bound remote CPU metadata freeze; no payload export or actor calls."""
import importlib.util,json,pathlib,shlex,subprocess,hashlib,time
R=pathlib.Path(__file__).resolve().parents[4];HERE=pathlib.Path(__file__).parent
rel='reproduction/operations/qwen7b_warmup/'
frozen=json.loads((R/'reproduction/reports/qwen7b-algorithm-improvement-20261021/peer-newseed-after366-inventory/frozen-audited100-plan.json').read_bytes())
priors=frozen['prior_snapshots']+[dict(complete=dict(path='/mnt/data/shaoshitong/qwen7b-newseed-to-hpc3-20261021/audited100-after366-20261021-636015f5-a02/complete.json',bytes=342205,sha256='09e1700e6a67b39b76c13d7088797ff5f1cebbc23917a97d4aa10e06e96a0819'),expected_batches=100,expected_questions=12800,expected_trajectories=64000)]
reference=json.loads((HERE/'actual-source-after466.json').read_bytes())
publication=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()
files={n:(R/rel/n).read_text() for n in ('peer_newseed_delta.py','peer_newseed_freeze_after466.py')}
for n,raw in files.items():
 tracked=subprocess.check_output(['git','show',publication+':'+rel+n],cwd=R)
 if tracked!=raw.encode():raise ValueError('Unpublished source: '+n)
remote_sha=subprocess.check_output(['git','ls-remote','origin','refs/heads/codex/qwen7b-probe-warmup-20261019'],cwd=R,text=True).split()[0]
if remote_sha!=publication:raise ValueError('Remote branch must equal exact HEAD before source freeze')
packet=dict(reference=reference,priors=priors,code=files,publication=publication)
worker='''import sys,json,pathlib,types,hashlib,os,time
p=json.loads(sys.stdin.buffer.read());root=pathlib.Path('/mnt/data/shaoshitong/qwen7b-newseed-to-hpc3-20261021/metadata-plans/after466-'+p['reference']['delta_records_sha256'][:12]+'-'+p['publication'][:8]);root.mkdir(parents=True,exist_ok=True)
for name,raw in p['code'].items():
 dest=root/name
 if dest.exists():assert dest.read_bytes()==raw.encode()
 else:dest.write_text(raw)
mod=types.ModuleType('finite_freezer');mod.__package__='finite';pkg=types.ModuleType('finite');pkg.__path__=[];sys.modules['finite']=pkg
d=types.ModuleType('finite.peer_newseed_delta');exec(compile(p['code']['peer_newseed_delta.py'],str(root/'peer_newseed_delta.py'),'exec'),d.__dict__);sys.modules[d.__name__]=d
exec(compile(p['code']['peer_newseed_freeze_after466.py'],str(root/'peer_newseed_freeze_after466.py'),'exec'),mod.__dict__)
refpath=root/'parent505-reference.json'
if refpath.exists():assert json.loads(refpath.read_bytes())==p['reference']
else:refpath.write_bytes(d.canonical(p['reference'])+bytes([10]))
result=mod.freeze(p['reference'],p['priors'],root/'audited243-plan.json',which='existing243')
result['source_publication']=p['publication'];result['parent505_reference']=d.load(refpath)[1];result['root']=str(root)
print(json.dumps(result))'''
s=importlib.util.spec_from_file_location('ctl','/Users/shaoshitong/Documents/remote_codex_control/control.py');ctl=importlib.util.module_from_spec(s);s.loader.exec_module(ctl)
cmd='python3 -c '+shlex.quote(worker)
if len(cmd.encode())>=16000:raise ValueError('Oversized control command')
started=time.time();r=subprocess.run(ctl.route('h100-10',cmd),input=json.dumps(packet).encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=150)
if r.returncode:raise RuntimeError(r.stderr.decode(errors='replace')[-2500:])
v=json.loads(r.stdout.decode().splitlines()[-1]);(HERE/'actual-audited243-frozen.json').write_text(json.dumps(v,indent=2)+'\n')
print(json.dumps({k:v[k]for k in('plan','source_publication','parent505_reference','root')}));print(json.dumps({k:v['metadata_plan'][k]for k in('batches','questions','trajectories','domain_counts','batch_records_sha256','prior_identity_overlap')}));print('elapsed_seconds',time.time()-started)
