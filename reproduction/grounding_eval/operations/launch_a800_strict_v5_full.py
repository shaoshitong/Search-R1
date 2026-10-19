"""Explicitly approved strict-v5 CF-only continuation; no normal generation."""
import argparse,base64,hashlib,importlib.util,json,pathlib,shlex,subprocess
p=argparse.ArgumentParser(__doc__);p.add_argument('--method',choices=['Search-G1','Soft-to-Hard','OPSD3304'],default='Search-G1');p.add_argument('--execute',action='store_true');p.add_argument('--review',default='/Users/shaoshitong/Documents/Codex/2026-10-03/xia/outputs/grounding-eval-20261019/g1-strict-v5-pilot-root-acceptance.json');args=p.parse_args()
b='/data/shaoshitong/grounding-eval-20261019';prefix={'Search-G1':'search-g1','Soft-to-Hard':'soft-to-hard','OPSD3304':'opsd3304'}[args.method]
settings={'Search-G1':dict(run_id='db474b6abf654ad9892b02ae67ab6f86',normal='search-g1-normal',checkpoint='/data/shaoshitong/Search-G1-a800-fresh200-a20f0475ec824c69ba1554f39e9ff512/checkpoints/search-g1-common8/actor/global_step_200',config='normal-config.json'),'Soft-to-Hard':dict(run_id='1db662001a0f4d998928bcf6f5f33c9d',normal='soft-to-hard-normal',checkpoint='/data/shaoshitong/Search-G1-a800-fresh200-5e1f65d8e7d44363b02938c8d19852ab/checkpoints/search-g1-common8/actor/global_step_200',config='normal-config.json'),'OPSD3304':dict(run_id='H100 native training job3304; formal-3304; actor/global_step_200',normal='opsd3304-normal-v4',checkpoint=b+'/assets/opsd3304-iter200',config='normal-binding.json')}
x=settings[args.method];reviewraw=pathlib.Path(args.review).read_bytes();review=json.loads(reviewraw);assert review['accepted']and review['source_commit']=='16c36aa777f684f245b1d38cb2939ef175131831'and review['pilot_audit_sha256']=='1fad1cb490e182920dc2f9f9edccb8520bbc2949a5425fdcda998786b00c4cca'
if args.execute:assert args.method in review.get('authorized_methods',['Search-G1']),'This method has no explicit full-run authorization'
normal=b+'/'+x['normal'];job=dict(mode='full_after_pilot_review',module='reproduction.grounding_eval.strict_v5',method=args.method,run_id=x['run_id'],source_commit=review['source_commit'],python='/data/search-g1/env/bin/python',runtime=b+'/runtime-16c36aa7',source=normal+'/normal.jsonl',checkpoint=x['checkpoint'],normal_config=normal+'/'+x['config'],control_output=b+'/'+prefix+'-cf-strict-v5-full-control',cf_output=b+'/'+prefix+'-cf-strict-v5',gpu_uuid='GPU-21508d9d-bd43-ff94-7bee-1227ec7b007b',pilot_manual_review=dict(approved=True,reviewed_pilot_sha256=review['pilot_audit_sha256'],review_receipt_sha256=hashlib.sha256(reviewraw).hexdigest(),reviewed_at=review['reviewed_at']))
controller=(pathlib.Path(__file__).parent/'cf_only_full_controller.py').read_bytes();payload=dict(job=job,controller=base64.b64encode(controller).decode(),controller_sha256=hashlib.sha256(controller).hexdigest(),review=base64.b64encode(reviewraw).decode())
code=r'''
import pathlib,json,sys,base64,hashlib,os,subprocess,time,fcntl
p=json.load(sys.stdin);job=p['job'];runtime=pathlib.Path(job['runtime']);b=runtime.parent;assert b==pathlib.Path('/data/shaoshitong/grounding-eval-20261019')
receipt=json.loads((runtime/'deployment-receipt.json').read_text());assert receipt['source_commit']==job['source_commit']
for rel,v in receipt['files'].items():assert hashlib.sha256((runtime/rel).read_bytes()).hexdigest()==v['sha256']
control=pathlib.Path(job['control_output']);control.mkdir(exist_ok=True);lock=open(control/'launch.lock','a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
if(control/'launch.json').exists():print(json.dumps({'state':'EXISTING_INSPECT_ONLY','launch':json.loads((control/'launch.json').read_text())}));raise SystemExit(0)
sys.path.insert(0,str(runtime));from reproduction.grounding_eval.normal_a800 import gpu_lease
fd,index=gpu_lease(job['gpu_uuid']);assert index==5
raw=base64.b64decode(p['controller']);assert hashlib.sha256(raw).hexdigest()==p['controller_sha256'];script=b/('cf-only-full-controller-'+p['controller_sha256'][:12]+'.py')
if script.exists():assert script.read_bytes()==raw
else:
 with script.open('xb')as f:os.fchmod(f.fileno(),0o600);f.write(raw);f.flush();os.fsync(f.fileno())
review=base64.b64decode(p['review']);assert hashlib.sha256(review).hexdigest()==job['pilot_manual_review']['review_receipt_sha256']
with(control/'root-review.json').open('xb')as f:f.write(review);f.flush();os.fsync(f.fileno())
jobpath=b/(job['method'].lower().replace('->','-')+'-cf-strict-v5-full.job.json');raw=json.dumps(job,sort_keys=True,indent=2).encode()+b'\n'
if jobpath.exists():assert jobpath.read_bytes()==raw
else:
 with jobpath.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
cf=pathlib.Path(job['cf_output']);existing={f.name:hashlib.sha256(f.read_bytes()).hexdigest()for f in(cf/'results').glob('*.json')}
if job['method']=='Search-G1':assert len(existing)==16,'G1 continuation requires exactly the reviewed16 intact results'
command=[job['python'],'-B',str(script),'--job',str(jobpath)];os.close(fd)
with(control/'controller.log').open('xb')as log:child=subprocess.Popen(command,cwd=b,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1'),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
launch=dict(pid=child.pid,birth_ticks=(pathlib.Path('/proc')/str(child.pid)/'stat').read_text().split()[21],command=command,job_sha256=hashlib.sha256(raw).hexdigest(),controller_sha256=p['controller_sha256'],source_commit=job['source_commit'],gpu_uuid=job['gpu_uuid'],launched_at=time.time(),mode=job['mode'],preserved_result_file_sha256=existing)
with(control/'launch.json').open('x')as f:json.dump(launch,f);f.flush();os.fsync(f.fileno())
print(json.dumps(launch))
'''
compile(code,'remote-strict-v5-full-launch','exec')
if not args.execute:print(json.dumps(dict(state='PREPARED_NOT_EXECUTED',job=job)));raise SystemExit(0)
s=importlib.util.spec_from_file_location('ctl','/Users/shaoshitong/Documents/remote_codex_control/control.py');ctl=importlib.util.module_from_spec(s);s.loader.exec_module(ctl)
r=subprocess.run(ctl.route('baai-a800','/usr/bin/python3 -c '+shlex.quote(code)),input=json.dumps(payload),capture_output=True,text=True,timeout=40);audit=dict(transport_exit=r.returncode,stderr=r.stderr[-1500:],launch=json.loads(r.stdout.strip().splitlines()[-1])if not r.returncode else r.stdout[-2500:]);pathlib.Path('/Users/shaoshitong/Documents/Codex/2026-10-03/xia/outputs/grounding-eval-20261019/'+prefix+'-strict-v5-full-launch.json').write_text(json.dumps(audit,indent=2));print(json.dumps(audit));raise SystemExit(r.returncode)
