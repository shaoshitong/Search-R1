"""Readonly shared donor metadata; no rollout/EM/tensor payload or writer calls."""
import json,pathlib,hashlib,subprocess,importlib.util,shlex,time
R=pathlib.Path(__file__).resolve().parents[4];HERE=pathlib.Path(__file__).parent
p=R/'reproduction/reports/qwen7b-algorithm-improvement-20261021/peer-newseed-after366-inventory/frozen-audited100-plan.json';frozen=json.loads(p.read_bytes());priors=list(frozen['prior_snapshots'])+[dict(complete=dict(path='/mnt/data/shaoshitong/qwen7b-newseed-to-hpc3-20261021/audited100-after366-20261021-636015f5-a02/complete.json',bytes=342205,sha256='09e1700e6a67b39b76c13d7088797ff5f1cebbc23917a97d4aa10e06e96a0819'),expected_batches=100,expected_questions=12800,expected_trajectories=64000)]
worker='''import pathlib,json,hashlib,time,socket,sys,os,resource
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2]);resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3));resource.setrlimit(resource.RLIMIT_CPU,(120,120))
B=pathlib.Path('/mnt/shared/workspace/training_data_node_10/shaoshitong/qwen7b-resample-seed20261021-g5-20261020');prior=json.loads(sys.stdin.buffer.read())
def load(p):
 p=pathlib.Path(p);a=p.stat();data=p.read_bytes();b=p.stat();assert(a.st_size,a.st_mtime_ns,a.st_ino)==(b.st_size,b.st_mtime_ns,b.st_ino);return json.loads(data),dict(path=str(p),bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),mtime=b.st_mtime)
def checked(ref):
 x,p=load(ref['path']);assert all(p[k]==ref[k]for k in('path','bytes','sha256'));return x,p
cfg,cref=load(B/'config.json');manifest,mref=load(B/'inputs/manifest.json');assert cfg['run_id']=='70188b8e50fd4b45bc420ce58014f46e'and cfg['seed']==20261021 and cfg['actor_optimizer_updates']==0
old={};oldrefs=[]
for ref in prior:
 c,cp=checked(ref['complete']);plan,pp=checked(c['plan']);rows=plan['selected_batches'];assert len(rows)==ref['expected_batches'];oldrefs.append(dict(complete=cp,plan=pp,batches=len(rows)))
 for row in rows:
  assert row['input']not in old;old[row['input']]=row
assert len(old)==466
sealed=[];delta=[];audit_counts={};old_same=True
for p in sorted((B/'sampling').glob('shard-*/batches/batch*/complete.json')):
 s,sp=load(p);inp='inputs/'+p.parent.parent.parent.name+'/'+p.parent.name+'.json';assert s['run_id']==cfg['run_id']and s['model_revision']==cfg['model_revision']and s['actor_optimizer_updates']==0 and s['trajectories']==5*s['questions'];row=dict(input=inp,input_sha256=s['input_sha256'],seal=str(p.relative_to(B)),seal_sha256=sp['sha256'],seal_mtime=sp['mtime'],questions=s['questions'],trajectories=s['trajectories'])
 if inp in old:
  x=old[inp];assert all(row[k]==x[k]for k in('input_sha256','seal_sha256','questions','trajectories'))
 else:
  ap=B/'audit/batches'/pathlib.Path(inp).parent.name/(pathlib.Path(inp).stem+'.json')
  if ap.exists():
   a,ar=load(ap);row['existing_audit']=dict(status=a.get('status'),proof=ar);audit_counts[a.get('status','UNKNOWN')]=audit_counts.get(a.get('status','UNKNOWN'),0)+1
  else:row['existing_audit']=dict(status='ABSENT');audit_counts['ABSENT']=audit_counts.get('ABSENT',0)+1
  delta.append(row)
 sealed.append(row)
assert set(old)<=set(x['input']for x in sealed)
state={}
for rel in('complete.json','failure.json','progress.json','audit/latest.json'):
 p=B/rel
 if p.exists():
  v,r=load(p);state[rel]=dict(ref=r,summary={k:v.get(k)for k in('status','epoch','run_id','seed','model_revision','questions','trajectories','batches','verified_batches','verified_unique_qids','verified_unique_trajectories','failed_batches','sealed_pending_batches','unsealed_batches','strict_em_recomputed','error','message')if k in v})
seal_digest=hashlib.sha256(json.dumps(sealed,sort_keys=True,separators=(',',':')).encode()).hexdigest();delta_digest=hashlib.sha256(json.dumps(delta,sort_keys=True,separators=(',',':')).encode()).hexdigest()
out=dict(status='FRESH_SOURCE_SEALED_METADATA_MINUS_ACCEPTED466',epoch=time.time(),hostname=socket.gethostname(),source_base=str(B),source_config=cref,input_manifest=mref,expected_input_batches=len(manifest['batches']),expected_input_questions=manifest['questions'],current_sealed_batches=len(sealed),current_sealed_questions=sum(x['questions']for x in sealed),current_sealed_new5_trajectories=sum(x['trajectories']for x in sealed),current_sealed_membership_sha256=seal_digest,accepted_prior_batches=466,accepted_prior_questions=sum(x['questions']for x in old.values()),accepted_all_prior_input_and_seal_SHA_unchanged=True,accepted_prior_refs=oldrefs,delta_batches=len(delta),delta_questions=sum(x['questions']for x in delta),delta_new5_trajectories=sum(x['trajectories']for x in delta),delta_audit_status_counts=audit_counts,delta_records_sha256=delta_digest,delta_batches_metadata=delta,all_source_inputs_sealed=len(sealed)==len(manifest['batches']),source_state=state,oldest_new_seal_mtime=min((x['seal_mtime']for x in delta),default=None),latest_new_seal_mtime=max((x['seal_mtime']for x in delta),default=None),payload_SHA_recomputed=False,strict_EM_recomputed=False,producer_or_process_modified=False,payload_exported=False)
print(json.dumps(out))'''
s=importlib.util.spec_from_file_location('safe_donor_route','/Users/shaoshitong/Documents/remote_codex_control/control.py');ctl=importlib.util.module_from_spec(s);s.loader.exec_module(ctl);cmd='python3 -c '+shlex.quote(worker);assert len(cmd.encode())<16000
started=time.time();r=subprocess.run(ctl.route('h100-10',cmd),input=json.dumps(priors).encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=55)
if r.returncode:raise RuntimeError('Source metadata access failed; do not restart producer: '+r.stderr.decode(errors='replace')[-1200:])
x=json.loads(r.stdout.decode().splitlines()[-1]);assert len(r.stdout)<2*1024**2;x['operator_seconds']=time.time()-started;x['operator_sha256']=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest();(HERE/'actual-source-after466.json').write_text(json.dumps(x,indent=2)+'\n');small={k:v for k,v in x.items()if k not in('delta_batches_metadata','accepted_prior_refs')};print(json.dumps(small))
