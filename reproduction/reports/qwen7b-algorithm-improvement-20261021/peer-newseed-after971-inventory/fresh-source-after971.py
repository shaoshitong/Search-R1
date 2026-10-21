"""Read donor metadata after the five accepted snapshots; never read rollouts."""
import hashlib, importlib.util, json, pathlib, shlex, subprocess, time
R=pathlib.Path(__file__).resolve().parents[4]
HERE=pathlib.Path(__file__).parent
prior=json.loads((R/'reproduction/reports/qwen7b-algorithm-improvement-20261021/peer-newseed-after466-inventory/frozen-audited262-plan.json').read_text())['prior_snapshots']
prior=prior+[dict(complete=dict(path='/mnt/data/shaoshitong/qwen7b-newseed-to-hpc3-20261021/audited262-after709-20261021-d3c45767/complete.json',bytes=875719,sha256='77c88160d009e331a2ccdfa1c632554c52e021c18b173bbf07c0cb5a2259c6be'),expected_batches=262,expected_questions=33536,expected_trajectories=167680)]
worker=r'''import pathlib,json,hashlib,time,socket,sys,os,resource,importlib
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2]);resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3));resource.setrlimit(resource.RLIMIT_CPU,(100,100))
code=pathlib.Path('/mnt/data/shaoshitong/qwen7b-newseed-to-hpc3-20261021/_code-frozen-delta-audited262-after709-20261021-d3c45767-4fbd31d3b41a');sys.path.insert(0,str(code))
from reproduction.operations.qwen7b_warmup import peer_newseed_delta as d
priors=json.loads(sys.stdin.buffer.read());B=d.BASE
def checked(ref):
 v,r=d.load(pathlib.Path(ref['path']));assert all(r[k]==ref[k]for k in('path','bytes','sha256'));return v,r
old={k:set()for k in('inputs','qids','hashes','tids')};oldrefs=[]
for x in priors:
 c,cr0=checked(x['complete']);p,pr=checked(c['plan']);m,mr0=checked(c['canonical_membership']);assert len(p['selected_batches'])==x['expected_batches'] and len(m['groups'])==x['expected_questions'] and c['trajectories']==x['expected_trajectories']
 now=dict(inputs={r['input']for r in p['selected_batches']},qids={g['qid']for g in m['groups']},hashes={g['question_hash']for g in m['groups']},tids={t for g in m['groups']for t in g['trajectory_ids']})
 assert len(now['qids'])==x['expected_questions'] and len(now['tids'])==5*x['expected_questions']
 for k in old:
  assert k=='hashes' or not old[k].intersection(now[k]);old[k].update(now[k])
 oldrefs.append(dict(complete=cr0,plan=pr,membership=mr0))
cfg,cr=d.load(B/'config.json');d.contract(cfg);manifest,mr=d.load(B/'inputs/manifest.json')
assert cr['sha256']=='856a5580c9faf6968aee9535bbc600d9c07f0b018dc13e85611fa76105bba789' and mr['sha256']=='445fc6914ab291bbdad52e1200642653b21c2dd62d3eaa7741631bcf61f71bfd'
assert len(old['inputs'])==971 and len(old['qids'])==124194 and len(old['tids'])==620970
oldrows={}
for p in oldrefs:
 plan,_=checked(p['plan'])
 for row in plan['selected_batches']:assert row['input'] not in oldrows;oldrows[row['input']]=row
seen={k:set()for k in('qids','hashes','tids')};delta=[];sealed=[];pending=[];domains={};audit_counts={};priorhash=within=0
for declared in manifest['batches']:
 rel=d.relative(B,declared['path']);inp=pathlib.Path(rel);srel=str(pathlib.Path('sampling')/inp.parent.name/'batches'/inp.stem/'complete.json')
 if not(B/srel).exists():pending.append(rel);continue
 seal,sr=d.load(d.regular(B,srel));expected=dict(run_id=d.RUN,model_revision=d.REVISION,input_sha256=declared['sha256'],batch=inp.stem,questions=declared['questions'],trajectories=5*declared['questions'],hidden_cached=False,actor_optimizer_updates=0)
 assert all(seal.get(k)==v for k,v in expected.items()),rel
 row=dict(input=rel,input_sha256=declared['sha256'],seal=srel,seal_sha256=sr['sha256'],questions=seal['questions'],trajectories=seal['trajectories'])
 sealed.append(row)
 if rel in oldrows:
  assert all(row[k]==oldrows[rel][k]for k in row),rel
  continue
 batch,br=d.load(d.regular(B,rel));assert br['sha256']==row['input_sha256'] and batch['split']=='train' and batch['model_revision']==d.REVISION
 bc,bcr=d.load(d.regular(B,str(pathlib.Path('sampling')/inp.parent.name/'batches'/inp.stem/'config.json')));d.contract(bc)
 audit=d.cached_audit(B,rel,cr['sha256'],br['sha256'],row['questions']);audit_counts[audit['status']]=audit_counts.get(audit['status'],0)+1
 qids=[];hashes=[];tids=[];own_domains={}
 for q in batch['rows']:
  qid,h=q['qid'],q['question_hash'];ts=[d.RUN+'-outer000001-c%04d-g%d'%(q['candidate_ordinal'],j)for j in range(5)]
  assert q['split']=='train' and qid not in old['qids'] and qid not in seen['qids'] and not old['tids'].intersection(ts) and not seen['tids'].intersection(ts)
  priorhash+=int(h in old['hashes']);within+=int(h in seen['hashes']);seen['qids'].add(qid);seen['hashes'].add(h);seen['tids'].update(ts)
  qids.append(qid);hashes.append(h);tids+=ts;domain=json.loads(qid)[0];domains[domain]=domains.get(domain,0)+1;own_domains[domain]=own_domains.get(domain,0)+1
 assert len(qids)==row['questions']
 if audit['status']=='EXISTING_INDEPENDENT_AUDIT_CURRENT_FSTAT_BOUND':assert d.digest(sorted(tids))==audit['own_trajectory_ids_sha256']
 row.update(input_ref=br,config=bcr,audit=audit,ordered_qids_sha256=d.digest(qids),ordered_question_hashes_sha256=d.digest(hashes),expected_own_G5_IDs_sha256=d.digest(tids),domain_counts=own_domains)
 delta.append(row)
assert set(oldrows)<=set(r['input']for r in sealed)
proc={}
p=pathlib.Path('/proc/3532221')
if p.exists():
 st=(p/'stat').read_text();fields=st[st.rfind(')')+2:].split();cmd=(p/'cmdline').read_bytes();proc=dict(pid=3532221,state=fields[0],birth_ticks=int(fields[19]),cmdline_sha256=hashlib.sha256(cmd).hexdigest(),canonical_source_path_present=bytes(str(B),'utf-8')in cmd or b'qwen7b-resample-seed20261021' in cmd)
else:proc=dict(pid=3532221,exists=False)
writers=[]
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  raw=(p/'cmdline').read_bytes()
  if any(x in raw for x in (b'peer_newseed_frozen_subset_export',b'peer_newseed_hf_stream_upload',b'peer_newseed_export.py')) and b'python' in raw and b'-c\0' not in raw:
   st=(p/'stat').read_text();fields=st[st.rfind(')')+2:].split();writers.append(dict(pid=int(p.name),birth_ticks=int(fields[19]),cmdline_sha256=hashlib.sha256(raw).hexdigest()))
 except(OSError,ValueError):pass
state={}
for rel in('complete.json','failure.json','progress.json','audit/latest.json'):
 if(B/rel).exists():
  v,ref=d.load(B/rel);state[rel]=dict(ref=ref,summary={k:v[k]for k in('status','epoch','questions','trajectories','batches','verified_batches','failed_batches','sealed_pending_batches','unsealed_batches','error','message')if k in v})
print(json.dumps(dict(status='FRESH_SOURCE_SEALED_METADATA_MINUS_ACCEPTED971',epoch=time.time(),hostname=socket.gethostname(),source_base=str(B),source_config=cr,input_manifest=mr,accepted_prior_batches=971,accepted_prior_questions=124194,accepted_prior_refs=oldrefs,prior_snapshots=priors,accepted_all_prior_input_and_seal_SHA_unchanged=True,current_sealed_batches=len(sealed),current_sealed_questions=sum(x['questions']for x in sealed),delta_batches=len(delta),delta_questions=len(seen['qids']),delta_new5_trajectories=len(seen['tids']),delta_audit_status_counts=audit_counts,domain_counts=domains,prior_training_text_hash_overlap_questions=priorhash,within_delta_training_text_hash_duplicates=within,delta_records_sha256=d.digest(delta),delta_batches_metadata=delta,remaining_unsealed_batches=len(pending),all169028_complete=not pending,worker3532221=proc,existing_matching_export_processes=writers,source_state=state,payload_SHA_recomputed=False,strict_EM_recomputed=False,producer_or_process_modified=False,payload_exported=False)))'''
s=importlib.util.spec_from_file_location('safe_donor_route','/Users/shaoshitong/Documents/remote_codex_control/control.py');ctl=importlib.util.module_from_spec(s);s.loader.exec_module(ctl)
started=time.time();cmd='python3 -c '+shlex.quote(worker);assert len(cmd.encode())<16000
r=subprocess.run(ctl.route('h100-10',cmd),input=json.dumps(prior).encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=60)
if r.returncode:raise RuntimeError('Donor metadata query failed: '+r.stderr.decode(errors='replace')[-1600:])
v=json.loads(r.stdout.decode().splitlines()[-1]);assert len(r.stdout)<4*1024**2
v.update(operator_seconds=time.time()-started,operator_sha256=hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest())
(HERE/'actual-source-after971.json').write_text(json.dumps(v,indent=2)+'\n')
print(json.dumps({k:x for k,x in v.items()if k not in('delta_batches_metadata','accepted_prior_refs','prior_snapshots')}))
