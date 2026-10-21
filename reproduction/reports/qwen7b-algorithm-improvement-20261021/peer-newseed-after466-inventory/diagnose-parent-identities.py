"""Read-only source metadata identity diagnosis; no payload/EM/feature read."""
import importlib.util,json,pathlib,shlex,subprocess,time
R=pathlib.Path(__file__).resolve().parents[4];HERE=pathlib.Path(__file__).parent
x=json.loads((HERE/'actual-source-after466.json').read_bytes());f=json.loads((R/'reproduction/reports/qwen7b-algorithm-improvement-20261021/peer-newseed-after366-inventory/frozen-audited100-plan.json').read_bytes())
prior=f['prior_snapshots']+[dict(complete=dict(path='/mnt/data/shaoshitong/qwen7b-newseed-to-hpc3-20261021/audited100-after366-20261021-636015f5-a02/complete.json',bytes=342205,sha256='09e1700e6a67b39b76c13d7088797ff5f1cebbc23917a97d4aa10e06e96a0819'),expected_batches=100,expected_questions=12800,expected_trajectories=64000)]
worker='''import json,sys,pathlib,types,hashlib,time
p=json.loads(sys.stdin.buffer.read());root=pathlib.Path('/mnt/data/shaoshitong/qwen7b-newseed-to-hpc3-20261021/metadata-plans/after466-8e23374458fb');pkg=types.ModuleType('finite');pkg.__path__=[];sys.modules['finite']=pkg
d=types.ModuleType('finite.peer_newseed_delta');exec(compile((root/'peer_newseed_delta.py').read_bytes(),str(root/'peer_newseed_delta.py'),'exec'),d.__dict__);sys.modules[d.__name__]=d
m=types.ModuleType('finite.freezer');m.__package__='finite';exec(compile((root/'peer_newseed_freeze_after466.py').read_bytes(),str(root/'peer_newseed_freeze_after466.py'),'exec'),m.__dict__)
old,refs=m.prior_membership(p['priors']);tot={};seen={k:set()for k in('qids','hashes','tids')};examples=[];domains={};rowsout=[]
for item in p['reference']['delta_batches_metadata']:
 batch,ref=d.load(d.regular(d.BASE,item['input']));assert ref['sha256']==item['input_sha256'];part='existing243'if item['existing_audit']['status']=='VERIFIED'else'missing262';s=tot.setdefault(part,dict(questions=0,prior_qid_overlap=0,prior_hash_overlap=0,prior_tid_overlap=0,within_qid_duplicates=0,within_hash_duplicates=0,within_tid_duplicates=0))
 for q in batch['rows']:
  qid,h=q['qid'],q['question_hash'];ts=[d.RUN+'-outer000001-c%04d-g%d'%(q['candidate_ordinal'],j)for j in range(5)];s['questions']+=1;dd=domains.setdefault(part,{});dom=json.loads(qid)[0];dd[dom]=dd.get(dom,0)+1
  flags=dict(prior_qid_overlap=int(qid in old['qids']),prior_hash_overlap=int(h in old['hashes']),prior_tid_overlap=len(old['tids'].intersection(ts)),within_qid_duplicates=int(qid in seen['qids']),within_hash_duplicates=int(h in seen['hashes']),within_tid_duplicates=len(seen['tids'].intersection(ts)))
  for k,v in flags.items():s[k]+=v
  if any(flags.values())and len(examples)<10:examples.append(dict(input=item['input'],qid_sha256=hashlib.sha256(qid.encode()).hexdigest(),question_hash=h,flags=flags))
  seen['qids'].add(qid);seen['hashes'].add(h);seen['tids'].update(ts)
 rowsout.append(dict(input=item['input'],questions=len(batch['rows']),input_sha256=ref['sha256']))
print(json.dumps(dict(status='FINITE505_IDENTITY_OVERLAP_DIAGNOSIS',epoch=time.time(),parent_sha256=p['reference']['delta_records_sha256'],totals=tot,domains=domains,prior_counts={k:len(v)for k,v in old.items()},new_unique_counts={k:len(v)for k,v in seen.items()},examples=examples,payload_read=False,EM_read=False,producer_modified=False)))'''
s=importlib.util.spec_from_file_location('ctl','/Users/shaoshitong/Documents/remote_codex_control/control.py');ctl=importlib.util.module_from_spec(s);s.loader.exec_module(ctl)
r=subprocess.run(ctl.route('h100-10','python3 -c '+shlex.quote(worker)),input=json.dumps(dict(reference=x,priors=prior)).encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=150)
if r.returncode:raise RuntimeError(r.stderr.decode()[-1500:])
x=json.loads(r.stdout.decode().splitlines()[-1]);(HERE/'actual-parent505-identity-diagnosis.json').write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(x))
