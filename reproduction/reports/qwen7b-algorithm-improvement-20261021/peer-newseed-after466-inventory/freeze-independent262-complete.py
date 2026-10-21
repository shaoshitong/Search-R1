"""Freeze fixed262 only after its own full proof and the finite243 source seal."""
import hashlib,importlib.util,json,pathlib,shlex,subprocess
R=pathlib.Path(__file__).resolve().parents[4];HERE=pathlib.Path(__file__).parent;P='reproduction/operations/qwen7b_warmup/'
first=json.loads((HERE/'actual-audited243-frozen.json').read_bytes());launch=json.loads((HERE/'actual-independent262-launch.json').read_bytes());reference=json.loads((HERE/'actual-source-after466.json').read_bytes());attestation=json.loads((HERE/'actual-batch300-reseal-attestation.json').read_bytes());pub=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip()
code={n:(R/P/n).read_text()for n in('peer_newseed_delta.py','peer_newseed_freeze_after466.py')}
for n,raw in code.items():
 if subprocess.check_output(['git','show',pub+':'+P+n],cwd=R)!=raw.encode():raise ValueError('Unpublished source')
if subprocess.check_output(['git','ls-remote','origin','refs/heads/codex/qwen7b-probe-warmup-20261019'],cwd=R,text=True).split()[0]!=pub:raise ValueError('Publish before freeze')
worker='''import pathlib,sys,json,types,hashlib,time
p=json.loads(sys.stdin.buffer.read());proofroot=pathlib.Path(p['launch']['root']);complete=proofroot/'complete.json';prior243=pathlib.Path('/mnt/data/shaoshitong/qwen7b-newseed-to-hpc3-20261021/audited243-after466-20261021-45227948/complete.json')
attpath=pathlib.Path(p['attestation']['proof']['path']).parents[3]/'complete.json'
if not attpath.exists()or not prior243.exists():
 print(json.dumps(dict(status='WAITING_FOR_ACTUAL_PROOF262_AND_SOURCE243_COMPLETE',proof262_complete_exists=attpath.exists(),source243_complete_exists=prior243.exists(),epoch=time.time())));sys.exit(0)
root=pathlib.Path('/mnt/data/shaoshitong/qwen7b-newseed-to-hpc3-20261021/metadata-plans/after466-8e23374458fb-'+p['publication'][:8]);root.mkdir(parents=True,exist_ok=True)
for name,raw in p['code'].items():
 dst=root/name
 if dst.exists():assert dst.read_bytes()==raw.encode()
 else:dst.write_text(raw)
pkg=types.ModuleType('finite');pkg.__path__=[];sys.modules['finite']=pkg;d=types.ModuleType('finite.peer_newseed_delta');exec(compile(p['code']['peer_newseed_delta.py'],str(root/'peer_newseed_delta.py'),'exec'),d.__dict__);sys.modules[d.__name__]=d;m=types.ModuleType('finite.freezer');m.__package__='finite';exec(compile(p['code']['peer_newseed_freeze_after466.py'],str(root/'peer_newseed_freeze_after466.py'),'exec'),m.__dict__)
att,attref=d.load(attpath);assert att==p['attestation'];failure,fref=d.load(proofroot/'failure.json');tail,tref=d.load(proofroot/'tail9-complete.json');assert failure['checked_batches']==252 and tail['verified']==9 and tail['failed']==0 and att['questions']==128
overrides={att['input']:dict(audit_root=str(attpath.parent/'audit'),auditor_sha256=att['auditor_sha256'],attestation=attref)}
c,cr=d.load(prior243);assert c['questions']==31010 and c['trajectories']==155050 and c['frozen_sealed_batches']==243
plan,pr=m.checked(c['plan']);assert plan['subset_selection']['path']==p['first']['plan']['path']and plan['subset_selection']['sha256']==p['first']['plan']['sha256']
priors=list(p['first']['metadata_plan']['prior_snapshots'])+[dict(complete=cr,expected_batches=243,expected_questions=31010,expected_trajectories=155050)]
r=m.freeze(p['reference'],priors,root/'audited262-plan.json',which='missing262',audit_root=proofroot/'audit',audit_overrides=overrides)
qualified=dict(status='FIXED262_FRESH_SHA_CAUSAL_OWN_ID_ACCEPTED_WITH_ONE_EXPLICIT_METADATA_RESEAL_ATTESTATION',batches=262,questions=33536,trajectories=167680,pinned_original_chain_batches=261,explicit_metadata_reseal_attestation_batches=1,original_failure_preserved=fref,tail9_complete=tref,metadata_reseal_attestation=attref,all_current_source_fstats_held=True,bulk_payload_SHA_recomputed=True,strict_EM_tokenizer_recomputed=False,source_reference=p['launch']['reference'],producer_modified=False,epoch=time.time());qp=root/'qualified262-complete.json';qp.write_bytes(d.canonical(qualified)+bytes([10]));proofref=d.load(qp)[1]
r.update(source_publication=p['publication'],independent262_proof_complete=proofref,prior243_source_complete=cr,root=str(root));print(json.dumps(r))'''
s=importlib.util.spec_from_file_location('ctl','/Users/shaoshitong/Documents/remote_codex_control/control.py');ctl=importlib.util.module_from_spec(s);s.loader.exec_module(ctl)
cmd='python3 -c '+shlex.quote(worker);assert len(cmd.encode())<16000
r=subprocess.run(ctl.route('h100-10',cmd),input=json.dumps(dict(first=first,launch=launch,reference=reference,attestation=attestation,publication=pub,code=code)).encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=150)
if r.returncode:raise RuntimeError(r.stderr.decode(errors='replace')[-1800:])
x=json.loads(r.stdout.decode().splitlines()[-1]);(HERE/'actual-audited262-frozen.json').write_text(json.dumps(x,indent=2)+'\n')
print(json.dumps({k:v for k,v in x.items()if k!='metadata_plan'}))
if'metadata_plan'in x:print(json.dumps({k:x['metadata_plan'][k]for k in('batches','questions','trajectories','domain_counts','batch_records_sha256','prior_identity_overlap','within_selected_training_hash_duplicates')}))
