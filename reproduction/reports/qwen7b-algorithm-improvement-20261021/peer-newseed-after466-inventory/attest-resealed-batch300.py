"""Explicitly re-attest one metadata-resealed batch, without altering source."""
import hashlib,importlib.util,json,pathlib,subprocess,shlex
R=pathlib.Path(__file__).resolve().parents[4];HERE=pathlib.Path(__file__).parent;P='reproduction/operations/qwen7b_warmup/'
pub=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip();names=['peer_newseed_audit_reference.py','peer_newseed_audit_resealed_metadata.py'];code={P+n:(R/P/n).read_text()for n in names}
for n,b in code.items():
 if subprocess.check_output(['git','show',pub+':'+n],cwd=R)!=b.encode():raise ValueError('Publish attestation code first')
if subprocess.check_output(['git','ls-remote','origin','refs/heads/codex/qwen7b-probe-warmup-20261019'],cwd=R,text=True).split()[0]!=pub:raise ValueError('Remote source differs')
x=json.loads((HERE/'actual-independent262-launch.json').read_bytes());newsha=hashlib.sha256(code[P+names[-1]].encode()).hexdigest()
worker='''import json,pathlib,sys,hashlib,os,fcntl,types,time,resource
p=json.loads(sys.stdin.buffer.read());oldroot=pathlib.Path(p['launch']['root']);assert not pathlib.Path('/proc/%d/stat'%p['launch']['pid']).exists();guard=(oldroot/'controller.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);root=oldroot/('metadata-reseal-attestation-'+p['newsha'][:12]);root.mkdir(parents=True,exist_ok=True);code=root/'code'
for name,raw in p['code'].items():
 dst=code/name;dst.parent.mkdir(parents=True,exist_ok=True)
 for folder in[dst.parent,dst.parent.parent,dst.parent.parent.parent]:
  f=folder/'__init__.py'
  if not f.exists():f.write_bytes(b'')
 if dst.exists():assert dst.read_bytes()==raw.encode()
 else:dst.write_text(raw)
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2]);resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3));sys.path.insert(0,str(code));from reproduction.operations.qwen7b_warmup import peer_newseed_audit_resealed_metadata as a
reference=json.loads((oldroot/'parent505-reference.json').read_bytes());rows=[r for r in reference['delta_batches_metadata']if r['existing_audit']['status']!='VERIFIED'];r=rows[252];assert r['input']=='inputs/shard-01/batch-000300.json';B=pathlib.Path(reference['source_base']);inp=B/r['input'];assert a.sha(inp)==r['input_sha256']and a.sha(B/r['seal'])==r['seal_sha256'];aud=a.Audit(types.SimpleNamespace(base=str(B),audit_dir=str(root/'audit'),path_map=[],max_new_batches=0));aud.root_config=a.read(B/'config.json');aud.root_config_hash=a.sha(B/'config.json');started=time.time();result=aud.batch(inp,a.read(inp),r['input_sha256']);proof=root/'audit/batches'/inp.parent.name/(inp.stem+'.json');b=proof.read_bytes();receipt=dict(status='ONE_METADATA_RESEALED_BATCH_FRESH_SHA_AND_ORIGINAL_CHAIN_CAUSAL_RECONCILIATION_PASSED',input=r['input'],questions=result['questions'],trajectories=result['trajectories'],proof=dict(path=str(proof),bytes=len(b),sha256=hashlib.sha256(b).hexdigest()),source_publication=p['publication'],auditor_sha256=p['newsha'],pinned_original_auditor_sha256='3def055adebfdb38e9326b9db385338c44800160c1b0a64e1ee78d8c3b68b906',metadata_reseal_attestation=result['metadata_reseal_attestation'],independent_sha256=result['independent_sha256'],original_failure_preserved=True,producer_modified=False,strict_EM_tokenizer_recomputed=False,elapsed_seconds=time.time()-started,epoch=time.time());a.atomic(root/'complete.json',receipt);print(json.dumps(receipt))'''
s=importlib.util.spec_from_file_location('ctl','/Users/shaoshitong/Documents/remote_codex_control/control.py');ctl=importlib.util.module_from_spec(s);s.loader.exec_module(ctl)
r=subprocess.run(ctl.route('h100-10','python3 -c '+shlex.quote(worker)),input=json.dumps(dict(launch=x,code=code,newsha=newsha,publication=pub)).encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=55)
if r.returncode:raise RuntimeError(r.stderr.decode(errors='replace')[-1800:])
v=json.loads(r.stdout.decode().splitlines()[-1]);(HERE/'actual-batch300-reseal-attestation.json').write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v))
