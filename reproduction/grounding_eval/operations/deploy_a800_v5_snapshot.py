# Execute from the xia workspace root; immutable code-only snapshot.
import base64,hashlib,importlib.util,json,pathlib,shlex,subprocess,time
repo=pathlib.Path('work/grounding-eval-strict-20261019').resolve();commit='16c36aa777f684f245b1d38cb2939ef175131831'
paths=subprocess.check_output(['git','-C',str(repo),'ls-tree','-r','--name-only',commit,'--','reproduction/grounding_eval','search_r1/qa_protocol.py','search_r1/__init__.py'],text=True).splitlines()
files={}
for name in paths:
 raw=subprocess.check_output(['git','-C',str(repo),'show',commit+':'+name]);files[name]=dict(content=base64.b64encode(raw).decode(),sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw))
proofpath=pathlib.Path('outputs/grounding-eval-20261019/inventory-a800-hashes-retry.json');proof=proofpath.read_bytes()
payload=dict(root='/data/shaoshitong/grounding-eval-20261019',source_root='/data/shaoshitong/grounding-eval-20261019/runtime-16c36aa7',commit=commit,files=files,proof=dict(content=base64.b64encode(proof).decode(),sha256=hashlib.sha256(proof).hexdigest(),bytes=len(proof)))
script=r'''import pathlib,json,sys,hashlib,base64,os,datetime
p=json.load(sys.stdin);b=pathlib.Path(p['root']);assert str(b)=='/data/shaoshitong/grounding-eval-20261019'
b.mkdir(mode=0o700,exist_ok=True);runtime=pathlib.Path(p['source_root']);assert runtime.parent==b and runtime.name=='runtime-'+p['commit'][:8];runtime.mkdir(mode=0o700,exist_ok=True)
def put(path,item):
 raw=base64.b64decode(item['content']);assert len(raw)==item['bytes'] and hashlib.sha256(raw).hexdigest()==item['sha256'];path.parent.mkdir(parents=True,exist_ok=True)
 if path.exists():assert path.is_file() and not path.is_symlink() and path.read_bytes()==raw
 else:
  with path.open('xb')as f:os.fchmod(f.fileno(),0o600);f.write(raw);f.flush();os.fsync(f.fileno())
 assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
for name,item in p['files'].items():
 rel=pathlib.PurePosixPath(name);assert not rel.is_absolute() and '..'not in rel.parts and (name.startswith('reproduction/grounding_eval/') or name in ('search_r1/qa_protocol.py','search_r1/__init__.py'));put(runtime/name,item)
put(b/'inventory-a800-hashes-retry.json',p['proof'])
receipt=dict(checked_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_commit=p['commit'],runtime=str(runtime),files={k:{'sha256':v['sha256'],'bytes':v['bytes']}for k,v in p['files'].items()},checkpoint_proof_sha256=p['proof']['sha256'],gpu_started=False)
raw=json.dumps(receipt,sort_keys=True,indent=2).encode()+b'\n';put(runtime/'deployment-receipt.json',dict(content=base64.b64encode(raw).decode(),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
print(json.dumps(receipt))
'''
spec=importlib.util.spec_from_file_location('ctl','/Users/shaoshitong/Documents/remote_codex_control/control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl)
r=subprocess.run(ctl.route('baai-a800','/usr/bin/python3 -c '+shlex.quote(script)),input=json.dumps(payload),text=True,capture_output=True,timeout=80)
value=dict(transport_exit=r.returncode,stderr=r.stderr,receipt=json.loads(r.stdout)if not r.returncode else r.stdout)
pathlib.Path('outputs/grounding-eval-20261019/a800-deployment-receipt-16c36aa7.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))
