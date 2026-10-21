"""Audit the nine unattempted fixed members; preserve original failure and proofs."""
import hashlib,importlib.util,json,pathlib,subprocess,shlex
R=pathlib.Path(__file__).resolve().parents[4];HERE=pathlib.Path(__file__).parent;x=json.loads((HERE/'actual-independent262-launch.json').read_bytes())
worker='''import pathlib,json,sys,hashlib,importlib.util,types,fcntl,time,os,resource
x=json.loads(sys.stdin.buffer.read());root=pathlib.Path(x['root']);p=pathlib.Path('/proc/%d/stat'%x['pid']);assert not p.exists(),'Original proof worker still running';g=(root/'controller.lock').open('a');fcntl.flock(g,fcntl.LOCK_EX|fcntl.LOCK_NB);os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2]);resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3))
code=root/'code/reproduction/operations/qwen7b_warmup/peer_newseed_audit_reference.py';assert hashlib.sha256(code.read_bytes()).hexdigest()=='3def055adebfdb38e9326b9db385338c44800160c1b0a64e1ee78d8c3b68b906';s=importlib.util.spec_from_file_location('pinned',code);a=importlib.util.module_from_spec(s);s.loader.exec_module(a);ref=json.loads((root/'parent505-reference.json').read_bytes());selected=[r for r in ref['delta_batches_metadata']if r['existing_audit']['status']!='VERIFIED'];B=pathlib.Path(ref['source_base']);aud=a.Audit(types.SimpleNamespace(base=str(B),audit_dir=str(root/'audit'),path_map=[],max_new_batches=0));aud.root_config=a.read(B/'config.json');aud.root_config_hash=a.sha(B/'config.json');out=dict(status='FIXED262_TAIL9_PINNED_AUDIT_COMPLETED',source_reference=x['reference'],start_epoch=time.time(),members=[],producer_modified=False)
for item in selected[253:]:
 inp=B/item['input'];assert a.sha(inp)==item['input_sha256'];assert a.sha(B/item['seal'])==item['seal_sha256']
 try:
  proof=aud.batch(inp,a.read(inp),item['input_sha256']);row=dict(input=item['input'],status='VERIFIED',questions=proof['questions'],proof=str(root/'audit/batches'/inp.parent.name/(inp.stem+'.json')))
 except Exception as e:row=dict(input=item['input'],status='FAILED',error=type(e).__name__,message=str(e))
 out['members'].append(row);a.atomic(root/'tail9-progress.json',dict(out,epoch=time.time()))
out['verified']=sum(r['status']=='VERIFIED'for r in out['members']);out['failed']=9-out['verified'];out['epoch']=time.time();a.atomic(root/'tail9-complete.json',out);print(json.dumps(out))'''
s=importlib.util.spec_from_file_location('ctl','/Users/shaoshitong/Documents/remote_codex_control/control.py');ctl=importlib.util.module_from_spec(s);s.loader.exec_module(ctl)
r=subprocess.run(ctl.route('h100-10','python3 -c '+shlex.quote(worker)),input=json.dumps(x).encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=55)
if r.returncode:raise RuntimeError(r.stderr.decode(errors='replace')[-1200:])
x=json.loads(r.stdout.decode().splitlines()[-1]);(HERE/'actual-tail9-audit.json').write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(x))
