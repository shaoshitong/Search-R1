"""Single guarded launch of fixed262 proof on H100 source CPU, never producer."""
import hashlib,importlib.util,json,pathlib,shlex,subprocess,time
R=pathlib.Path(__file__).resolve().parents[4];HERE=pathlib.Path(__file__).parent;PREFIX='reproduction/operations/qwen7b_warmup/'
names=['peer_newseed_delta.py','peer_newseed_audit_reference.py','peer_newseed_proof_remaining262.py']
publication=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R,text=True).strip();code={PREFIX+n:(R/PREFIX/n).read_text()for n in names}
for n,raw in code.items():
 if subprocess.check_output(['git','show',publication+':'+n],cwd=R)!=raw.encode():raise ValueError('Unpublished source '+n)
remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/codex/qwen7b-probe-warmup-20261019'],cwd=R,text=True).split()[0]
if remote!=publication:raise ValueError('Current HEAD must be actual published branch')
reference=json.loads((HERE/'actual-source-after466.json').read_bytes());h=hashlib.sha256(code[PREFIX+'peer_newseed_proof_remaining262.py'].encode()).hexdigest()
root='/mnt/data/shaoshitong/qwen7b-newseed-to-hpc3-20261021/source-proof262-after466-'+reference['delta_records_sha256'][:12]+'-'+h[:8]
worker='''import pathlib,json,sys,hashlib,subprocess,os,fcntl,time
p=json.loads(sys.stdin.buffer.read());root=pathlib.Path(p['root']);root.mkdir(parents=True,exist_ok=True);code=root/'code';code.mkdir(exist_ok=True)
with(root/'launch.lock').open('a')as guard:
 fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 for name,raw in p['code'].items():
  dst=code/name;dst.parent.mkdir(parents=True,exist_ok=True)
  for folder in [dst.parent,dst.parent.parent,dst.parent.parent.parent]:
   init=folder/'__init__.py'
   if not init.exists():init.write_bytes(b'')
  if dst.exists():assert dst.read_bytes()==raw.encode()
  else:dst.write_text(raw)
 reference=root/'parent505-reference.json';blob=json.dumps(p['reference'],sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()+bytes([10]);rsha=hashlib.sha256(blob).hexdigest()
 if reference.exists():assert reference.read_bytes()==blob
 else:reference.write_bytes(blob)
 if(root/'launch-intent.json').exists():
  intent=json.loads((root/'launch-intent.json').read_bytes());assert intent['source_publication']==p['publication'];print(json.dumps(dict(status='EXISTING_LAUNCH_PRESERVED_NO_RETRY',root=str(root),intent=intent)));sys.exit(0)
 intent=dict(source_publication=p['publication'],source_reference=dict(path=str(reference),bytes=len(blob),sha256=rsha),code={name:hashlib.sha256(raw.encode()).hexdigest()for name,raw in p['code'].items()},epoch=time.time())
 (root/'launch-intent.json').write_text(json.dumps(intent))
 env=os.environ.copy();env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
 log=(root/'controller.log').open('ab',buffering=0)
 child=subprocess.Popen([sys.executable,'-u','-m','reproduction.operations.qwen7b_warmup.peer_newseed_proof_remaining262','--reference',str(reference),'--reference-sha256',rsha,'--output',str(root)],cwd=code,env=env,stdout=log,stderr=log,start_new_session=True)
 stat=pathlib.Path('/proc/%d/stat'%child.pid).read_text();receipt=dict(status='FIXED262_INDEPENDENT_CPU_PROOF_LAUNCHED',root=str(root),pid=child.pid,start_ticks=int(stat.split(') ')[1].split()[19]),source_publication=p['publication'],reference=intent['source_reference'],epoch=time.time());(root/'launch.json').write_text(json.dumps(receipt));print(json.dumps(receipt))'''
s=importlib.util.spec_from_file_location('ctl','/Users/shaoshitong/Documents/remote_codex_control/control.py');ctl=importlib.util.module_from_spec(s);s.loader.exec_module(ctl)
cmd='python3 -c '+shlex.quote(worker)
if len(cmd.encode())>=16000:raise ValueError('Oversized control command')
r=subprocess.run(ctl.route('h100-10',cmd),input=json.dumps(dict(root=root,reference=reference,code=code,publication=publication)).encode(),stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=55)
if r.returncode:raise RuntimeError(r.stderr.decode(errors='replace')[-1500:])
x=json.loads(r.stdout.decode().splitlines()[-1]);(HERE/'actual-independent262-launch.json').write_text(json.dumps(x,indent=2)+'\n');print(json.dumps(x))
