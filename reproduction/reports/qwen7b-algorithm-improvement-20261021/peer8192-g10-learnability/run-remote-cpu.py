"""One immutable, CPU-only remote analysis launch; archives only code/config."""
import base64,gzip,hashlib,importlib.util,io,json,pathlib,shlex,subprocess,tarfile
ROOT=pathlib.Path(__file__).resolve().parents[4]
assert (ROOT/'reproduction/operations/qwen7b_warmup/peer_g10_learnability.py').is_file()
CONTROL=ROOT.parents[1]
B='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/peer8192-g10-learnability-20261021'
P='reproduction/operations/qwen7b_warmup/'
FILES=[P+n for n in ('peer_g10_learnability.py','g10_grouping.py','replay_probe_layers.py','peer_newseed_train.py','peer_newseed_fit_operation.py','runtime.sh')]
commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT).decode().strip();remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/codex/qwen7b-probe-warmup-20261019'],cwd=ROOT).split()[0].decode();assert commit==remote
stream=io.BytesIO();files={}
with gzip.GzipFile(fileobj=stream,mode='wb',mtime=0,filename='') as gz:
 with tarfile.open(fileobj=gz,mode='w') as tar:
  for name in FILES:
   raw=(ROOT/name).read_bytes();assert raw==subprocess.check_output(['git','show',commit+':'+name],cwd=ROOT)
   files[name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest());item=tarfile.TarInfo(name);item.size=len(raw);item.mode=0o444;tar.addfile(item,io.BytesIO(raw))
data=stream.getvalue();code=B+'/code-'+commit[:12]
spec=importlib.util.spec_from_file_location('learnability_cpu_private_control',CONTROL/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl);ip=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0']).decode().strip()
def call(source,timeout=55):
 command='python3 -c '+shlex.quote(source);assert len(command.encode())<16000
 result=ctl.run(command,account='mou176',source_address=ip,timeout=timeout)
 if result.returncode:raise RuntimeError('CPU action not confirmed: '+result.stderr.decode(errors='replace')[-1000:]+' '+result.stdout.decode(errors='replace')[-1000:])
 values=[json.loads(line) for line in result.stdout.decode().splitlines() if line.startswith('{')];return values[-1] if values else {}
common='import base64,fcntl,gzip,hashlib,io,json,os,pathlib,subprocess,tarfile,time\nb=pathlib.Path(%r);c=pathlib.Path(%r);b.mkdir(parents=True,exist_ok=True)\ng=(b/"operator.lock").open("a");fcntl.flock(g,fcntl.LOCK_EX|fcntl.LOCK_NB)\n'%(B,code)
for i,start in enumerate(range(0,len(data),4096)):
 block=data[start:start+4096];source=common+'p=b/%r;x=base64.b64decode(%r);assert hashlib.sha256(x).hexdigest()==%r\nif p.exists():assert p.read_bytes()==x\nelse:p.write_bytes(x)\n'%(('code-%s-%05d.chunk'%(commit[:12],i)),base64.b64encode(block).decode(),hashlib.sha256(block).hexdigest());call(source)
manifest=dict(source_commit=commit,code_root=code,files=files,bundle_bytes=len(data),bundle_sha256=hashlib.sha256(data).hexdigest())
source=common+'expected='+repr(manifest)+'''\nparts=sorted(b.glob('code-'+expected['source_commit'][:12]+'-*.chunk'));data=b''.join(p.read_bytes() for p in parts);assert len(data)==expected['bundle_bytes'] and hashlib.sha256(data).hexdigest()==expected['bundle_sha256']
if not c.exists():
 c.mkdir()
 with tarfile.open(fileobj=io.BytesIO(data),mode='r:gz') as tar:
  assert {x.name for x in tar.getmembers()}==set(expected['files'])
  for x in tar.getmembers():
   rel=pathlib.Path(x.name);assert x.isfile() and not rel.is_absolute() and '..' not in rel.parts;raw=tar.extractfile(x).read();v=expected['files'][x.name];assert len(raw)==v['bytes'] and hashlib.sha256(raw).hexdigest()==v['sha256'];p=c/rel;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(raw);p.chmod(0o444)
for n,v in expected['files'].items():
 p=c/n;assert p.stat().st_size==v['bytes'] and hashlib.sha256(p.read_bytes()).hexdigest()==v['sha256']
p=b/'source-publication.json'
if p.exists():assert json.loads(p.read_text())==expected
else:p.write_text(json.dumps(expected,sort_keys=True)+'\\n')
print(json.dumps(dict(status='CPU_ANALYSIS_CODE_ALL_SHA_STAGED',source_commit=expected['source_commit'],files=len(expected['files']))))
''';print(json.dumps(call(source)))
binding=json.loads((ROOT/'reproduction/reports/qwen7b-algorithm-improvement-20261021/actual-launches/peer-batched-actual-raw-and-input-acceptance.json').read_bytes())
def code_ref(name):return dict(files[P+name],path=code+'/'+P+name)
config=dict(preparation=binding['preparation'],raw_complete=binding['raw_complete'],grouping_code=code_ref('g10_grouping.py'),training_plan_source=code_ref('peer_newseed_train.py'),training_factory_source=code_ref('peer_newseed_fit_operation.py'),optional_actual_fit_config='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/peer-newseed8192-g10-train-v1/config.json')
source=common+'config='+repr(config)+'''\np=b/'config.json'
if p.exists():assert json.loads(p.read_text())==config
else:p.write_text(json.dumps(config,sort_keys=True)+'\\n')
if (b/'output/complete.json').exists():print(json.dumps(dict(status='EXISTING_COMPLETE_PRESERVED')))
elif (b/'launch.json').exists():print(json.dumps(dict(status='EXISTING_LAUNCH_PRESERVED',launch=json.loads((b/'launch.json').read_text()))))
else:
 assert not (b/'launch-intent.json').exists(),'Unknown prior outcome; do not repeat'
 (b/'launch-intent.json').write_text(json.dumps(dict(source_commit=%r,code_root=str(c),epoch=time.time())))
 env=dict(os.environ,QWEN7B_CODE_ROOT=str(c),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
 with (b/'analysis.log').open('ab') as log:child=subprocess.Popen(['bash',str(c/'reproduction/operations/qwen7b_warmup/runtime.sh'),'-u','-m','reproduction.operations.qwen7b_warmup.peer_g10_learnability','--config',str(p),'--output',str(b/'output')],env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 birth=pathlib.Path('/proc/%%d/stat'%%child.pid).read_text().rsplit(')',1)[1].split()[19];value=dict(status='CPU_ANALYSIS_STARTED_NOT_PROGRESS_VERIFIED',pid=child.pid,start_ticks=birth,code_root=str(c),source_commit=%r,CUDA_VISIBLE_DEVICES='',epoch=time.time());(b/'launch.json').write_text(json.dumps(value));print(json.dumps(value))
'''%(commit,commit)
value=call(source);path=pathlib.Path(__file__).with_name('actual-launch.json');path.write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(value))
