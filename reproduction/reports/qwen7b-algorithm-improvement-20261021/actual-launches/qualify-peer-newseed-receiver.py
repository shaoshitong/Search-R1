"""Code-only stage, synthetic remote stdlib qualification and metadata preflight."""
import base64,hashlib,importlib.util,json,os,shlex,stat,subprocess,time,sys
from pathlib import Path

REPO=Path(__file__).resolve().parents[4]
CONTROL=REPO.parents[1]
PUB='755ed4ba3eef237ab2244c09c17cdac3966a43fc'
SNAPSHOT='newseed-20261021-185747-445fc691'
OUT=Path(__file__).resolve().parent
MODULE=REPO/'reproduction/operations/qwen7b_warmup/peer_newseed_receive.py'
TEST=MODULE.with_name('test_peer_newseed_receive.py')
for p in (MODULE,TEST):
 if p.read_bytes()!=subprocess.check_output(['git','show',PUB+':'+str(p.relative_to(REPO))],cwd=str(REPO)):
  raise ValueError('Unpublished receiver/test bytes')
actual=subprocess.check_output(['git','ls-remote','origin','refs/heads/codex/qwen7b-probe-warmup-20261019'],cwd=str(REPO)).split()[0].decode()
if actual!=PUB: raise ValueError('Origin receiver publication differs')
spec=importlib.util.spec_from_file_location('receiver',str(MODULE));receiver=importlib.util.module_from_spec(spec);spec.loader.exec_module(receiver)
address=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0']).decode().strip()
ssh=['ssh','-T','-B','en0','-b',address,'-oProxyCommand=none','-oProxyJump=none','-oBatchMode=yes','-oStrictHostKeyChecking=yes','-oConnectTimeout=8','hkust2']
if os.environ.get('PEER_QUALIFY_TARGET_ONLY')!='1':
    remote_code=receiver.install_code(ssh,receiver.K2/SNAPSHOT)
    probe='import json,hashlib,pathlib,time; p=pathlib.Path(%r); b=p.parents[2]; print(json.dumps(dict(code=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest(),receiver_started=(b/"receiver.lock").exists(),catalog_created=(b/"catalog.json").exists(),complete_exists=(b/"complete.json").exists(),epoch=time.time())))'%str(remote_code)
    stage=receiver.remote(ssh,'python3 -c '+shlex.quote(probe),timeout=45)
    assert stage['sha256']==hashlib.sha256(MODULE.read_bytes()).hexdigest()
    (OUT/'peer-newseed-receiver-HKUST2-stage.json').write_text(json.dumps(dict(source_commit=PUB,**stage),indent=2)+'\n')
    print(json.dumps(dict(stage='HKUST2_RECEIVER_CODE_STAGED_ONLY',observed=stage)),flush=True)

# Exact published two-file package on separate shared synthetic test namespace.
control_file=CONTROL/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py'
spec=importlib.util.spec_from_file_location('direct',str(control_file));control=importlib.util.module_from_spec(spec);spec.loader.exec_module(control)
fixture=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/peer-newseed-receiver-stdlib-fixture-20261021')/PUB

def target(code,timeout=55):
 r=control.run('python3 -c '+shlex.quote(code),account='mou176',source_address=address,timeout=timeout)
 if r.returncode: raise RuntimeError('Target synthetic control failed; no production receiver started')
 lines=[json.loads(x) for x in r.stdout.decode(errors='replace').splitlines() if x.startswith('{')]
 return lines[-1] if lines else {}

files={'fixture_pkg/__init__.py':b'', 'fixture_pkg/peer_newseed_receive.py':MODULE.read_bytes(), 'fixture_pkg/test_peer_newseed_receive.py':TEST.read_bytes()}
for relative,raw in files.items():
 p=fixture/relative; digest=hashlib.sha256(raw).hexdigest(); n=(len(raw)+4095)//4096
 for i in range(n):
  block=base64.b64encode(raw[i*4096:(i+1)*4096]).decode()
  code='import base64,pathlib; p=pathlib.Path(%r); assert not any(q.is_symlink() for q in (p,)+tuple(p.parents)); p.mkdir(parents=True,exist_ok=True); f=p/%r; b=base64.b64decode(%r); assert not f.exists() or f.read_bytes()==b; f.write_bytes(b) if not f.exists() else None'%(str(p.parent),p.name+'.%05d.chunk'%i,block)
  assert len(('python3 -c '+shlex.quote(code)).encode())<16000;target(code)
 code='import hashlib,pathlib; p=pathlib.Path(%r); p.parent.mkdir(parents=True,exist_ok=True); b=b"".join((p.parent/(p.name+".%%05d.chunk"%%i)).read_bytes() for i in range(%d)); assert hashlib.sha256(b).hexdigest()==%r; assert not p.exists() or p.read_bytes()==b; p.write_bytes(b) if not p.exists() else None'%(str(p),n,digest)
 target(code)

qualification='''import hashlib,json,pathlib,sys,tempfile,time,unittest,io
root=pathlib.Path(%r);sys.path.insert(0,str(root));tmp=root/'synthetic-tmp';tmp.mkdir(exist_ok=True);tempfile.tempdir=str(tmp)
from fixture_pkg import peer_newseed_receive as m
from fixture_pkg import test_peer_newseed_receive as t
original=m.atomic; suppressed=[]
def test_atomic(path,value):
 if pathlib.Path(path).name=='complete.json':
  assert str(path).startswith(str(tmp)+'/') and value.get('status')==m.STATUS
  suppressed.append(str(path));return
 original(path,value)
m.atomic=test_atomic
skip={'test_strict_routes_no_private_key_copy_secret_argv_or_proxy','test_rsync_transport_excludes_target_host_and_secret'}
suite=unittest.TestSuite(t.ReceiveTests(name) for name in unittest.defaultTestLoader.getTestCaseNames(t.ReceiveTests) if name not in skip);log=io.StringIO();result=unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
(root/'synthetic-tests.log').write_text(log.getvalue())
assert not list(tmp.rglob('complete.json'))
v=dict(status='SYNTHETIC_ONLY_REMOTE_STDLIB_QUALIFICATION_PASSED' if result.wasSuccessful() else 'SYNTHETIC_ONLY_REMOTE_STDLIB_QUALIFICATION_FAILED',python=sys.version.split()[0],tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),synthetic_complete_writes_suppressed=len(suppressed),production_receiver_started=False,HF_catalog_created=False,receiver_sha256=hashlib.sha256((root/'fixture_pkg/peer_newseed_receive.py').read_bytes()).hexdigest(),test_sha256=hashlib.sha256((root/'fixture_pkg/test_peer_newseed_receive.py').read_bytes()).hexdigest(),epoch=time.time())
(root/'qualification.json').write_text(json.dumps(v,sort_keys=True)+'\\n');print(json.dumps(v));sys.exit(0 if result.wasSuccessful() else 1)
'''%str(fixture)
actual=target(qualification,timeout=55)
(OUT/'peer-newseed-receiver-HPC3-stdlib-qualification.json').write_text(json.dumps(dict(source_commit=PUB,fixture_root=str(fixture),**actual),indent=2)+'\n')
print(json.dumps(actual),flush=True)

if os.environ.get('PEER_QUALIFY_TARGET_ONLY')=='1':sys.exit(0)

# Private official metadata only, never resolve archives or inherit proxies.
token_path=Path.home()/'.cache/huggingface/token'
if stat.S_IMODE(token_path.stat().st_mode)!=0o600: raise ValueError('Private HF RAM token input must have mode600')
token=token_path.read_text().strip()
preflight='''import json,requests,socket,sys,time
secret=json.loads(sys.stdin.readline());out=[]
for label,port in [('direct',None),('HKUST2-17894',17894),('HKUST2-17897',17897),('HKUST2-17898',17898)]:
 started=time.monotonic();value=dict(exit=label)
 try:
  if port:
   s=socket.create_connection(('127.0.0.1',port),timeout=1);s.close()
  with requests.Session() as session:
   session.trust_env=False;session.headers['Authorization']='Bearer '+secret['hf_token']
   if port:session.proxies={'http':'http://127.0.0.1:'+str(port),'https':'http://127.0.0.1:'+str(port)}
   response=session.get('https://huggingface.co/api/datasets/Sutoonq/Qwen25-7B-Probe-H100-Transfer-20261020',timeout=(3,5))
   value['http_status']=response.status_code
   if response.status_code==200:
    v=response.json();value['status']='PRIVATE_REPO_MATCH' if v.get('private') is True and v.get('id','').lower()=='sutoonq/qwen25-7b-probe-h100-transfer-20261020' else 'REPOSITORY_ID_OR_PRIVACY_MISMATCH'
   else:value['status']='HTTP_REJECTED'
 except Exception as error:value.update(status='FAILED',error_type=type(error).__name__)
 value['elapsed_seconds']=time.monotonic()-started;out.append(value)
print(json.dumps(dict(status='READ_ONLY_OFFICIAL_HF_METADATA_PREFLIGHT',exits=out,proxy_environment_inherited=False,Mac_proxy_used=False,archive_bytes_downloaded=0,epoch=time.time())))
'''
result=receiver.remote(ssh,'python3 -c '+shlex.quote(preflight),dict(hf_token=token),timeout=55);token=None
(OUT/'peer-newseed-receiver-HKUST2-HF-preflight.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result),flush=True)
# Second independent stage read, no active data writer or fake catalog.
second=receiver.remote(ssh,'python3 -c '+shlex.quote(probe),timeout=45)
assert second['sha256']==stage['sha256'] and not second['receiver_started'] and not second['catalog_created'] and not second['complete_exists']
(OUT/'peer-newseed-receiver-HKUST2-stage-observation2.json').write_text(json.dumps(second,indent=2)+'\n')
print(json.dumps(dict(stage='CODE_ONLY_SECOND_FRESH_VERIFICATION',observed=second)),flush=True)
