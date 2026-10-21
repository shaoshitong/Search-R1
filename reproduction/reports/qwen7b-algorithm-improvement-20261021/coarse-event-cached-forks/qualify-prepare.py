"""Code-only immutable CPU qualification; never submits or waits for GPU data."""
import argparse,base64,hashlib,importlib.util,json,pathlib,shlex,subprocess,zlib
ROOT=pathlib.Path(__file__).resolve().parents[4];CONTROL=ROOT.parents[1];OUT=pathlib.Path(__file__).parent;PKG='reproduction/operations/qwen7b_warmup/'
p=argparse.ArgumentParser();p.add_argument('--commit',required=True);v=p.parse_args()
remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/codex/qwen7b-probe-warmup-20261019'],cwd=ROOT).decode().split()[0]
if remote!=v.commit:raise ValueError('Publish exact source before code-only CPU qualification')
names=('coarse_event_cached_train.py','test_coarse_event_cached_train.py','prior_logit_event_loss.py','direct_count_large_small_checkpoint.py');blobs={}
for n in names:
 origin='282bc1597fd720c6e6fad5c836effb52d5e65d78'if n=='direct_count_large_small_checkpoint.py'else v.commit
 raw=subprocess.check_output(['git','show',origin+':'+PKG+n],cwd=ROOT)
 if raw!=(ROOT/PKG/n).read_bytes():raise ValueError('Unpublished source differs')
 blobs[n]=raw
helper_remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/codex/large-count-smallcp-audit-20261021'],cwd=ROOT).decode().split()[0]
if helper_remote!='282bc1597fd720c6e6fad5c836effb52d5e65d78':raise ValueError('Published small-checkpoint helper origin differs')
spec=importlib.util.spec_from_file_location('coarse_prepare_private_control',CONTROL/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl);ip=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0']).decode().strip()
common="""import pathlib,json,hashlib,base64,fcntl,subprocess,os,shutil,time
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/coarse-event-cached-prepare-code-v1');b.mkdir(exist_ok=True);lock=(b/'operator.lock').open('a+');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);code=b/('code-'+v['commit'][:12]);code.mkdir(exist_ok=True)
def proof(p):
 p=pathlib.Path(p);raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
"""
def call(body,extra=None):
 script='v='+repr(dict(commit=v.commit,**(extra or {})))+'\n'+common+body;compile(script,'remote-code-only','exec');packed=base64.b64encode(zlib.compress(script.encode())).decode();command='python3 -c '+shlex.quote('import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(packed)+')))')
 if len(command.encode())>=16000:raise ValueError('Code-only command exceeds16KB')
 r=ctl.run(command,account='mou176',source_address=ip,timeout=70)
 if r.returncode:raise RuntimeError(r.stderr[-3000:])
 return json.loads(r.stdout.splitlines()[-1])
refs={}
for n,raw in blobs.items():
 refs[n]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),parts=(len(raw)+4095)//4096)
 for i,start in enumerate(range(0,len(raw),4096)):
  part=raw[start:start+4096];call("raw=base64.b64decode(v['data']);assert hashlib.sha256(raw).hexdigest()==v['sha'];p=code/(v['name']+'.part%04d'%v['index'])\nif p.exists():assert p.read_bytes()==raw\nelse:p.write_bytes(raw)\nprint(json.dumps(dict(code_part_accepted=True,index=v['index'])))",dict(name=n,index=i,data=base64.b64encode(part).decode(),sha=hashlib.sha256(part).hexdigest()))
body="""source=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/direct-count-frozenE1-fit30016-hold5120-v2/source-publication.json');pub=json.loads(source.read_text());assert pub['commit']=='805123aebf8af83a7c1c44947f39c294f7aeb624'
for name,ref in pub['files'].items():
 src=pathlib.Path(pub['code_root'])/name;actual=proof(src);assert actual['bytes']==ref['bytes']and actual['sha256']==ref['sha256'];dst=code/name;dst.parent.mkdir(parents=True,exist_ok=True)
 if dst.exists():assert dst.read_bytes()==src.read_bytes()
 else:shutil.copyfile(str(src),str(dst));dst.chmod(0o444)
for name,ref in v['files'].items():
 raw=b''.join((code/(name+'.part%04d'%i)).read_bytes()for i in range(ref['parts']));assert len(raw)==ref['bytes']and hashlib.sha256(raw).hexdigest()==ref['sha256'];compile(raw,name,'exec');dst=code/('reproduction/operations/qwen7b_warmup/'+name)
 if dst.exists():assert dst.read_bytes()==raw
 else:dst.write_bytes(raw);dst.chmod(0o444)
receipt=code/'cpu-qualification.json'
if receipt.exists():raise RuntimeError('Existing immutable qualification; inspect instead of repeating')
fixture="""+repr("""import os,resource,unittest,json,time
os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2]);resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3));resource.setrlimit(resource.RLIMIT_CPU,(55,55))
started=time.time();suite=unittest.defaultTestLoader.loadTestsFromName('reproduction.operations.qwen7b_warmup.test_coarse_event_cached_train');r=unittest.TextTestRunner(verbosity=2).run(suite);assert r.wasSuccessful()
from reproduction.operations.qwen7b_warmup import coarse_event_cached_train as m
print(json.dumps(dict(status='SYNTHETIC_COARSE_HEAD_CP0_REPLAY_RECOVERY_CPU_QUALIFIED_REAL_INPUTS_NOT_YET_ACCEPTED',tests=r.testsRun,actual_input_status=m.waiting_inputs(None),GPU_used=False,real_model_or_cache_tensors_loaded=False,new_model_scoring=False,active_Count_or_screen_modified=False,elapsed_seconds=time.time()-started,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)))
""")+"""
env=dict(os.environ,QWEN7B_CODE_ROOT=str(code),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
r=subprocess.run(['bash',str(code/'reproduction/operations/qwen7b_warmup/runtime.sh'),'-u','-c',fixture],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,env=env,universal_newlines=True,timeout=65);log=code/'cpu-fixtures.log';log.write_text(r.stdout)
if r.returncode:raise RuntimeError(r.stdout[-3000:])
result=json.loads([s for s in r.stdout.splitlines()if s.startswith('{')][-1]);result.update(source_commit=v['commit'],small_checkpoint_helper_origin='282bc1597fd720c6e6fad5c836effb52d5e65d78',code_root=str(code),baseline_publication=proof(source),files={n:proof(code/('reproduction/operations/qwen7b_warmup/'+n))for n in v['files']},log=proof(log));receipt.write_text(json.dumps(result,sort_keys=True)+'\\n');print(json.dumps(dict(result=result,receipt=proof(receipt))))
"""
result=call(body,dict(files=refs));(OUT/'actual-cpu-qualification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
