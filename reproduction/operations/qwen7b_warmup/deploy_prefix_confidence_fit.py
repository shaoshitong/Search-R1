"""One immutable, bounded CPU matched2/full8 fit after complete data extraction."""
import argparse,base64,hashlib,importlib.util,json,shlex,subprocess
from pathlib import Path
from .deploy_confirmation_materialization import make_bundle
from . import prefix_confidence_data as d
BRANCH='codex/prefix-confidence-incremental-20261021';P='reproduction/operations/qwen7b_warmup/';ROOT=d.BASE/'fit-deployment-v2';MANIFEST='_prefix_confidence_source_manifest.json';SEEDS=tuple(P+n for n in ('prefix_confidence_pipeline.py','test_prefix_confidence_pipeline.py','prefix_confidence_calibration.py','test_prefix_confidence_calibration.py','runtime.sh'));PROTOCOL='reproduction/reports/qwen7b-algorithm-improvement-20261021/prefix-confidence-PROTOCOL.md';PROTOCOL_COMMIT='a1ec8bbe44a24deb9ada77b59f3edf57cdc653e2'
COMMON=r'''
import base64,fcntl,gzip,hashlib,io,json,os,subprocess,tarfile,time
from pathlib import Path
b=Path(v['base']);b.mkdir(parents=True,exist_ok=True);experiment=b.parent
def read(p):return json.loads(Path(p).read_text())
def fp(p):
 p=Path(p);h=hashlib.sha256()
 with p.open('rb') as f:
  for x in iter(lambda:f.read(4194304),b''):h.update(x)
 return dict(path=str(p),bytes=p.stat().st_size,sha256=h.hexdigest())
def atomic(p,x):
 p=Path(p);q=p.with_name(p.name+'.tmp');q.write_text(json.dumps(x,sort_keys=True));q.replace(p)
def once(p,x):
 if p.exists():assert read(p)==x
 else:atomic(p,x)
def guard():
 g=(b/'operator.lock').open('a');fcntl.flock(g,fcntl.LOCK_EX|fcntl.LOCK_NB);return g
'''
INIT=r'''
g=guard();once(b/'bundle-metadata.json',v);print(json.dumps(dict(status='PREFIX_CONFIDENCE_FIT_BUNDLE_PREPARED')))
'''
CHUNK=r'''
g=guard();p=b/('bundle-%04d.part'%v['index']);x=base64.b64decode(v['data']);assert hashlib.sha256(x).hexdigest()==v['sha256']
if p.exists():assert p.read_bytes()==x
else:p.write_bytes(x)
print(json.dumps(dict(status='CPU_FIT_CODE_CHUNK_STORED',index=v['index'])))
'''
EXTRACT=r'''
g=guard();meta=read(b/'bundle-metadata.json');payload=b''.join(p.read_bytes() for p in sorted(b.glob('bundle-*.part')));assert len(payload)==meta['bundle_bytes'] and hashlib.sha256(payload).hexdigest()==meta['bundle_sha256'];code=b/('code-'+meta['commit'][:12]);stage=code.with_name(code.name+'.stage')
with tarfile.open(fileobj=io.BytesIO(payload),mode='r:gz') as tar:
 manifest=json.loads(tar.extractfile(v['manifest']).read());assert manifest['commit']==meta['commit'];files=manifest['files'];assert all(x.isfile() and not Path(x.name).is_absolute() and '..' not in Path(x.name).parts for x in tar.getmembers())
 if not code.exists():
  stage.mkdir(exist_ok=True)
  for x in tar.getmembers():
   data=tar.extractfile(x).read();p=stage/x.name;p.parent.mkdir(parents=True,exist_ok=True)
   if x.name in files:assert len(data)==files[x.name]['bytes'] and hashlib.sha256(data).hexdigest()==files[x.name]['sha256']
   if p.exists():assert p.read_bytes()==data
   else:p.write_bytes(data)
   p.chmod(0o444)
  stage.rename(code)
for name,x in files.items():assert fp(code/name)==dict(path=str(code/name),**x)
pub=dict(commit=meta['commit'],code_root=str(code),files=files,manifest=fp(code/v['manifest']),code_only=True);once(b/'source-publication.json',pub);env=dict(os.environ,QWEN7B_CODE_ROOT=str(code),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2')
r=subprocess.run(['bash',str(code/'reproduction/operations/qwen7b_warmup/runtime.sh'),'-m','unittest','reproduction.operations.qwen7b_warmup.test_prefix_confidence_calibration','reproduction.operations.qwen7b_warmup.test_prefix_confidence_pipeline'],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True,timeout=50);(b/'actual-cpu-tests.log').write_text(r.stdout);assert r.returncode==0,r.stdout[-2500:]
r=subprocess.run(['bash',str(code/'reproduction/operations/qwen7b_warmup/runtime.sh'),'-m','reproduction.operations.qwen7b_warmup.prefix_confidence_pipeline','--action','prepare','--code-root',str(code),'--commit',meta['commit'],'--protocol-commit',v['protocol_commit']],env=env,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True,timeout=45);(b/'actual-input-preflight.log').write_text(r.stdout);assert r.returncode==0,r.stdout[-2500:]
prepared=dict(status='V2_IMMUTABLE_FIT_SOURCE_CPU_FIXTURES_AND_ACTUAL_INPUT_PREFLIGHT_PASSED',publication=fp(b/'source-publication.json'),tests=fp(b/'actual-cpu-tests.log'),config=fp(experiment/'fit-config-v2.json'),protocol=fp(code/'reproduction/reports/qwen7b-algorithm-improvement-20261021/prefix-confidence-PROTOCOL.md'),code_root=str(code),external_payload_opened=False,real_optimizer_started=False);once(b/'preparation.json',prepared);print(json.dumps(prepared))
'''

def render_script(code,sha):
 return '''#!/usr/bin/env bash
set -euo pipefail
export QWEN7B_CODE_ROOT=@@CODE@@ CUDA_VISIBLE_DEVICES=''
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
exec 9>@@ROOT@@/allocation.lock
flock -n 9
set +e
bash "$QWEN7B_CODE_ROOT/reproduction/operations/qwen7b_warmup/runtime.sh" -u -m reproduction.operations.qwen7b_warmup.prefix_confidence_pipeline --action run --config-sha256 @@SHA@@
result=$?
python3 - @@ROOT@@ "$result" <<'EXIT'
import json,sys,time
from pathlib import Path
b=Path(sys.argv[1]);p=b/'exit.json';q=p.with_suffix('.tmp');q.write_text(json.dumps(dict(returncode=int(sys.argv[2]),epoch=time.time())));q.replace(p)
EXIT
exit "$result"
'''.replace('@@CODE@@',shlex.quote(code)).replace('@@ROOT@@',shlex.quote(str(ROOT))).replace('@@SHA@@',shlex.quote(sha))
START=r'''
g=guard();prep=read(b/'preparation.json');pub=read(b/'source-publication.json');assert fp(b/'source-publication.json')==prep['publication'] and fp(experiment/'fit-config-v2.json')==prep['config']
for name,x in pub['files'].items():assert fp(Path(pub['code_root'])/name)==dict(path=str(Path(pub['code_root'])/name),**x)
if (b/'launch.json').exists():print(json.dumps(dict(status='EXISTING_SINGLE_FIT_LAUNCH_PRESERVED',launch=read(b/'launch.json'))))
else:
 assert not (b/'launch-intent.json').exists(),'Unknown prior launch: inspect actual PID/locks before retry';script=b/'analysis.sh'
 if script.exists():assert script.read_text()==v['runner']
 else:script.write_text(v['runner'])
 subprocess.check_call(['bash','-n',str(script)]);atomic(b/'launch-intent.json',dict(config=prep['config'],script=fp(script),time=time.time()))
 log=(b/'analysis.log').open('ab');p=subprocess.Popen(['bash',str(script)],stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True);log.close();birth=Path('/proc/%d/stat'%p.pid).read_text().rsplit(')',1)[1].split()[19];value=dict(pid=p.pid,start_ticks=birth,config=prep['config'],script=fp(script),started=time.time(),CPU=2,GPUs=0);atomic(b/'launch.json',value);print(json.dumps(dict(status='BOUNDED_CPU_TWO_MATCHED_MODES_STARTED_NOT_PROGRESS_VERIFIED',launch=value)))
'''
STATUS=r'''
x=dict(status='ACTUAL_CPU_PREFIX_CONFIDENCE_TWO_MODE_FIT_STATE',observed_epoch=time.time())
for name in ('launch.json','exit.json'):
 p=b/name
 if p.exists():x[name[:-5]]=read(p)
for name in ('fit-started.json','progress.json','failure.json','frozen-candidate-and-gate.json','external-progress.json'):
 p=experiment/'fit'/name
 if p.exists():x[name[:-5]]=read(p)
if (experiment/'fit/internal-complete.json').exists():x['internal_complete_ref']=fp(experiment/'fit/internal-complete.json');x['internal_complete']=read(experiment/'fit/internal-complete.json')
if (experiment/'fit/complete.json').exists():x['complete_ref']=fp(experiment/'fit/complete.json');x['complete']=read(experiment/'fit/complete.json')
if (b/'analysis.log').exists():x['log_tail']=(b/'analysis.log').read_text()[-2000:]
print(json.dumps(x))
'''
def main():
 p=argparse.ArgumentParser();p.add_argument('--control-root',type=Path,required=True);p.add_argument('--commit',required=True);p.add_argument('--action',choices=('stage','start','status'),required=True);a=p.parse_args();repo=Path(__file__).resolve().parents[3];spec=importlib.util.spec_from_file_location('prefix_fit_control',a.control_root/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');control=importlib.util.module_from_spec(spec);spec.loader.exec_module(control);address=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0'],text=True).strip()
 def call(src,extra=None):
  v=dict(base=str(ROOT));v.update(extra or {});cmd='python3 -c '+shlex.quote('v='+repr(v)+'\n'+COMMON+src)
  if len(cmd.encode())>=16000:raise ValueError('CPU control command must stay below16KB')
  r=control.run(cmd,account='mou176',source_address=address,timeout=55);rows=[json.loads(x) for x in r.stdout.decode(errors='replace').splitlines() if x.startswith('{')]
  if r.returncode or not rows:raise RuntimeError('Actual bounded CPU outcome unconfirmed: '+r.stderr.decode(errors='replace')[-2000:]+' '+r.stdout.decode(errors='replace')[-2500:])
  return rows[-1]
 if a.action=='stage':
  assert subprocess.check_output(['git','ls-remote','origin','refs/heads/'+BRANCH],cwd=repo,text=True).split()[0]==a.commit;paths=subprocess.check_output(['git','ls-tree','-r','--name-only',a.commit],cwd=repo,text=True).splitlines();cache={}
  def load(name):
   if name not in cache:cache[name]=subprocess.check_output(['git','show',a.commit+':'+name],cwd=repo)
   return cache[name]
  payload,report=make_bundle(paths,load,a.commit,seeds=SEEDS,resource_files=[PROTOCOL],manifest_name=MANIFEST);assert load(PROTOCOL)==subprocess.check_output(['git','show',PROTOCOL_COMMIT+':'+PROTOCOL],cwd=repo);meta=dict(commit=a.commit,bundle_bytes=len(payload),bundle_sha256=hashlib.sha256(payload).hexdigest(),closure=report);call(INIT,meta)
  for i,start in enumerate(range(0,len(payload),4096)):
   x=payload[start:start+4096];call(CHUNK,dict(index=i,data=base64.b64encode(x).decode(),sha256=hashlib.sha256(x).hexdigest()))
  result=call(EXTRACT,dict(manifest=MANIFEST,protocol_commit=PROTOCOL_COMMIT))
 elif a.action=='start':
  prep=call("print(json.dumps(read(b/'preparation.json')))");script=render_script(prep['code_root'],prep['config']['sha256']);subprocess.run(['bash','-n'],input=script,text=True,check=True);exports='\n'.join(x for x in script.splitlines() if x.startswith('export '));subprocess.run(['bash','-euc',exports+'\ntest "$QWEN7B_CODE_ROOT" = '+shlex.quote(prep['code_root'])],check=True);result=call(START,dict(runner=script))
 else:result=call(STATUS)
 print(json.dumps(result,sort_keys=True));out=a.control_root/'outputs/qwen7b-warmup-20261019/reports'/('prefix-confidence-fit-'+a.action+'-latest.json');out.write_text(json.dumps(result,sort_keys=True,indent=2)+'\n')
if __name__=='__main__':main()
