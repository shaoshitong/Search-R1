"""Stage independent CP1 acceptance and run only CPU steps in owned743978."""
import argparse,base64,gzip,hashlib,importlib.util,json,shlex,subprocess
from pathlib import Path
P='reproduction/operations/qwen7b_warmup/'
FILES=[P+n for n in ('direct_count_large_small_checkpoint.py','test_direct_count_large_small_checkpoint.py','deploy_large_count_small_checkpoint.py')]
TARGET='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/direct-count-frozenE1-fit30016-hold5120-v2'
BASE='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/direct-count-large-smallCP1-independent-v2'
CONFIG_SHA='fb637c02812dd070cf392a58df75e0d219bbb10d1a2d24cd6e201f3638147767'
BRANCH='codex/large-count-smallcp-audit-20261021'
COMMON=r'''
import pathlib,json,hashlib,os,fcntl,time,subprocess,base64,gzip
b=pathlib.Path(v['base']);b.mkdir(exist_ok=True);code=b/('code-'+v['commit'][:12]);code.mkdir(exist_ok=True);target=pathlib.Path(v['target'])
def proof(p):
 p=pathlib.Path(p);raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def read(p):return json.loads(pathlib.Path(p).read_bytes())
def immutable(p,val):
 raw=json.dumps(val,sort_keys=True,separators=(',',':'),allow_nan=False).encode()+b'\n'
 if p.exists():assert p.read_bytes()==raw,'Existing smallCP audit source differs'
 else:p.write_bytes(raw)
lock=(b/'operator.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
def allocation():
 s=subprocess.check_output(['scontrol','show','job','-o','743978'],universal_newlines=True)
 fields={k:s.split(k+'=',1)[1].split()[0]for k in ('JobId','JobName','UserId','JobState','Command','WorkDir')}
 assert fields['JobId']=='743978'and fields['JobName']=='q7b-frozenE1-head30016'and fields['JobState']=='RUNNING'
 assert fields['UserId'].endswith('('+str(os.getuid())+')')and fields['Command']==str(target/'capacity.sbatch')and fields['WorkDir']==str(target)
 return fields
'''
CHUNK=r'''
raw=base64.b64decode(v['part']);assert hashlib.sha256(raw).hexdigest()==v['part_sha'];p=code/('part-%04d'%v['index'])
if p.exists():assert p.read_bytes()==raw
else:p.write_bytes(raw)
print(json.dumps(dict(status='SMALL_CP_AUDIT_SOURCE_CHUNK_ACCEPTED',index=v['index'])))
'''
STAGE=r'''
data=b''.join((code/('part-%04d'%i)).read_bytes()for i in range(v['parts']));assert hashlib.sha256(data).hexdigest()==v['bundle_sha'];payload=json.loads(gzip.decompress(data));assert set(payload)==set(v['files'])
for name,text in payload.items():
 raw=base64.b64decode(text);assert len(raw)==v['files'][name]['bytes']and hashlib.sha256(raw).hexdigest()==v['files'][name]['sha256'];p=code/name
 if p.exists():assert p.read_bytes()==raw
 else:p.write_bytes(raw);p.chmod(0o444)
assert proof(target/'config.json')['sha256']==v['config_sha'];c=read(target/'config.json');runtime=pathlib.Path(c['implementation']['path']).parents[3];source=read(target/'source-publication.json')
assert source['commit'].startswith('805123aebf8a')and source['code_root']==str(runtime)and proof(c['implementation']['path'])==c['implementation']
pub=dict(commit=v['commit'],files={k:proof(code/k)for k in payload},code_root=str(code),config=proof(target/'config.json'),producer_publication=proof(target/'source-publication.json'),runtime_root=str(runtime),jobid=743978,old_model_payload_read=False)
immutable(b/'source-publication.json',pub);print(json.dumps(dict(status='SMALL_CP1_CPU_AUDITOR_SOURCE_STAGED',publication=proof(b/'source-publication.json'))))
'''
RUN=r'''
pub=read(b/'source-publication.json');assert pub['commit']==v['commit']
for ref in pub['files'].values():assert proof(ref['path'])==ref
job=allocation();runtime=pathlib.Path(pub['runtime_root']);env=dict(os.environ,QWEN7B_CODE_ROOT=str(runtime),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1')
cmd=['srun','--jobid=743978','--overlap','--gres=gpu:0','--ntasks=1','--cpus-per-task=2','--mem=3G','bash',str(runtime/'reproduction/operations/qwen7b_warmup/runtime.sh')]
if v['action']=='fixture':
 script="import importlib.util,sys,unittest;import reproduction.operations.qwen7b_warmup as pkg;from pathlib import Path;root=Path("+repr(str(code))+");name='reproduction.operations.qwen7b_warmup.direct_count_large_small_checkpoint';s=importlib.util.spec_from_file_location(name,root/'direct_count_large_small_checkpoint.py');m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);pkg.direct_count_large_small_checkpoint=m;s=importlib.util.spec_from_file_location('reproduction.operations.qwen7b_warmup.test_direct_count_large_small_checkpoint',root/'test_direct_count_large_small_checkpoint.py');t=importlib.util.module_from_spec(s);s.loader.exec_module(t);r=unittest.TextTestRunner().run(unittest.defaultTestLoader.loadTestsFromModule(t));assert r.wasSuccessful();print(__import__('json').dumps(dict(status='SMALL_CP1_SYNTHETIC_CPU_FIXTURE_PASSED',CUDA_initialized=__import__('torch').cuda.is_initialized())))"
 cmd+=['-c',script]
else:
 assert(b/'fixture-complete.json').exists(),'Fixture must pass before real CP1'
 cp=target/'checkpoints/cursor-000001.pt'
 if not cp.exists():print(json.dumps(dict(status='ACTUAL_CP1_NOT_YET_PRESENT',actual_checkpoint_loaded=False,job=job)));raise SystemExit(0)
 assert 0<cp.stat().st_size<=64*1024**2,'Only small new-head CP1; never load legacy model'
 cmd+=['-u',str(code/'direct_count_large_small_checkpoint.py'),'--base',str(target),'--config-sha256',v['config_sha'],'--output',str(b/'independent-CP1-acceptance.json')]
started=time.time();r=subprocess.run(cmd,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=42)
log=b/(v['action']+'-cpu.log');log.write_text(r.stdout+r.stderr)
if r.returncode:
 immutable(b/(v['action']+'-failure.json'),dict(returncode=r.returncode,log=proof(log),publication=proof(b/'source-publication.json')));raise RuntimeError(r.stderr[-2000:])
rows=[json.loads(x)for x in r.stdout.splitlines()if x.startswith('{')];assert rows
out=dict(status='OWNED_ALLOCATION_CPU_STEP_COMPLETED',result=rows[-1],seconds=time.time()-started,job=job,log=proof(log),source=proof(b/'source-publication.json'),GPU=0,CPU=2)
if v['action']=='fixture':immutable(b/'fixture-complete.json',out)
else:immutable(b/'actual-step-complete.json',out)
print(json.dumps(out))
'''
STATUS=r'''
cp=target/'checkpoints/cursor-000001.pt';out=dict(epoch=time.time(),CP1_exists=cp.exists(),CP1_bytes=cp.stat().st_size if cp.exists()else None,accepted=(b/'independent-CP1-acceptance.json').exists(),source=proof(b/'source-publication.json')if(b/'source-publication.json').exists()else None)
if out['accepted']:out.update(result=read(b/'independent-CP1-acceptance.json'),receipt=proof(b/'independent-CP1-acceptance.json'))
print(json.dumps(out))
'''


def main():
 p=argparse.ArgumentParser();p.add_argument('--control-root',type=Path,required=True);p.add_argument('--commit',required=True);p.add_argument('--action',choices=('stage','fixture','audit','status'),required=True);a=p.parse_args();repo=Path(__file__).resolve().parents[3];payload={};files={}
 for name in FILES:
  raw=subprocess.check_output(['git','show',a.commit+':'+name],cwd=repo);assert raw==(repo/name).read_bytes();key=Path(name).name;payload[key]=base64.b64encode(raw).decode();files[key]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
 values=dict(base=BASE,target=TARGET,config_sha=CONFIG_SHA,commit=a.commit,action=a.action,files=files);blocks=dict(chunk=CHUNK,stage=STAGE,fixture=RUN,audit=RUN,status=STATUS)
 spec=importlib.util.spec_from_file_location('small_cp_private_control',a.control_root/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl);ip=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0'],text=True).strip()
 def call(action,extra=None):
  script='v='+repr(dict(values,**(extra or {})))+'\n'+COMMON+'\n'+blocks[action];compile(script,action,'exec');command='python3 -c '+shlex.quote(script);assert len(command.encode())<16000;r=ctl.run(command,account='mou176',source_address=ip,timeout=55);rows=[json.loads(x)for x in r.stdout.decode(errors='replace').splitlines()if x.startswith('{')]
  if r.returncode or not rows:raise RuntimeError(r.stdout.decode(errors='replace')[-2200:]+r.stderr.decode(errors='replace')[-1800:])
  return rows[-1]
 if a.action=='stage':
  assert subprocess.check_output(['git','ls-remote','origin','refs/heads/'+BRANCH],cwd=repo,text=True).split()[0]==a.commit;blob=gzip.compress(json.dumps(payload,sort_keys=True).encode(),mtime=0)
  for i,start in enumerate(range(0,len(blob),4000)):
   part=blob[start:start+4000];call('chunk',dict(index=i,part=base64.b64encode(part).decode(),part_sha=hashlib.sha256(part).hexdigest()))
  out=call('stage',dict(parts=(len(blob)+3999)//4000,bundle_sha=hashlib.sha256(blob).hexdigest()))
 else:out=call(a.action)
 target=a.control_root/'outputs/qwen7b-warmup-20261019/reports'/('large-count-smallcp-'+a.action+'.json');target.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n');print(json.dumps(out))


if __name__=='__main__':main()
