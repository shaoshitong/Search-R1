"""Immutable Count53312 code stage, bounded CPU prepare and guarded single submit."""
import argparse,base64,hashlib,importlib.util,json,pathlib,shlex,subprocess,zlib
R=pathlib.Path(__file__).resolve().parents[4];C=R.parents[1];HERE=pathlib.Path(__file__).parent;PKG='reproduction/operations/qwen7b_warmup/'
A='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021';BASE=A+'/count53312-frozenE1-data-scaling-v1';OWN=('count53312_train.py','count53312_incremental_cache.py','count53312_cache_layout.py','test_count53312_train.py','direct_count_large_screen.py')
COMMON='''import pathlib,json,hashlib,base64,subprocess,fcntl,os,time,re,shutil
b=pathlib.Path(v['base']);b.mkdir(exist_ok=True);guard=(b/'deployment.lock').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);code=b/('code-'+v['commit'][:12]);code.mkdir(exist_ok=True)
def read(p):return json.loads(pathlib.Path(p).read_bytes())
def proof(p):
 p=pathlib.Path(p);raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def immutable(p,x):
 if p.exists():assert read(p)==x,'Immutable metadata differs'
 else:
  with p.open('x')as f:f.write(json.dumps(x,sort_keys=True)+'\\n');f.flush();os.fsync(f.fileno())
def checked(ref):
 assert proof(ref['path'])==ref,'Fresh metadata SHA differs';return read(ref['path'])
def active():
 r=subprocess.run(['squeue','-h','-u','mou176','-n','q7b-count53312','-o','%i|%j|%T|%R'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=12);assert r.returncode==0,r.stderr[-1000:];return r.stdout.strip().splitlines()
'''
STAGE='''qroot=pathlib.Path(v['A'])/'coarse-event-cached-prepare-code-v1/code-c888a2281859';q=read(qroot/'cpu-qualification.json');original=checked(q['baseline_publication']);files={};origins={}
for name,ref in original['files'].items():
 src=qroot/name;actual=proof(src)
 if src.name in q['files']:assert actual==q['files'][src.name];origin='c888a2281859e1a930218d490aae5bbeb5adcc22'
 else:assert actual['bytes']==ref['bytes']and actual['sha256']==ref['sha256'];origin=original['commit']
 dst=code/name;dst.parent.mkdir(parents=True,exist_ok=True)
 if dst.exists():assert dst.read_bytes()==src.read_bytes()
 else:shutil.copyfile(str(src),str(dst));dst.chmod(0o444)
 files[name]=dict(bytes=actual['bytes'],sha256=actual['sha256']);origins[name]=origin
for name,ref in q['files'].items():
 if name in ('coarse_event_cached_train.py','test_coarse_event_cached_train.py'):continue
 src=pathlib.Path(ref['path']);assert proof(src)==ref;dst=code/v['pkg']/name
 if dst.exists():assert dst.read_bytes()==src.read_bytes()
 else:shutil.copyfile(str(src),str(dst));dst.chmod(0o444)
 files[v['pkg']+name]=dict(bytes=ref['bytes'],sha256=ref['sha256']);origins[v['pkg']+name]='282bc1597fd720c6e6fad5c836effb52d5e65d78'if name=='direct_count_large_small_checkpoint.py'else'c888a2281859e1a930218d490aae5bbeb5adcc22'
for name,ref in v['new'].items():
 raw=b''.join((code/(name+'.part%04d'%i)).read_bytes()for i in range(ref['parts']));assert len(raw)==ref['bytes']and hashlib.sha256(raw).hexdigest()==ref['sha256'];compile(raw,name,'exec');dst=code/v['pkg']/name
 if dst.exists():assert dst.read_bytes()==raw
 else:dst.write_bytes(raw);dst.chmod(0o444)
 files[v['pkg']+name]=dict(bytes=len(raw),sha256=ref['sha256']);origins[v['pkg']+name]=v['commit']
packet=checked(v['packet']);assert packet['baseline_config']['sha256']=='fb637c02812dd070cf392a58df75e0d219bbb10d1a2d24cd6e201f3638147767';assert packet['baseline_CP0']['sha256']=='acd1c4f4bc277a8cab83847f966cfbf2462b6391b6db1d9c52c9434447f562df'
pub=dict(commit=v['commit'],remote_sha=v['commit'],code_root=str(code),files=files,file_origins=origins,baseline_runtime_commit=original['commit'],input_packet=v['packet'],controller_source=v['controller_source'],new_native_rows=232960,old_rows_recomputed=0,full_bank_CUDA_resident=False);immutable(b/'source-publication.json',pub);print(json.dumps(dict(status='COUNT53312_IMMUTABLE_SOURCE_STAGED',publication=proof(b/'source-publication.json'),code_root=str(code),input_packet=v['packet'])))
'''
PREPARE='''assert not active(),'Existing own job must be reconciled';assert not(b/'submission-intent.json').exists(),'Unknown prior submission';pub=read(b/'source-publication.json');assert pub['commit']==v['commit'];assert checked(v['packet'])
for name in ('training.lock','prepare.lock','audit-first.lock'):
 f=(b/name).open('a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);f.close()
fixture="import os,resource,unittest,json,time;os.sched_setaffinity(0,sorted(os.sched_getaffinity(0))[:2]);resource.setrlimit(resource.RLIMIT_CPU,(55,55));started=time.time();suite=unittest.defaultTestLoader.loadTestsFromName('reproduction.operations.qwen7b_warmup.test_count53312_train');r=unittest.TextTestRunner(verbosity=2).run(suite);assert r.wasSuccessful();from reproduction.operations.qwen7b_warmup import count53312_train as m;prepared=m.prepare("+repr(v['packet']['path'])+");print(json.dumps(dict(status='COUNT53312_CPU_GATHER_AND_CP0_RECOVERY_QUALIFIED_ACTUAL_CONFIG_PREPARED',tests=r.testsRun,elapsed=time.time()-started,peak_RSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,prepared=prepared,real_model_tensors_loaded=False,GPU_submitted=False)))"
env=dict(os.environ,QWEN7B_CODE_ROOT=str(code),CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1');log=b/'cpu-prepare.log';peak=0;deadline=time.monotonic()+60;failure=None
with log.open('a')as stream:
 child=subprocess.Popen(['bash',str(code/v['pkg']/'runtime.sh'),'-u','-c',fixture],stdout=stream,stderr=subprocess.STDOUT,env=env)
 while child.poll()is None:
  try:
   rss=[int(line.split()[1])for line in pathlib.Path('/proc/%d/status'%child.pid).read_text().splitlines()if line.startswith('VmRSS:')];peak=max([peak]+rss)
  except FileNotFoundError:pass
  if peak>3*1024**2 or time.monotonic()>deadline:failure='owned CPU prepare RSS/wall bound exceeded';child.kill();break
  time.sleep(.1)
 child.wait()
assert not failure and child.returncode==0,(failure or'fixture/metadata failed')+' '+log.read_text()[-4000:]
result=json.loads([r for r in log.read_text().splitlines()if r.startswith('{')][-1]);result.update(log=proof(log),publication=proof(b/'source-publication.json'),controller_commit=v['commit'],entry=proof(b/'training.sbatch'),config=proof(b/'config.json'),monitor_peak_RSS_KiB=peak);subprocess.check_call(['bash','-n',str(b/'training.sbatch')]);immutable(b/'cpu-qualification.json',result);print(json.dumps(dict(result=result,qualification=proof(b/'cpu-qualification.json'))))
'''
SUBMIT='''assert not active(),'Same-name own job already active';assert not(b/'submission-intent.json').exists(),'Prior unknown submit intent: reconcile before retry';pub=read(b/'source-publication.json');assert pub['commit']==v['commit'];q=read(b/'cpu-qualification.json');assert q['tests']==3
for key in ('publication','entry','config'):assert proof(q[key]['path'])==q[key]
for name in ('training.lock','prepare.lock','audit-first.lock'):
 f=(b/name).open('a+');fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);f.close()
immutable(b/'submission-intent.json',dict(controller=v['commit'],entry=q['entry'],config=q['config'],epoch=time.time(),held_before_identity_release=True));r=subprocess.run(['sbatch','--hold','--parsable','--account=mou176',str(b/'training.sbatch')],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=25)
if r.returncode or not re.fullmatch(r'[0-9]+(?:;[^ ]+)?',r.stdout.strip()):raise RuntimeError('Submission uncertain; retained intent, inspect before retry: '+r.stdout[-500:]+r.stderr[-1000:])
job=r.stdout.strip().split(';')[0];raw=subprocess.check_output(['scontrol','show','job','-o',job],universal_newlines=True)
for item in ('JobName=q7b-count53312','QOS=normal','Partition=acd_u','Account=mou176','TimeLimit=03:00:00','MinMemoryNode=128G','NumNodes=1','Command='+str(b/'training.sbatch'),'WorkDir='+str(b),'Dependency=(null)'):assert item in raw,item
assert re.search(r'NumCPUs=8(?: |$)',raw)and('gres/gpu=1'in raw or'gres:gpu=1'in raw)
identity=dict(job_id=job,job=raw,controller=v['commit'],entry=q['entry'],config=q['config'],epoch=time.time());immutable(b/'submission-identity.json',identity);subprocess.check_call(['scontrol','release',job]);result=dict(status='COUNT53312_SINGLE_JOB_SUBMITTED_RELEASED',job_id=job,identity=proof(b/'submission-identity.json'),entry=q['entry'],config=q['config']);immutable(b/'submission-result.json',result);print(json.dumps(result))
'''
STATUS='''out=dict(epoch=time.time(),active_same_name_jobs=active())
for name in ('source-publication.json','preparation.json','cpu-qualification.json','submission-intent.json','submission-identity.json','submission-result.json','controller.json','progress.json','first-update-replay.json','first-cp-independent-audit.json','latest.json','failure.json','complete.json','evaluation-progress.json'):
 if(b/name).exists():
  try:out[name[:-5]]=read(b/name)
  except FileNotFoundError:pass
for root in ('added-cache',''):
 p=b/root/('progress.json'if root else'cache-complete.json')
 if p.exists():out['added_cache_progress'if root else'cache_complete']=read(p)
print(json.dumps(out))
'''
def main():
 p=argparse.ArgumentParser();p.add_argument('--action',choices=('stage','prepare','submit','status'),required=True);p.add_argument('--packet',required=True,help='Local small descriptor JSON {path,bytes,sha256}, not raw payload');a=p.parse_args();packet=json.loads(pathlib.Path(a.packet).read_text());own=pathlib.Path(__file__);commit=subprocess.check_output(['git','log','-1','--format=%H','--',str(own.relative_to(R))],cwd=R).decode().strip();remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/codex/qwen7b-probe-warmup-20261019'],cwd=R).decode().split()[0];subprocess.check_call(['git','merge-base','--is-ancestor',commit,remote],cwd=R);assert subprocess.check_output(['git','show',commit+':'+str(own.relative_to(R))],cwd=R)==own.read_bytes();spec=importlib.util.spec_from_file_location('count53312_private',C/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl);ip=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0']).decode().strip();source=dict(path=str(own.relative_to(R)),bytes=own.stat().st_size,sha256=hashlib.sha256(own.read_bytes()).hexdigest())
 def call(body,**extra):
  values=dict(base=BASE,A=A,pkg=PKG,commit=commit,controller_source=source,packet=packet,**extra);script='v='+repr(values)+'\n'+COMMON+body;compile(script,'count53312-remote','exec');packed=base64.b64encode(zlib.compress(script.encode())).decode();cmd='python3 -c '+shlex.quote('import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(packed)+')))');assert len(cmd.encode())<16000;r=ctl.run(cmd,account='mou176',source_address=ip,timeout=70)
  if r.returncode:raise RuntimeError(r.stdout[-4500:]+r.stderr[-800:])
  return json.loads(r.stdout.splitlines()[-1])
 if a.action=='stage':
  refs={}
  for name in OWN:
   raw=subprocess.check_output(['git','show',commit+':'+PKG+name],cwd=R);assert raw==(R/PKG/name).read_bytes();refs[name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),parts=(len(raw)+4095)//4096)
   for i,start in enumerate(range(0,len(raw),4096)):
    chunk=raw[start:start+4096];call("raw=base64.b64decode(v['data']);assert hashlib.sha256(raw).hexdigest()==v['sha'];p=code/(v['name']+'.part%04d'%v['i'])\nif p.exists():assert p.read_bytes()==raw\nelse:p.write_bytes(raw)\nprint(json.dumps(dict(code_part_accepted=True,index=v['i'])))",name=name,i=i,data=base64.b64encode(chunk).decode(),sha=hashlib.sha256(chunk).hexdigest())
  result=call(STAGE,new=refs)
 else:result=call({'prepare':PREPARE,'submit':SUBMIT,'status':STATUS}[a.action])
 (HERE/('actual-'+a.action+'.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result),flush=True)
if __name__=='__main__':main()
