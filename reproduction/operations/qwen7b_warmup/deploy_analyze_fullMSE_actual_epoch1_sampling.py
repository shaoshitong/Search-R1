"""Stage and durably execute one metadata-only actual E1 sampler audit."""
import argparse,base64,hashlib,importlib.util,json,shlex,subprocess,zlib
from pathlib import Path
R=Path(__file__).resolve().parents[3];CONTROL=R.parents[1];P='reproduction/operations/qwen7b_warmup/';BRANCH='codex/q7b-fullMSE750190-actualE1-sampler-20261023';NAME='analyze_fullMSE_actual_epoch1_sampling.py'
REMOTE='''import pathlib,json,base64,zlib,hashlib,fcntl,subprocess,os,time
A=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021');assert A.is_dir();out=A/'fullMSE-750190-actual-E1-sampler-diagnostic-20261023-v1';out.mkdir(exist_ok=True);code=out/('code-'+v['commit'][:12]);code.mkdir(exist_ok=True)
def read(p):return json.loads(p.read_bytes())
def proof(p):
 raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def ticks(pid):
 try:return pathlib.Path('/proc/%s/stat'%pid).read_text().rsplit(')',1)[1].split()[19]
 except(FileNotFoundError,ProcessLookupError):return None
raw=zlib.decompress(base64.b64decode(v['source']));assert hashlib.sha256(raw).hexdigest()==v['sha256'];compile(raw,'actual_E1_sampler','exec');worker=code/v['name']
if worker.exists():assert worker.read_bytes()==raw
else:worker.write_bytes(raw);worker.chmod(0o444)
r=dict(status='ACTUAL_E1_SAMPLER_METADATA_ANALYSIS_STAGED',source_commit=v['commit'],worker=proof(worker),noCP_or_RAW_or_plan_regeneration=True,Slurm_mutated=False)
if v['action']=='start':
 guard=(out/'launch.lock').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);launch=read(out/'launch.json')if(out/'launch.json').exists()else None
 if(out/'result.json').is_file():r.update(status='EXISTING_ACTUAL_SAMPLER_RESULT_REUSED',result=read(out/'result.json'),receipt=proof(out/'result.json'))
 elif launch and ticks(launch['pid'])==launch['start_ticks']:r.update(status='EXISTING_DURABLE_METADATA_ANALYSIS_REUSED',launch=launch)
 else:
  assert not(out/'failure.json').exists(),'Actual metadata failure requires diagnosis, not duplicate launch';python='/data/user/jxu957/experiments/cosearch-continue300-x203-20261008/runtime/retriever-python310-candidate/bin/python3.10';assert pathlib.Path(python).is_file();env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
  with(out/'analysis.log').open('ab')as stream:child=subprocess.Popen([python,'-u',str(worker),'--source-sha',v['sha256']],stdout=stream,stderr=stream,stdin=subprocess.DEVNULL,start_new_session=True,close_fds=True,env=env)
  launched=dict(pid=child.pid,start_ticks=ticks(child.pid),source=proof(worker),time=time.time());(out/'launch.json').write_text(json.dumps(launched,sort_keys=True)+chr(10));r.update(status='ACTUAL_SAMPLER_DURABLE_METADATA_ANALYSIS_STARTED',launch=launched)
for name in('launch.json','failure.json'):
 if(out/name).exists():r[name]=read(out/name)
if(out/'result.json').exists():r.update(receipt=proof(out/'result.json'),result=read(out/'result.json'))
print(json.dumps(r))
'''
def main():
 p=argparse.ArgumentParser();p.add_argument('--commit',required=True);p.add_argument('--action',choices=('stage','start','status'),default='stage');a=p.parse_args();remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+BRANCH],cwd=R,text=True).split()[0];subprocess.check_call(['git','merge-base','--is-ancestor',a.commit,remote],cwd=R);raw=subprocess.check_output(['git','show',a.commit+':'+P+NAME],cwd=R);v=dict(commit=a.commit,name=NAME,source=base64.b64encode(zlib.compress(raw)).decode(),sha256=hashlib.sha256(raw).hexdigest(),action=a.action);script='v='+repr(v)+'\n'+REMOTE;compile(script,'actual_sampler_remote','exec');cmd='python3 -c '+shlex.quote('import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(base64.b64encode(zlib.compress(script.encode())).decode())+')))');assert len(cmd.encode())<16000
 spec=importlib.util.spec_from_file_location('control',CONTROL/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl);ip=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0'],text=True).strip();reply=ctl.run(cmd,account='mou176',source_address=ip,timeout=35)
 if reply.returncode:raise RuntimeError(reply.stdout[-1800:]+reply.stderr[-1800:])
 result=json.loads(reply.stdout);out=R/'reproduction/reports/qwen7b-algorithm-improvement-20261021/fullMSE-750190-actual-E1-sampler';out.mkdir(parents=True,exist_ok=True);(out/('actual-'+a.action+'.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
compile(REMOTE,'actual_sampler_remote','exec')
if __name__=='__main__':main()
