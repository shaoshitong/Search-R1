"""Final54 CPU full-G10 audit using existing two own live allocations."""
import argparse,base64,hashlib,importlib.util,json,subprocess
from pathlib import Path
R=Path(__file__).resolve().parents[3];P=Path(__file__).parent
REPORT=R/'reproduction/reports/qwen7b-algorithm-improvement-20261021/hpc3-tail-plan-completion';A='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021';OUT=A+'/hpc3-tail-completion-final54-v1';TAIL=A+'/hpc3-tail-newseedG5-20261021-v1'
def main():
 p=argparse.ArgumentParser();p.add_argument('--action',choices=('stage','prepare','run','status','finalize'),required=True);p.add_argument('--worker',type=int,choices=(0,1));p.add_argument('--commit');p.add_argument('--allocation-job');p.add_argument('--allocation-job-name');args=p.parse_args();commit=args.commit or subprocess.check_output(['git','rev-parse','HEAD'],cwd=R).decode().strip();origin=subprocess.check_output(['git','ls-remote','origin','refs/heads/codex/qwen7b-probe-warmup-20261019'],cwd=R).decode().split()[0]
 if commit!=origin and subprocess.run(['git','merge-base','--is-ancestor',commit,origin],cwd=R).returncode:raise ValueError('Published source required')
 for f in(P/'hpc3_tail_final54_full_audit.py',Path(__file__)):
  if subprocess.check_output(['git','show',commit+':'+str(f.relative_to(R))],cwd=R)!=f.read_bytes():raise ValueError('Unpublished full-delta entry')
 spec=importlib.util.spec_from_file_location('transport',P/'deploy_hpc3_tail_g5.py');t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t);v=dict(out=OUT,tail=TAIL,commit=commit,code=OUT+'/code-'+commit[:12]);common='V='+repr(v)+'\n'+'''import pathlib,json,hashlib,os,subprocess,time,sys,fcntl
out=pathlib.Path(V['out']);code=pathlib.Path(V['code']);out.mkdir(exist_ok=True);code.mkdir(exist_ok=True)
def read(p):return json.loads(pathlib.Path(p).read_bytes())
def proof(p):
 p=pathlib.Path(p);raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def once(p,raw):
 if p.exists():assert p.read_bytes()==raw,'Immutable audit source/config differs'
 else:
  with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
'''
 if args.action=='stage':
  raw=(P/'hpc3_tail_final54_full_audit.py').read_bytes()
  for i in range(0,len(raw),4096):t.remote(common+'part='+repr(dict(i=i//4096,data=base64.b64encode(raw[i:i+4096]).decode()))+'\n'+'''import base64
once(code/('part-%04d'%part['i']),base64.b64decode(part['data']));print(json.dumps(dict(part=part['i'])))''')
  result=t.remote(common+'spec='+repr(dict(parts=(len(raw)+4095)//4096,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))+'\n'+'''raw=b''.join((code/('part-%04d'%i)).read_bytes()for i in range(spec['parts']));assert len(raw)==spec['bytes']and hashlib.sha256(raw).hexdigest()==spec['sha256'];compile(raw.decode(),'deltaaudit','exec');once(code/'audit.py',raw);pub=dict(source_commit=V['commit'],code=proof(code/'audit.py'),frozen_batches=54,questions=6818,trajectories_G10=68180,CPU_only=True);once(code/'source-publication.json',(json.dumps(pub,sort_keys=True)+'\\n').encode());print(json.dumps(dict(status='FINAL54_G10_AUDIT_CODE_STAGED',publication=proof(code/'source-publication.json'))))''')
 elif args.action in('prepare','finalize'):
  result=t.remote(common+'action='+repr(args.action)+'\n'+'''env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2');pub=read(code/'source-publication.json');assert pub['source_commit']==V['commit']and proof(code/'audit.py')==pub['code'];r=subprocess.run([sys.executable,str(code/'audit.py'),'--action',action],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,universal_newlines=True,env=env,timeout=150);(out/('actual-'+action+'.log')).write_text(r.stdout)
if r.returncode:raise RuntimeError(r.stdout[-3000:])
rows=[json.loads(x)for x in r.stdout.splitlines()if x.startswith('{')];assert rows;print(json.dumps(rows[-1]))''',timeout=165)
 elif args.action=='run':
  if args.worker is None or not args.allocation_job or not args.allocation_job_name:p.error('--worker, --allocation-job and --allocation-job-name required')
  step='''import pathlib,json,os
b=pathlib.Path(X['tail']);conf=json.loads((b/'config.json').read_bytes());root=conf['qualified_runtime_root'];assert os.environ.get('SLURM_JOB_ID')==X['allocation_job'];env=dict(os.environ,CUDA_VISIBLE_DEVICES='',QWEN7B_CODE_ROOT=root,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1');os.execvpe('bash',['bash',root+'/reproduction/operations/qwen7b_warmup/runtime.sh',X['audit'],'--action','worker','--worker',str(X['worker']),'--allocation-job',X['allocation_job']],env)
'''
  result=t.remote(common+'worker='+str(args.worker)+'\nallocation_job='+repr(args.allocation_job)+'\nallocation_name='+repr(args.allocation_job_name)+'\nstep='+repr(step)+'\n'+'''d=out/('worker-%d'%worker);d.mkdir(exist_ok=True);launch=(d/'launch.lock').open('a+');fcntl.flock(launch,fcntl.LOCK_EX|fcntl.LOCK_NB)
if(d/'complete.json').exists():print(json.dumps(dict(status='ALREADY_COMPLETE_NO_REAUDIT',complete=proof(d/'complete.json'))));sys.exit(0)
guard=(d/'audit.lock').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(guard,fcntl.LOCK_UN);guard.close();assert not(d/'controller.json').exists(),'Previous controller exists: reconcile actual state, no blind replay';pub=read(code/'source-publication.json');assert pub['source_commit']==V['commit']and proof(code/'audit.py')==pub['code'];import re;job=allocation_job;actual=subprocess.check_output(['scontrol','show','job',job,'-o'],universal_newlines=True);assert re.search(r'(?:^|\s)JobId='+re.escape(job)+r'(?:\s|$)',actual)and re.search(r'(?:^|\s)JobName='+re.escape(allocation_name)+r'(?:\s|$)',actual);assert int(re.search(r'UserId=[^ (]+\((\d+)\)',actual).group(1))==os.getuid()and 'JobState=RUNNING'in actual and int(re.search(r'NumCPUs=(\d+)',actual).group(1))>=4,'Explicit owned live allocation identity/resource mismatch';assert(out/'prepared.json').exists();x=dict(tail=V['tail'],worker=worker,allocation_job=job,audit=str(code/'audit.py'));script='X='+repr(x)+'\\n'+step;intent=dict(allocation_job_id=job,allocation_job_name=allocation_name,allocation_identity=actual,source_producer_job_id=('745518','745519')[worker],worker=worker,source_publication=proof(code/'source-publication.json'),frozen=proof(out/'frozen-seals.json'),CPU=2,GPU=0,epoch=time.time());once(d/'launch-intent.json',(json.dumps(intent,sort_keys=True)+'\\n').encode());cmd=['srun','--jobid='+job,'--overlap','--nodes=1','--ntasks=1','--cpus-per-task=2','--gres=gpu:0','--mem=3G','env','CUDA_VISIBLE_DEVICES=','python3','-c',script];log=d/'step.log'
with log.open('ab')as stream:child=subprocess.run(cmd,stdout=stream,stderr=stream,timeout=2200)
if child.returncode:raise RuntimeError('Own CPU final54 audit exited '+str(child.returncode)+'; log='+str(log))
assert(d/'complete.json').exists();print(json.dumps(dict(status='OWN_CPU_FINAL54_WORKER_FINISHED',worker=worker,complete=proof(d/'complete.json'),summary=read(d/'complete.json'),log=proof(log)),sort_keys=True))''',timeout=2220)
 else:
  result=t.remote(common+'''workers=[]
for w in(0,1):
 d=out/('worker-%d'%w);v=dict(worker=w)
 for name in('controller','progress','complete'):
  p=d/(name+'.json')
  if p.exists():v[name]=dict(ref=proof(p),value=read(p))
 p=d/'step.log'
 if p.exists():v['log_tail']=p.read_text(errors='replace')[-1600:]
 workers.append(v)
print(json.dumps(dict(epoch=time.time(),workers=workers)))''')
 REPORT.mkdir(parents=True,exist_ok=True);dest=REPORT/('final54-'+args.action+(''if args.worker is None else'-worker%d'%args.worker)+'.json');dest.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,sort_keys=True))
if __name__=='__main__':main()
