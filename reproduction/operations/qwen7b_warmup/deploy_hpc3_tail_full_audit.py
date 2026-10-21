"""Own existing allocation CPU2/gpu0 audit of fixed73 batches; no GPU submit."""
import argparse,base64,hashlib,importlib.util,json,subprocess
from pathlib import Path
R=Path(__file__).resolve().parents[3];P=Path(__file__).parent
REPORT=R/'reproduction/reports/qwen7b-algorithm-improvement-20261021/hpc3-tail73-full-new5-audit'
A='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021';OUT=A+'/hpc3-tail73-full-new5-audit-v1';TAIL=A+'/hpc3-tail-newseedG5-20261021-v1'
def main():
 p=argparse.ArgumentParser();p.add_argument('--action',choices=('stage','run','status','finalize'),required=True);p.add_argument('--worker',type=int,choices=(0,1));p.add_argument('--commit');args=p.parse_args();commit=args.commit or subprocess.check_output(['git','rev-parse','HEAD'],cwd=R).decode().strip();origin=subprocess.check_output(['git','ls-remote','origin','refs/heads/codex/qwen7b-probe-warmup-20261019'],cwd=R).decode().split()[0]
 if commit!=origin and subprocess.run(['git','merge-base','--is-ancestor',commit,origin],cwd=R).returncode:raise ValueError('Published source required')
 for f in(P/'hpc3_tail_frozen_full_audit.py',Path(__file__)):
  if subprocess.check_output(['git','show',commit+':'+str(f.relative_to(R))],cwd=R)!=f.read_bytes():raise ValueError('Unpublished audit entry')
 spec=importlib.util.spec_from_file_location('transport',P/'deploy_hpc3_tail_g5.py');t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)
 v=dict(out=OUT,tail=TAIL,commit=commit,code=OUT+'/code-'+commit[:12]);common='V='+repr(v)+'\n'+'''import pathlib,json,hashlib,os,subprocess,time,sys,fcntl
out=pathlib.Path(V['out']);code=pathlib.Path(V['code']);out.mkdir(exist_ok=True);code.mkdir(exist_ok=True)
def read(p):return json.loads(pathlib.Path(p).read_bytes())
def proof(p):
 p=pathlib.Path(p);raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def once(p,raw):
 if p.exists():assert p.read_bytes()==raw,'Immutable full audit source differs'
 else:
  with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
'''
 if args.action=='stage':
  raw=(P/'hpc3_tail_frozen_full_audit.py').read_bytes()
  for i in range(0,len(raw),4096):t.remote(common+'part='+repr(dict(i=i//4096,data=base64.b64encode(raw[i:i+4096]).decode()))+'\n'+'''import base64
piece=base64.b64decode(part['data']);once(code/('part-%04d'%part['i']),piece);print(json.dumps(dict(part=part['i'])))''')
  result=t.remote(common+'spec='+repr(dict(parts=(len(raw)+4095)//4096,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))+'\n'+'''raw=b''.join((code/('part-%04d'%i)).read_bytes()for i in range(spec['parts']));assert len(raw)==spec['bytes']and hashlib.sha256(raw).hexdigest()==spec['sha256'];compile(raw.decode(),'frozenfullaudit','exec');once(code/'audit.py',raw);pub=dict(source_commit=V['commit'],code=proof(code/'audit.py'),frozen_batches=73,questions=9344,new5_trajectories=46720,CPU_only=True);once(code/'source-publication.json',(json.dumps(pub,sort_keys=True)+'\\n').encode());print(json.dumps(dict(status='FIXED73_FULL_AUDIT_CODE_STAGED',publication=proof(code/'source-publication.json'))))''')
 elif args.action=='run':
  if args.worker is None:p.error('--worker required')
  step=r'''import pathlib,json,os
b=pathlib.Path(X['tail']);c=json.loads((b/'workers'/('worker-%d'%X['worker'])/'controller.json').read_bytes());s=(pathlib.Path('/proc')/str(c['pid'])/'stat').read_text().rsplit(')',1)[1].split();assert int(s[19])==int(c['birth_ticks'])and c['Slurm_job_id']==X['job'];assert b'hpc3_tail_g5.py'in(pathlib.Path('/proc')/str(c['pid'])/'cmdline').read_bytes()
conf=json.loads((b/'config.json').read_bytes());root=conf['qualified_runtime_root'];env=dict(os.environ,CUDA_VISIBLE_DEVICES='',QWEN7B_CODE_ROOT=root,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1');os.execvpe('bash',['bash',root+'/reproduction/operations/qwen7b_warmup/runtime.sh',X['audit'],'--worker',str(X['worker'])],env)
'''
  result=t.remote(common+'worker='+str(args.worker)+'\nstep='+repr(step)+'\n'+'''d=out/('worker-%d'%worker);d.mkdir(exist_ok=True);launch=(d/'launch.lock').open('a+');fcntl.flock(launch,fcntl.LOCK_EX|fcntl.LOCK_NB)
if(d/'complete.json').exists():print(json.dumps(dict(status='ALREADY_COMPLETE_NO_REAUDIT',complete=proof(d/'complete.json'))));sys.exit(0)
guard=(d/'audit.lock').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(guard,fcntl.LOCK_UN);guard.close()
pub=read(code/'source-publication.json');assert pub['source_commit']==V['commit']and proof(code/'audit.py')==pub['code'];job=(pathlib.Path(V['tail'])/('worker-%d.job.id'%worker)).read_text().strip();assert job==('745518','745519')[worker];state=subprocess.check_output(['squeue','-h','-j',job,'-o','%T'],universal_newlines=True).strip();assert state=='RUNNING','Own allocation must still be live'
x=dict(tail=V['tail'],worker=worker,job=job,audit=str(code/'audit.py'));script='X='+repr(x)+'\\n'+step
intent=dict(job_id=job,worker=worker,source_publication=proof(code/'source-publication.json'),frozen_batches=73,CPU=2,GPU=0,epoch=time.time());once(d/('launch-intent-'+V['commit'][:12]+'.json'),(json.dumps(intent,sort_keys=True)+'\\n').encode())
cmd=['srun','--jobid='+job,'--overlap','--nodes=1','--ntasks=1','--cpus-per-task=2','--gres=gpu:0','--mem=3G','env','CUDA_VISIBLE_DEVICES=','python3','-c',script];log=d/('step-'+V['commit'][:12]+'.log')
with log.open('ab')as stream:child=subprocess.run(cmd,stdout=stream,stderr=stream,timeout=1100)
if child.returncode:raise RuntimeError('Own CPU audit exited '+str(child.returncode)+'; log='+str(log))
assert(d/'complete.json').exists();print(json.dumps(dict(status='OWN_CPU_FULL_FROZEN_WORKER_FINISHED',worker=worker,complete=proof(d/'complete.json'),summary=read(d/'complete.json'),log=proof(log)),sort_keys=True))''',timeout=1120)
 else:
  result=t.remote(common+'''workers=[]
for w in(0,1):
 d=out/('worker-%d'%w);value=dict(worker=w)
 for name in('controller','progress','complete'):
  p=d/(name+'.json')
  if p.exists():value[name]=dict(ref=proof(p),value=read(p))
 workers.append(value)
print(json.dumps(dict(epoch=time.time(),workers=workers)))''')
  if args.action=='finalize':
   if not all('complete'in w for w in result['workers']):raise RuntimeError('Both workers must really complete')
   refs=[w['complete']['ref']for w in result['workers']]
   result=t.remote(common+'refs='+repr(refs)+'\n'+'''values=[]
for ref in refs:
 assert proof(ref['path'])==ref;values.append(read(ref['path']))
assert sum(x['batches']for x in values)==73 and sum(x['questions']for x in values)==9344 and sum(x['trajectories']for x in values)==46720 and all(x['strict_EM_mismatches']==0 and not x['CUDA_initialized']and not x['model_tensors_loaded']for x in values)
value=dict(status='FIXED73_9344_HPC3_NEW5_FULL_PAYLOAD_SHA_STRICT_EM_CAUSAL_NATIVE_SEEDS_QUALIFIED',worker_completes=refs,batches=73,questions=9344,trajectories=46720,strict_EM_recomputed=46720,strict_EM_mismatches=0,new5_full_source_qualified=True,old5_qualification='metadata_only_not_revalidated_here',G10_raw_accepted=False,current124194_fit_expanded=False,feature_bank_created=False,potentialOverlap=True,source_commit=V['commit'],epoch=time.time());once(out/'complete.json',(json.dumps(value,sort_keys=True)+'\\n').encode());print(json.dumps(dict(complete=proof(out/'complete.json'),summary=value)))''')
 REPORT.mkdir(parents=True,exist_ok=True);path=REPORT/('actual-'+args.action+(''if args.worker is None else'-worker%d'%args.worker)+'.json');path.write_text(json.dumps(result,indent=2)+'\n')
 if args.action=='status':print(json.dumps(dict(epoch=result['epoch'],workers=[dict(worker=w['worker'],progress=w.get('progress'),complete=w.get('complete',{}).get('ref'),controller=w.get('controller'))for w in result['workers']]),sort_keys=True))
 else:print(json.dumps(result,sort_keys=True))
if __name__=='__main__':main()
