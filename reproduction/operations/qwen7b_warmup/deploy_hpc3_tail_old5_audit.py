"""Stage and run the fixed9344 old5 complement in own live allocation CPU2."""
import argparse,base64,hashlib,importlib.util,json,subprocess
from pathlib import Path
R=Path(__file__).resolve().parents[3];P=Path(__file__).parent
REPORT=R/'reproduction/reports/qwen7b-algorithm-improvement-20261021/hpc3-tail9344-old5-g10-audit'
OUT='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/hpc3-tail9344-old5-qualified-g10-v1'
TAIL='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/hpc3-tail-newseedG5-20261021-v1'
def main():
 p=argparse.ArgumentParser();p.add_argument('--action',choices=('stage','run','status'),required=True);p.add_argument('--commit');a=p.parse_args();commit=a.commit or subprocess.check_output(['git','rev-parse','HEAD'],cwd=R).decode().strip();origin=subprocess.check_output(['git','ls-remote','origin','refs/heads/codex/qwen7b-probe-warmup-20261019'],cwd=R).decode().split()[0]
 if commit!=origin and subprocess.run(['git','merge-base','--is-ancestor',commit,origin],cwd=R).returncode:raise ValueError('Publish exact source before execution')
 for f in(P/'hpc3_tail_old5_subset_audit.py',Path(__file__)):
  if subprocess.check_output(['git','show',commit+':'+str(f.relative_to(R))],cwd=R)!=f.read_bytes():raise ValueError('Unpublished old5 entry')
 s=importlib.util.spec_from_file_location('transport',P/'deploy_hpc3_tail_g5.py');t=importlib.util.module_from_spec(s);s.loader.exec_module(t)
 v=dict(out=OUT,tail=TAIL,code=OUT+'/code-'+commit[:12],commit=commit);common='V='+repr(v)+'\n'+'''import pathlib,json,hashlib,os,sys,subprocess,time,fcntl
out=pathlib.Path(V['out']);code=pathlib.Path(V['code']);out.mkdir(exist_ok=True);code.mkdir(exist_ok=True)
def read(p):return json.loads(pathlib.Path(p).read_bytes())
def proof(p):
 p=pathlib.Path(p);raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def once(p,raw):
 if p.exists():assert p.read_bytes()==raw,'Immutable old5 source differs'
 else:
  with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
'''
 if a.action=='stage':
  raw=(P/'hpc3_tail_old5_subset_audit.py').read_bytes()
  for i in range(0,len(raw),4096):t.remote(common+'part='+repr(dict(i=i//4096,data=base64.b64encode(raw[i:i+4096]).decode()))+'\n'+'''import base64
once(code/('part-%04d'%part['i']),base64.b64decode(part['data']));print(json.dumps(dict(part=part['i'])))''')
  result=t.remote(common+'spec='+repr(dict(parts=(len(raw)+4095)//4096,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))+'\n'+'''raw=b''.join((code/('part-%04d'%i)).read_bytes()for i in range(spec['parts']));assert len(raw)==spec['bytes']and hashlib.sha256(raw).hexdigest()==spec['sha256'];compile(raw.decode(),'selected_old5','exec');once(code/'audit.py',raw);pub=dict(source_commit=V['commit'],code=proof(code/'audit.py'),fixed_old5_questions=9344,old5_trajectories=46720,CPU_only=True);once(code/'source-publication.json',(json.dumps(pub,sort_keys=True)+'\\n').encode());print(json.dumps(dict(status='FIXED9344_OLD5_CODE_STAGED',publication=proof(code/'source-publication.json'))))''')
 elif a.action=='run':
  step=r'''import pathlib,json,os
b=pathlib.Path(X['tail']);c=json.loads((b/'workers/worker-0/controller.json').read_bytes());p=pathlib.Path('/proc')/str(c['pid']);s=(p/'stat').read_text().rsplit(')',1)[1].split();assert int(s[19])==int(c['birth_ticks'])and c['Slurm_job_id']=='745518'and b'hpc3_tail_g5.py'in(p/'cmdline').read_bytes();conf=json.loads((b/'config.json').read_bytes());root=conf['qualified_runtime_root'];env=dict(os.environ,CUDA_VISIBLE_DEVICES='',QWEN7B_CODE_ROOT=root,OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1');os.execvpe('bash',['bash',root+'/reproduction/operations/qwen7b_warmup/runtime.sh',X['audit'],'--audit-fixed-old5'],env)
'''
  result=t.remote(common+'step='+repr(step)+'\n'+'''launch=(out/'launch.lock').open('a+');fcntl.flock(launch,fcntl.LOCK_EX|fcntl.LOCK_NB)
if(out/'complete.json').exists():print(json.dumps(dict(status='ALREADY_COMPLETE_NO_REAUDIT',complete=proof(out/'complete.json'))));sys.exit(0)
guard=(out/'audit.lock').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(guard,fcntl.LOCK_UN);guard.close();pub=read(code/'source-publication.json');assert pub['source_commit']==V['commit']and proof(code/'audit.py')==pub['code'];assert(pathlib.Path(V['tail'])/'worker-0.job.id').read_text().strip()=='745518';state=subprocess.check_output(['squeue','-h','-j','745518','-o','%T'],universal_newlines=True).strip();assert state=='RUNNING','Existing owned allocation required'
script='X='+repr(dict(tail=V['tail'],audit=str(code/'audit.py')))+'\\n'+step;once(code/'launch-intent.json',(json.dumps(dict(job_id='745518',CPU=2,GPU=0,source_publication=proof(code/'source-publication.json'),epoch=time.time()),sort_keys=True)+'\\n').encode());log=code/'step.log'
with log.open('ab')as stream:p=subprocess.run(['srun','--jobid=745518','--overlap','--nodes=1','--ntasks=1','--cpus-per-task=2','--gres=gpu:0','--mem=3G','env','CUDA_VISIBLE_DEVICES=','python3','-c',script],stdout=stream,stderr=stream,timeout=1100)
if p.returncode:raise RuntimeError('Selected old5 audit exited '+str(p.returncode)+' log='+str(log))
assert(out/'complete.json').exists();print(json.dumps(dict(complete=proof(out/'complete.json'),summary=read(out/'complete.json'),log=proof(log))))''',timeout=1120)
 else:
  result=t.remote(common+'''value=dict(epoch=time.time())
for name in('controller','progress','complete'):
 p=out/(name+'.json')
 if p.exists():value[name]=dict(ref=proof(p),value=read(p))
print(json.dumps(value))''')
 REPORT.mkdir(parents=True,exist_ok=True);(REPORT/('actual-'+a.action+'.json')).write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,sort_keys=True))
if __name__=='__main__':main()
