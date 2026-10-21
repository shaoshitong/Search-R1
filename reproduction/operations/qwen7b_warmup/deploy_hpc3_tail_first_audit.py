"""Stage one accepted CPU audit; run only inside each own allocation, GPU0."""
import argparse,base64,hashlib,importlib.util,json,subprocess,shlex
from pathlib import Path
R=Path(__file__).resolve().parents[3];PKG=R/'reproduction/operations/qwen7b_warmup'
REPORT=R/'reproduction/reports/qwen7b-algorithm-improvement-20261021/hpc3-tail-newseed-preparation'
A='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021'
OUT=A+'/hpc3-tail-firstbatch-audit-v1'
TAIL=A+'/hpc3-tail-newseedG5-20261021-v1'
SOURCE='8599d1fdc7ff8538be3478fb236b05446d7b8598'

def main():
 p=argparse.ArgumentParser();p.add_argument('--action',choices=['stage','run'],required=True);p.add_argument('--worker',type=int,choices=(0,1));a=p.parse_args()
 spec=importlib.util.spec_from_file_location('transport',PKG/'deploy_hpc3_tail_g5.py');t=importlib.util.module_from_spec(spec);spec.loader.exec_module(t)
 path=PKG/'audit_hpc3_tail_first_batch.py';rel=str(path.relative_to(R));raw=path.read_bytes();assert subprocess.check_output(['git','show',SOURCE+':'+rel],cwd=R)==raw
 commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=R).decode().strip();head=subprocess.check_output(['git','ls-remote','origin','refs/heads/codex/qwen7b-probe-warmup-20261019'],cwd=R).decode().split()[0];assert commit==head
 assert subprocess.check_output(['git','show',commit+':'+str(Path(__file__).relative_to(R))],cwd=R)==Path(__file__).read_bytes()
 code=OUT+'/code-'+SOURCE[:12]
 common='v='+repr(dict(out=OUT,tail=TAIL,code=code,source=SOURCE,helper=commit))+'\n'+'''import pathlib,json,hashlib,os,subprocess,time,sys
out=pathlib.Path(v['out']);code=pathlib.Path(v['code']);out.mkdir(exist_ok=True);code.mkdir(exist_ok=True)
def read(p):return json.loads(pathlib.Path(p).read_bytes())
def proof(p):
 p=pathlib.Path(p);raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def once(p,raw):
 if p.exists():assert p.read_bytes()==raw,'Independent audit source differs'
 else:
  with p.open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
'''
 if a.action=='stage':
  for i in range(0,len(raw),4096):
   part=raw[i:i+4096];t.remote(common+'part='+repr(dict(index=i//4096,data=base64.b64encode(part).decode(),sha=hashlib.sha256(part).hexdigest()))+'\n'+'''import base64
raw=base64.b64decode(part['data']);assert hashlib.sha256(raw).hexdigest()==part['sha'];once(code/('part-%04d'%part['index']),raw);print(json.dumps(dict(part=part['index'],bytes=len(raw))))''')
  value=t.remote(common+'spec='+repr(dict(parts=(len(raw)+4095)//4096,bytes=len(raw),sha=hashlib.sha256(raw).hexdigest()))+'\n'+'''raw=b''.join((code/('part-%04d'%i)).read_bytes()for i in range(spec['parts']));assert len(raw)==spec['bytes']and hashlib.sha256(raw).hexdigest()==spec['sha'];compile(raw.decode(),'audit','exec');once(code/'audit.py',raw);pub=dict(source_commit=v['source'],code=proof(code/'audit.py'),role='actual_first_own_seal_only_CPU_tokenizer_no_model');once(out/'source-publication.json',(json.dumps(pub,sort_keys=True)+'\\n').encode());print(json.dumps(dict(status='INDEPENDENT_FIRST_OWN_SEAL_AUDIT_CODE_STAGED',publication=proof(out/'source-publication.json'))))''')
 else:
  if a.worker is None:p.error('--worker required')
  probe=r'''import pathlib,json,os,subprocess,hashlib
b=pathlib.Path(V['tail']);d=b/'workers'/('worker-%d'%V['worker']);c=json.loads((d/'controller.json').read_bytes());p=pathlib.Path('/proc')/str(c['pid']);stat=(p/'stat').read_text().rsplit(')',1)[1].split();assert int(stat[19])==int(c['birth_ticks'])and c['Slurm_job_id']==V['job']
cmd=(p/'cmdline').read_bytes();assert b'hpc3_tail_g5.py'in cmd and b'368ad5d8df9e0238bdee8033964b56f0f4a6426e606f9514114fac0dcb05d835'in cmd
selected={x.split(b'=',1)[0].decode():x.split(b'=',1)[1].decode()for x in(p/'environ').read_bytes().split(b'\0')if x.startswith((b'CUDA_VISIBLE_DEVICES=',b'SLURM_JOB_GPUS='))};visible=selected['CUDA_VISIBLE_DEVICES'];assert len(visible.split(','))==2
hardware=[]
for info in pathlib.Path('/proc/driver/nvidia/gpus').glob('*/information'):
 try:
  fields=dict(line.split(':',1)for line in info.read_text().splitlines()if ':'in line);fields={k.strip():v.strip()for k,v in fields.items()};minor=fields.get('Device Minor')
  if minor in selected.get('SLURM_JOB_GPUS',visible).split(','):hardware.append({k:fields[k]for k in('Model','GPU UUID','Device Minor')if k in fields})
 except(OSError,ValueError):pass
r=json.loads((d/'retriever/ready.json').read_bytes());rp=pathlib.Path('/proc')/str(r['pid']);rs=(rp/'stat').read_text().rsplit(')',1)[1].split();assert int(rs[1])==c['pid'];assert r['run_id']=='36b784a3bf964e69a6130bcc41d68809'
receipt=dict(status='OWN_ACTUAL_CONTROLLER_BIRTH_JOB_ALLOCATION_BINDING_VERIFIED',worker=V['worker'],job_id=V['job'],controller_pid=c['pid'],controller_birth=int(stat[19]),controller_host=os.uname().nodename,retriever_pid=r['pid'],retriever_parent_pid=int(rs[1]),parent_GPU_binding=selected,read_only_proc_GPU_metadata=hardware,GPU_metadata_complete=len(hardware)==2,audit_step_GPU=0,nvidia_smi_not_required_in_GPU0_step=True)
q=pathlib.Path(V['out'])/('worker-%d-actual-hardware.json'%V['worker']);q.write_text(json.dumps(receipt,sort_keys=True)+'\n')
env=dict(os.environ,CUDA_VISIBLE_DEVICES='',QWEN7B_CODE_ROOT=V['oldroot'],OMP_NUM_THREADS='2',MKL_NUM_THREADS='2',OPENBLAS_NUM_THREADS='2',PYTHONDONTWRITEBYTECODE='1');os.execvpe('bash',['bash',V['runtime'],V['audit'],'--worker',str(V['worker'])],env)
'''
  value=t.remote(common+'worker='+str(a.worker)+'\nprobe='+repr(probe)+'\n'+'''pub=read(out/'source-publication.json');assert pub['source_commit']==v['source'];assert proof(pub['code']['path'])==pub['code'];b=pathlib.Path(v['tail']);job=(b/('worker-%d.job.id'%worker)).read_text().strip();assert job==('745518'if worker==0 else'745519')
state=subprocess.check_output(['squeue','-h','-j',job,'-o','%T'],universal_newlines=True).strip();assert state=='RUNNING','CPU audit requires its actual live owned allocation'
c=read(b/'config.json');oldroot=c['qualified_runtime_root'];runtime=oldroot+'/reproduction/operations/qwen7b_warmup/runtime.sh';x=dict(tail=v['tail'],out=v['out'],worker=worker,job=job,oldroot=oldroot,runtime=runtime,audit=pub['code']['path']);script='V='+repr(x)+'\\n'+probe
args=['srun','--jobid='+job,'--overlap','--nodes=1','--ntasks=1','--cpus-per-task=2','--gres=gpu:0','--mem=3G','env','CUDA_VISIBLE_DEVICES=','python3','-c',script]
r=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=65)
step_log=out/('worker-%d-step-%s.log'%(worker,v['helper'][:12]));step_log.write_text(r.stdout+r.stderr)
if r.returncode:raise RuntimeError('Own allocation CPU audit failure: '+r.stderr[-1400:]+r.stdout[-1200:])
rows=[json.loads(line)for line in r.stdout.splitlines()if line.startswith('{')];assert rows
print(json.dumps(dict(audit=rows[-1],hardware=read(out/('worker-%d-actual-hardware.json'%worker)),step_log=proof(step_log),step_CPU=2,step_GPU=0),sort_keys=True))''',timeout=85)
 REPORT.mkdir(exist_ok=True);(REPORT/('actual-first-audit-'+a.action+(''if a.worker is None else'-worker%d'%a.worker)+'.json')).write_text(json.dumps(value,indent=2)+'\n')
 if a.action=='run':
  audit=value['audit'];print(json.dumps(dict(worker=a.worker,status=audit['status'],questions=audit['questions'],trajectories=audit['trajectories'],strict_EM_recomputed=audit['strict_EM_recomputed'],strict_EM_mismatches=audit['strict_EM_mismatches'],real_G5_ownIDs_and_seeds_verified=audit['real_G5_ownIDs_and_seeds_verified'],Prefix20_causal_positions_verified=audit['Prefix20_causal_positions_verified'],seal=audit['seal'],actor_updates=audit['actor_updates'],hardware=value['hardware'],CUDA_initialized=audit['CUDA_initialized'],model_tensors_loaded=audit['model_tensors_loaded']),sort_keys=True))
 else:print(json.dumps(value,sort_keys=True))
if __name__=='__main__':main()
