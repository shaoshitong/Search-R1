"""Publish one finite CPU-only step beside an identity-bound active allocation."""
import argparse,base64,hashlib,importlib.util,json,pathlib,shlex,subprocess,zlib
P='reproduction/operations/qwen7b_warmup/'
def main():
 p=argparse.ArgumentParser();p.add_argument('--control-root',type=pathlib.Path,required=True);p.add_argument('--commit',required=True);p.add_argument('--action',choices=('start','status'),required=True);a=p.parse_args();repo=pathlib.Path(__file__).resolve().parents[3];branch='codex/main745501-readonly-prefetch-20261021';remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+branch],cwd=repo,text=True).split()[0];assert remote==a.commit
 names={'worker.py':P+'prefetch_main_feature_payloads.py','config.json':'reproduction/reports/qwen7b-algorithm-improvement-20261021/main745501-readonly-prefetch/config.json'};files={}
 for name,src in names.items():
  v=subprocess.check_output(['git','show',a.commit+':'+src],cwd=repo);assert v==(repo/src).read_bytes();files[name]=dict(data=base64.b64encode(zlib.compress(v)).decode(),bytes=len(v),sha256=hashlib.sha256(v).hexdigest())
 source='inputs='+repr(dict(files=files,commit=a.commit,action=a.action))+'\n'+r'''
import base64,zlib,pathlib,json,hashlib,os,fcntl,subprocess,time
c=json.loads(zlib.decompress(base64.b64decode(inputs['files']['config.json']['data'])));b=pathlib.Path(c['target']);main=pathlib.Path(c['main_base']);b.mkdir(mode=0o700,exist_ok=True)
def read(p):return json.loads(p.read_text())
def fp(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
def atomic(p,v):
 q=p.with_name(p.name+'.%d.tmp'%os.getpid());q.write_text(json.dumps(v,sort_keys=True)+'\n');q.replace(p)
def ticks(pid):
 try:return pathlib.Path('/proc/%d/stat'%pid).read_text().rsplit(')',1)[1].split()[19]
 except FileNotFoundError:return None
with(b/'operator.lock').open('a')as g:
 fcntl.flock(g,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if inputs['action']=='start':
  job=subprocess.check_output(['scontrol','show','job','-o',str(c['job_id'])],universal_newlines=True);v={x.split('=',1)[0]:x.split('=',1)[1]for x in job.split()if'='in x};assert v['UserId'].startswith('mou176(')and v['Account']=='mou176'and v['JobState']=='RUNNING'and v['NodeList']==c['node']and v['Command']==str(main/'training.sbatch');assert fp(main/'config.json')['sha256']==c['main_config_sha256']and read(main/'source-publication.json')['commit']==c['main_commit']
  for name,item in inputs['files'].items():
   raw=zlib.decompress(base64.b64decode(item['data']));assert len(raw)==item['bytes']and hashlib.sha256(raw).hexdigest()==item['sha256'];p=b/name
   if p.exists():assert p.read_bytes()==raw
   else:p.write_bytes(raw);p.chmod(0o444)
  pub=dict(commit=inputs['commit'],files={n:fp(b/n)for n in inputs['files']},main_config=fp(main/'config.json'),main_publication=fp(main/'source-publication.json'))
  if(b/'source-publication.json').exists():assert read(b/'source-publication.json')==pub
  else:atomic(b/'source-publication.json',pub)
  if(b/'launch.json').exists():v=read(b/'launch.json');answer=dict(status='EXISTING_PREFETCH_LAUNCH_PRESERVED',launch=v,same_birth=ticks(v['pid'])==v['start_ticks'])
  else:
   assert not(b/'launch-intent.json').exists(),'Unknown previous step outcome; never duplicate';atomic(b/'launch-intent.json',dict(time=time.time(),source=fp(b/'source-publication.json'),job_identity=v))
   cmd=['srun','--jobid='+str(c['job_id']),'--overlap','--exact','--ntasks=1','--cpus-per-task=2','--mem=2G','--gres=gpu:0','--time=00:11:00','env','CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=2','MKL_NUM_THREADS=2','OPENBLAS_NUM_THREADS=2','python3','-u',str(b/'worker.py'),'--config',str(b/'config.json')]
   with(b/'step.log').open('ab')as log:r=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
   answer=dict(status='FINITE_READ_ONLY_PREFETCH_CPU_STEP_STARTED_NOT_PROGRESS_VERIFIED',pid=r.pid,start_ticks=ticks(r.pid),job_id=c['job_id'],time=time.time());atomic(b/'launch.json',answer)
 else:
  answer=dict(status='ACTUAL_READ_ONLY_PREFETCH_OBSERVATION',time=time.time())
  for name in ('launch.json','process.json','progress.json','complete.json','failure.json'):
   if(b/name).exists():answer[name]=read(b/name)
  if(b/'launch.json').exists():v=read(b/'launch.json');answer['same_launch_birth']=ticks(v['pid'])==v['start_ticks']
  if(b/'step.log').exists():answer['log_tail']=(b/'step.log').read_text(errors='replace')[-1400:]
 print(json.dumps(answer))
'''
 command='python3 -c '+shlex.quote(source);assert len(command.encode())<16000
 spec=importlib.util.spec_from_file_location('prefetch_direct',a.control_root/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl);ip=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0'],text=True).strip();r=ctl.run(command,account='mou176',source_address=ip,timeout=35);rows=[json.loads(x)for x in r.stdout.decode(errors='replace').splitlines()if x.startswith('{')];assert r.returncode==0 and rows,r.stderr.decode(errors='replace')[-1800:]+r.stdout.decode(errors='replace')[-1800:];v=rows[-1];out=a.control_root/'outputs/qwen7b-warmup-20261019/reports'/('main745501-prefetch-'+a.action+'.json');out.write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v),flush=True)
if __name__=='__main__':main()
