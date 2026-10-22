import pathlib,json,subprocess,time,shutil,fcntl,hashlib
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/online-packed-full169028-M3to10-P20-20261022-v1')
def cmd(args):
 r=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=20)
 if r.returncode:raise RuntimeError(' '.join(args)+': '+r.stderr)
 return r.stdout.strip()
def proof(p):
 p=pathlib.Path(p);raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
results=[]
for loss,oldjob in [('mae','749569'),('z_mse_anchor','749570')]:
 base=b/loss;guard=(base/'root-cuda-recovery.lock').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);replacement=base/'root-cuda-replacement.json'
 if replacement.exists():results.append(json.loads(replacement.read_bytes()));continue
 prior=json.loads((base/'root-submission.json').read_bytes());assert prior['job_id']==oldjob and prior['publication_commit']=='7ac274ac8ff5dea0a12dfd44cfb03fe57491458f'
 acct=cmd(['sacct','-X','-n','-P','-j',oldjob,'--format=JobIDRaw,State,ExitCode,NodeList']);assert oldjob+'|FAILED|1:0|ACD1-13'in acct
 assert not any((base/n).exists()for n in ['controller.json','latest.json','complete.json','progress-rank-0.json','progress-rank-1.json'])
 name='q7b-full-var-'+loss;assert not cmd(['squeue','-h','-u','mou176','-n',name,'-o','%i|%T'])
 entry=base/'training.sbatch';assert proof(entry)==prior['entry'];failures=list(base.glob('failure-rank-*.json'));assert len(failures)==2
 for f in failures:assert'CUDA unknown error'in json.loads(f.read_bytes())['traceback']
 archive=base/'attempts'/('cuda-init-failed-'+oldjob);archive.mkdir(parents=True,exist_ok=True)
 for f in failures:
  p=archive/f.name
  if p.exists():assert p.read_bytes()==f.read_bytes()
  else:shutil.copy2(f,p)
 for n in ['root-submission.json','ready.json','training.sbatch','train-'+oldjob+'.log']:
  p=base/n
  if p.exists()and not(archive/n).exists():shutil.copy2(p,archive/n)
 intent=base/'root-cuda-replacement-intent.json';assert not intent.exists(),'Unknown prior replacement submit; reconcile squeue/sacct before retry'
 intent.write_text(json.dumps(dict(time=time.time(),old_job=oldjob,entry=proof(entry),reason='CUDA_init_before_any_update_node_ACD1-13_scontrol_requeue_disabled',excluded_node='ACD1-13'))+'\n')
 answer=cmd(['sbatch','--hold','--parsable','--partition=acd_ue','--exclude=ACD1-13',str(entry)]);job=answer.split(';')[0];assert job.isdigit()
 identity=cmd(['scontrol','show','job','-o',job])
 for field in ['JobName='+name,'UserId=mou176','Account=mou176','QOS=normal','Partition=acd_ue','NumCPUs=12','MinMemoryNode=128G','TimeLimit=08:00:00','Dependency=(null)','ExcNodeList=ACD1-13','Command='+str(entry)]:assert field in identity,field
 assert'gres/gpu=2'in identity or'TresPerNode=gres/gpu:2'in identity
 for f in failures:f.rename(archive/('original-'+f.name))
 current=dict(prior,job_id=job,time=time.time(),replaces_failed_job=oldjob,reason='pretraining_CUDA_initialization_node_failure',excluded_node='ACD1-13',parameter_updates_discarded=0,slurm_identity_before_release=identity,state='REPLACEMENT_SUBMITTED_NOT_PROGRESS_VERIFIED')
 (archive/'original-root-submission.json').write_bytes((base/'root-submission.json').read_bytes());(base/'root-submission.json').write_text(json.dumps(current)+'\n');replacement.write_text(json.dumps(current)+'\n')
 cmd(['scontrol','release',job]);current['queue_after']=cmd(['squeue','--start','-h','-j',job,'-o','%i|%P|%T|%S|%R']);replacement.write_text(json.dumps(current)+'\n');results.append(current)
out=dict(time=time.time(),source_code_unchanged=True,no_optimizer_updates_lost=True,results=results);(b/'root-cuda-replacement-summary.json').write_text(json.dumps(out)+'\n');print(json.dumps(out))
