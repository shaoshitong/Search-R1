import pathlib,json,subprocess,time,shutil,fcntl
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/online-packed-full169028-M3to10-P20-20261022-v1')
def cmd(args):
 r=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=20)
 if r.returncode:raise RuntimeError(' '.join(args)+': '+r.stderr)
 return r.stdout.strip()
node=cmd(['scontrol','show','node','ACD1-13','-o']);result={'time':time.time(),'original_failed_node_identity':node,'same_job_requeue':True,'parameter_updates_before_failure':0,'source_code_changed':False,'results':[]}
for loss,job in [('mae','749569'),('z_mse_anchor','749570')]:
 base=b/loss;guard=(base/'root-cuda-recovery.lock').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
 receipt=json.loads((base/'root-submission.json').read_bytes());assert receipt['job_id']==job and receipt['publication_commit']=='7ac274ac8ff5dea0a12dfd44cfb03fe57491458f'
 recovery=base/'root-cuda-initialization-recovery.json'
 if recovery.exists():
  previous=json.loads(recovery.read_bytes());result['results'].append(dict(job_id=job,loss=loss,status='EXISTING_RECOVERY_NO_NEW_REQUEUE',current=cmd(['scontrol','show','job','-o',job])));continue
 acct=cmd(['sacct','-X','-n','-P','-j',job,'--format=JobIDRaw,State,ExitCode,NodeList']);assert job+'|FAILED|1:0|ACD1-13'in acct
 identity=cmd(['scontrol','show','job','-o',job]);assert'JobName=q7b-full-var-'+loss+' 'in identity and'UserId=mou176'in identity and'JobState=FAILED'in identity and'Command='+str(base/'training.sbatch')in identity
 assert not any((base/n).exists()for n in ['controller.json','latest.json','complete.json','progress-rank-0.json','progress-rank-1.json'])
 failures=list(base.glob('failure-rank-*.json'));assert len(failures)==2
 for f in failures:
  text=json.loads(f.read_bytes())['traceback'];assert'torch.cuda.set_device(device)'in text and'CUDA unknown error'in text
 archive=base/'attempts'/('cuda-init-failed-'+job);archive.mkdir(parents=True,exist_ok=False)
 for f in failures:shutil.copy2(f,archive/f.name)
 for n in ['root-submission.json','ready.json','training.sbatch','train-'+job+'.log']:
  p=base/n
  if p.exists():shutil.copy2(p,archive/n)
 (archive/'accounting.json').write_text(json.dumps(dict(accounting=acct,slurm_identity=identity,source_unchanged=True,updates=0))+'\n')
 intent=dict(job_id=job,loss=loss,stage='REQUEUE_HOLD_INTENT_PRESERVE_EXACT_JOB_NO_NEW_SUBMIT',time=time.time(),excluded_node='ACD1-13',archive=str(archive));(base/'root-cuda-recovery-intent.json').write_text(json.dumps(intent)+'\n')
 cmd(['scontrol','requeuehold',job]);held=cmd(['scontrol','show','job','-o',job]);assert'JobState=PENDING'in held and'Priority=0'in held
 cmd(['scontrol','update','JobId='+job,'ExcNodeList=ACD1-13','Partition=acd_ue']);after=cmd(['scontrol','show','job','-o',job]);assert'ExcNodeList=ACD1-13'in after and'Partition=acd_ue'in after and'QOS=normal'in after
 for f in failures:f.rename(archive/('original-'+f.name))
 out=dict(job_id=job,loss=loss,time=time.time(),stage='SAME_JOB_HELD_BAD_NODE_EXCLUDED_READY_TO_RELEASE',source_unchanged=True,parameter_updates_discarded=0,excluded_node='ACD1-13',archive=str(archive),identity_after=after);recovery.write_text(json.dumps(out)+'\n')
 cmd(['scontrol','release',job]);out['queue_after']=cmd(['squeue','--start','-h','-j',job,'-o','%i|%P|%T|%S|%R']);out['stage']='SAME_JOB_REQUEUED_BAD_NODE_EXCLUDED_RELEASED';recovery.write_text(json.dumps(out)+'\n');result['results'].append(out)
(b/'root-cuda-recovery-summary.json').write_text(json.dumps(result)+'\n');print(json.dumps(result))
