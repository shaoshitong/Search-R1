import pathlib,subprocess,json,time,re
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/online-packed-full169028-M3to10-P20-20261022-v1')
def cmd(args):
 r=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=20)
 if r.returncode:raise RuntimeError(r.stderr)
 return r.stdout.strip()
part=cmd(['scontrol','show','partition','acd_ue','-o']);assert'PartitionName=acd_ue'in part and'State=UP'in part
results=[]
for loss,expected in [('mse','749568'),('mae','749569'),('z_mse_anchor','749570')]:
 base=b/loss;r=json.loads((base/'root-submission.json').read_bytes());assert r['job_id']==expected and r['publication_commit']=='7ac274ac8ff5dea0a12dfd44cfb03fe57491458f'
 info=cmd(['scontrol','show','job','-o',expected]);assert'JobName=q7b-full-var-'+loss+' 'in info and'UserId=mou176'in info and'Account=mou176'in info and'QOS=normal'in info and'Command='+str(base/'training.sbatch')in info
 before=cmd(['squeue','--start','-h','-j',expected,'-o','%i|%P|%T|%S|%R'])
 if'JobState=PENDING'in info and'Partition=acd_u 'in info:
  cmd(['scontrol','update','JobId='+expected,'Partition=acd_ue']);action='EXISTING_PENDING_JOB_RAISED_ONE_TIER_TO_ACD_UE'
 else:action='NO_CHANGE_STATE_OR_PARTITION_ALREADY_CHANGED'
 after=cmd(['scontrol','show','job','-o',expected]);forecast=cmd(['squeue','--start','-h','-j',expected,'-o','%i|%P|%T|%S|%R'])
 results.append(dict(loss=loss,job_id=expected,action=action,before=before,after_queue=forecast,identity_after=after))
out=dict(time=time.time(),medium_partition=part,existing_handles_only=True,QOS_unchanged='normal',no_emergency=True,new_submission=False,results=results)
p=b/'root-medium-priority.json';p.write_text(json.dumps(out)+'\n');print(json.dumps(out))
