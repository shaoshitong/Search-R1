"""Authorized scheduler-only stepwise escalation of two >30min pending jobs."""
import pathlib,subprocess,json,time,datetime
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/replay-matched9088-bank-v2-20261021')
def command(args):
 r=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=20)
 assert r.returncode==0,r.stderr;return r.stdout.strip()
partition=command(['scontrol','show','partition','acd_ue','-o']);assert 'PartitionName=acd_ue ' in partition
rows=[]
for i,job in [(2,'743089'),(3,'743090')]:
 root=b/'prepared'/('shard-%02d'%i);assert (root/'job.id').read_text().strip()==job
 before=command(['scontrol','show','job',job,'-o'])
 assert 'JobName=q7b-replay-bank-'+str(i)+' ' in before and 'Command='+str(root/'replay.sbatch') in before and 'UserId=mou176(21930)' in before
 if 'JobState=PENDING ' not in before:after=before;action='ALREADY_STARTED_NOT_CHANGED'
 elif 'Partition=acd_ue ' in before:after=before;action='ALREADY_MIDDLE_PARTITION_NOT_CHANGED'
 else:
  assert 'Partition=acd_u ' in before and not (root/'capture').exists()
  starts=[x.split('=',1)[1] for x in before.split() if x.startswith('StartTime=')];assert len(starts)==1
  eta=datetime.datetime.strptime(starts[0],'%Y-%m-%dT%H:%M:%S').timestamp()-time.time();assert eta>1800,eta
  command(['scontrol','update','JobId='+job,'Partition=acd_ue','TimeLimit=01:00:00'])
  after=command(['scontrol','show','job',job,'-o']);assert 'Partition=acd_ue ' in after and 'TimeLimit=01:00:00' in after
  action='SAME_PENDING_ID_NORMAL_TO_MIDDLE_ACTUAL_ETA_OVER30MIN'
 rows.append(dict(shard=i,job=job,action=action,before=before,after=after))
result=dict(status='R2_PENDING_SHARDS23_SCHEDULER_ONLY_ESCALATION',time=time.time(),reason='Two actual normal queue estimates2026-10-21T02:47:02 at01:54(~53min); parent explicitly authorized stepwise middle when>30min. Shard0 running and shard1~25min remain unchanged.',records=rows,no_resubmission=True,emergency_used=False,source_or_fixture_changed=False,prepared_commit='65f08f72b4e61cf72e3c43a202bda0508830a160',controller_commit='92bf7117404fb2b6c4dd23794a49773b777b67ed')
(b/'queue-escalation-shards23.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
