import subprocess,json,time
v={'epoch':time.time()}
for key,args in [('whoami',['id','-un']),('nodes',['sinfo','-N','-h','-o','%N|%P|%G|%t']),('partitions',['scontrol','show','partition']),('user_assoc',['sacctmgr','-nP','show','user','root','withassoc','format=User,DefaultAccount,Account,Partition,QOS']),('memory',['cat','/proc/meminfo'])]:
 try:
  r=subprocess.run(args,text=True,capture_output=True,timeout=15);v[key]={'rc':r.returncode,'stdout':r.stdout if key!='memory'else'\n'.join(x for x in r.stdout.splitlines()if x.startswith(('MemTotal:','MemAvailable:'))),'stderr':r.stderr[-500:]}
 except Exception as e:v[key]=str(e)
print(json.dumps(v))
