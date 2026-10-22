import json,subprocess,time,re
r={'epoch':time.time(),'mutation':False}
for name,cmd in {'nodes':['scontrol','show','nodes','-o'], 'old_usage':['sacct','-n','-j','749598','--format=JobID,State,AllocCPUS,Elapsed,TotalCPU,MaxRSS,AveRSS','-P'], 'partitions':['sinfo','-h','-o','%P|%a|%l|%D|%G']}.items():
 p=subprocess.run(cmd,capture_output=True,text=True,timeout=15)
 if name=='nodes':
  rows=[]
  for line in p.stdout.splitlines():
   d=dict(re.findall(r'(\S+?)=(\S*)',line))
   if 'emergency_acd' not in d.get('Partitions','').split(','):continue
   rows.append({k:d.get(k) for k in ['NodeName','State','Gres','CPUTot','CPUAlloc','RealMemory','AllocMem','FreeMem','CfgTRES','AllocTRES','Partitions']})
  r[name]={'returncode':p.returncode,'rows':rows,'stderr':p.stderr}
 else:r[name]={'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
print(json.dumps(r),flush=True)
