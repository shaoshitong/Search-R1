import json,re,subprocess,time
value=dict(epoch=time.time(),read_only=True,production_modified=False,pending_job_modified=False)
allowed={'JobId','UserId','JobState','Reason','Partition','NumNodes','NumCPUs','ReqTRES','AllocTRES','MinMemoryNode','MinMemoryCPU','MinCPUsNode','StartTime','EligibleTime','LastSchedEval','SchedNodeList','NodeList','ExcNodeList','TimeLimit','Priority','QOS','NodeName','CPUAlloc','CPUTot','RealMemory','AllocMem','FreeMem','State','Gres','CfgTRES'}
for name,cmd in {'job':['scontrol','show','job','-o','749812'],'excluded_node':['scontrol','show','node','-o','ACD1-13']}.items():
 r=subprocess.run(cmd,capture_output=True,text=True,timeout=20)
 fields={k:v for k,v in re.findall(r'(?:^|\s)([A-Za-z][A-Za-z0-9_/:]*)=(\S*)',r.stdout) if k in allowed}
 value[name]={'command':cmd,'returncode':r.returncode,'fields':fields,'stderr':r.stderr}
print(json.dumps(value),flush=True)
