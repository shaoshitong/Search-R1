import json,re,subprocess,time

value=dict(epoch=time.time(),read_only=True,production_modified=False,
           pending_job_modified=False,new_jobs_submitted=False)
cmd=['sacct','-j','749739,749089','-n','-P','-o',
     'JobIDRaw,JobName,State,ExitCode,MaxRSS,ReqMem,AllocTRES,Elapsed,NodeList']
p=subprocess.run(cmd,capture_output=True,text=True,timeout=20)
fields=['JobIDRaw','JobName','State','ExitCode','MaxRSS','ReqMem','AllocTRES','Elapsed','NodeList']
rows=[]
for line in p.stdout.splitlines():
 pieces=line.split('|')
 if len(pieces)>=len(fields):rows.append(dict(zip(fields,pieces)))
value['completed_diagnostics']=dict(command=cmd,returncode=p.returncode,
    rows=rows,stderr=p.stderr)
cmd=['scontrol','show','job','-o','749812']
p=subprocess.run(cmd,capture_output=True,text=True,timeout=20)
allowed={'JobId','JobState','Reason','Partition','NumNodes','NumCPUs',
         'ReqTRES','AllocTRES','MinMemoryNode','MinMemoryCPU','MinCPUsNode',
         'TresPerNode','TRESPerNode','TresPerTask','TRESPerTask',
         'Command','WorkDir','ReqNodeList','ExcNodeList','NodeList'}
selected={key:item for key,item in re.findall(r'(?:^|\s)([A-Za-z][A-Za-z0-9_/:]*)=(\S*)',p.stdout)
          if key in allowed}
value['pending_diagnostic']=dict(command=cmd,returncode=p.returncode,
    resource_fields=selected,stderr=p.stderr)
value['scope']='Completed-job accounting including batch/steps; pending-job resource/path allowlist only. No environment or credentials queried or returned.'
print(json.dumps(value,sort_keys=True),flush=True)
