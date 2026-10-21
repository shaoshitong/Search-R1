import subprocess,json,time
prefix='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021'
def cmd(args):
 r=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=12)
 assert r.returncode==0,r.stderr[-1000:]
 return r.stdout
rows=cmd(['squeue','-h','-u','mou176','-o','%i|%j|%T|%M|%R']).splitlines();own=[]
for row in rows:
 jid,name,state,elapsed,node=row.split('|',4)
 if not(name.startswith('q7b')or name.startswith('qwen7b')):continue
 raw=cmd(['scontrol','show','job','-o',jid])
 work=next((v.split('=',1)[1] for v in raw.split()if v.startswith('WorkDir=')),None)
 if work and work.startswith(prefix):
  own.append(dict(job=jid,name=name,state=state,elapsed=elapsed,node=node,WorkDir=work,Command=next((v.split('=',1)[1]for v in raw.split()if v.startswith('Command=')),None),TRES=next((v.split('=',1)[1]for v in raw.split()if v.startswith('AllocTRES=')),None)))
print(json.dumps(dict(epoch=time.time(),own_project_jobs=own,other_jobs_not_modified=True)))
