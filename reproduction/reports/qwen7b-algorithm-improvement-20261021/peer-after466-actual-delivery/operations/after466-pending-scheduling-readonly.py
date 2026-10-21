import subprocess,json,pathlib,time,hashlib
A=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021');b=A/'peer-joint124194-audited262-L26-v1';out=dict(epoch=time.time(),mutation=False,jobs={})
def cmd(args):
 p=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=12);return dict(returncode=p.returncode,stdout=p.stdout.strip(),stderr=p.stderr.strip())
for i,j in enumerate(('745395','745396')):
 job=cmd(['scontrol','show','job','-o',j]);assert job['returncode']==0;v={x.split('=',1)[0]:x.split('=',1)[1]for x in job['stdout'].split()if'='in x};script=b/('pair-%02d'%i)/'capture.sbatch';out['jobs'][j]=dict(fields=v,sbatch=dict(path=str(script),bytes=script.stat().st_size,sha256=hashlib.sha256(script.read_bytes()).hexdigest()))
out['own_associations']=cmd(['sacctmgr','-n','-P','show','assoc','where','user=mou176','format=User,Account,Partition,QOS,DefaultQOS'])
out['normal_medium_qos']=cmd(['sacctmgr','-n','-P','show','qos','where','name=normal,medium','format=Name,Priority,MaxTRESPU,MaxJobsPU,MaxSubmitPU'])
out['medium_partition']=cmd(['scontrol','show','partition','-o','acd_ue'])
out['start_estimates']=cmd(['squeue','--start','-h','-j','745395,745396','-o','%i|%T|%S|%R'])
print(json.dumps(out))
