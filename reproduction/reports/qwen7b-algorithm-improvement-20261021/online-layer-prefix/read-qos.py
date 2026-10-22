import subprocess,json
x={}
for key,args in [('allowed',['sacctmgr','-n','-P','show','assoc','where','user=mou176','format=Account,User,QOS']),('priorities',['sacctmgr','-n','-P','show','qos','format=Name,Priority'])]:
 r=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=15);x[key]=dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr[-600:])
r=subprocess.run(['sinfo','-h','-o','%P|%a|%l|%D|%C|%G'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=15);x['partitions']=r.stdout
print(json.dumps(x))
