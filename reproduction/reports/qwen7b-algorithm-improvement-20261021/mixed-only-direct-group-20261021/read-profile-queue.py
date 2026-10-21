import json,subprocess
out={}
for name,cmd in [('profile_queue',['squeue','--start','-h','-j','746155','-o','%i|%T|%S|%R']),('debug_nodes',['sinfo','-h','-p','debug','-N','-o','%N|%t|%C|%m']),('qos',['sacctmgr','-nP','show','assoc','where','user=mou176','format=Partition,QOS'])]:
 r=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=8);out[name]=dict(returncode=r.returncode,stdout=r.stdout[:4000],stderr=r.stderr[:500])
print(json.dumps(out))
