import socket,os,json,subprocess,time,pathlib
x={'time':time.time(),'hostname':socket.gethostname(),'uid':os.getuid()}
for name,cmd in [('uptime',['uptime']),('data_mount',['findmnt','-T','/data','-o','TARGET,SOURCE,FSTYPE,OPTIONS']),('slurm_ping',['scontrol','ping']),('data_df',['df','-PT','/data'])]:
 try:
  r=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=8);x[name]={'returncode':r.returncode,'stdout':r.stdout[:2500],'stderr':r.stderr[:1200]}
 except Exception as e:x[name]={'error':str(e)[:500]}
for path in ['/data','/data/user/jxu957','/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021']:
 try:
  s=os.stat(path);x[path]={'exists':True,'device':s.st_dev,'inode':s.st_ino,'uid':s.st_uid,'mode':s.st_mode}
 except Exception as e:x[path]={'error':str(e)}
print(json.dumps(x))
