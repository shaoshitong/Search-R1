import json,socket,subprocess,time
v=dict(epoch=time.time(),host=socket.gethostname(),tcp=[])
for addr in ['10.33.4.14','10.32.0.11']:
 t=time.monotonic()
 try:
  with socket.create_connection((addr,22),timeout=3)as s:v['tcp'].append(dict(address=addr,result='CONNECTED',banner=s.recv(160).decode(errors='replace'),seconds=time.monotonic()-t))
 except OSError as e:v['tcp'].append(dict(address=addr,error=str(e),seconds=time.monotonic()-t))
for k,cmd in [('slurm_node',['scontrol','show','node','h100-11']),('jobs',['squeue','-h','-w','h100-11','-o','%i|%j|%T|%S|%e'])]:
 r=subprocess.run(cmd,text=True,capture_output=True,timeout=10);v[k]=dict(rc=r.returncode,stdout=r.stdout,stderr=r.stderr)
print(json.dumps(v))
