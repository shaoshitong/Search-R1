import subprocess,json,time,socket
v=dict(epoch=time.time(),hostname=socket.gethostname())
for k,args in [('gpus',['nvidia-smi','--query-gpu=index,uuid,memory.total,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits']),('processes',['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv,noheader,nounits']),('slurm',['squeue','-h','-o','%i|%j|%T|%N|%b'])]:
 try:r=subprocess.run(args,text=True,capture_output=True,timeout=15);v[k]=dict(returncode=r.returncode,stdout=r.stdout[-14000:],stderr=r.stderr[-500:])
 except Exception as e:v[k]=str(e)
print(json.dumps(v))
