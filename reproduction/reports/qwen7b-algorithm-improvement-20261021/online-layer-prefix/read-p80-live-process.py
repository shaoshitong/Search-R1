import json,subprocess,time
source=r'''
import os,pathlib,json,subprocess,time
procs=[]
for p in pathlib.Path('/proc').glob('[0-9]*'):
 try:
  argv=(p/'cmdline').read_bytes().split(b'\0');cmd=b' '.join(argv)
  if b'-c'in argv or b'reproduction.operations.qwen7b_warmup.online_mixed_prefix_explore'not in argv or b'--prefix'not in argv or argv[argv.index(b'--prefix')+1]!=b'80':continue
  stat=(p/'stat').read_text().split();status=(p/'status').read_text();io=(p/'io').read_text()
  procs.append(dict(pid=int(p.name),state=stat[2],utime=stat[13],stime=stat[14],VmRSS=next((x for x in status.splitlines()if x.startswith('VmRSS:')),None),io=io))
 except (OSError,ValueError,IndexError):continue
pids={x['pid']for x in procs}
r=subprocess.run(['nvidia-smi','--query-compute-apps=pid,used_memory,gpu_uuid','--format=csv,noheader,nounits'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=5)
apps=[s for s in r.stdout.splitlines()if s.strip().split(',')[0].isdigit()and int(s.split(',')[0])in pids];uuids={s.split(',')[-1].strip()for s in apps}
r=subprocess.run(['nvidia-smi','--query-gpu=uuid,utilization.gpu,memory.used,memory.total','--format=csv,noheader,nounits'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=5)
print(json.dumps(dict(time=time.time(),job='749079',owned_python=procs,owned_GPU_apps=apps,owned_GPU_metrics=[s for s in r.stdout.splitlines()if s.split(',')[0].strip()in uuids])))
'''
r=subprocess.run(['srun','--jobid=749079','--overlap','--nodes=1','--ntasks=1','--cpus-per-task=1','python3','-c',source],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=20)
assert r.returncode==0,r.stderr[-1000:]
print(r.stdout.strip())
