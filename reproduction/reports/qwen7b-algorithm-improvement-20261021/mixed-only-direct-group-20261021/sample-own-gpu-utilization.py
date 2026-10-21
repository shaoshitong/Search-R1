import json,pathlib,subprocess,time
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/joint-mixed-direct-official-169028-v1')
c=json.loads((b/'controller.json').read_text());assert str(c['job_id'])=='746083'
pid=c['pid']
source='''import subprocess,json,time,pathlib
pid=__PID__;root=pathlib.Path('/proc')/str(pid)
args=(root/'cmdline').read_bytes().decode().split(chr(0));assert'reproduction.operations.qwen7b_warmup.joint_mixed_direct_pretrain'in args
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],universal_newlines=True)
uuid=[line.split(',')[1].strip()for line in apps.splitlines()if line.split(',')[0].strip()==str(pid)];assert len(uuid)==1
rows=[]
for i in range(12):
 rows.append(subprocess.check_output(['nvidia-smi','-i',uuid[0],'--query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw','--format=csv,noheader,nounits'],universal_newlines=True).strip())
 if i<11:time.sleep(1)
print(json.dumps(dict(pid=pid,own_GPU_only=True,samples=rows)))
'''.replace('__PID__',str(pid))
r=subprocess.run(['srun','--jobid=746083','--overlap','--gres=gpu:1','--ntasks=1','--cpus-per-task=1','--mem=128M','python3','-c',source],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=23)
print(r.stdout)
if r.returncode:raise RuntimeError(r.stderr[-1500:])
