import json,pathlib,subprocess
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/joint-mixed-direct-official-169028-b128-v1');c=json.loads((b/'controller.json').read_bytes());job=json.loads((b/'submission-result.json').read_bytes())['job_id'];assert c['job_id']==job
source='''import json,pathlib,subprocess,time
pid=__PID__;cmd=(pathlib.Path('/proc')/str(pid)/'cmdline').read_bytes().decode().split(chr(0));assert'reproduction.operations.qwen7b_warmup.joint_mixed_direct_b128'in cmd
apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,gpu_uuid','--format=csv,noheader,nounits'],universal_newlines=True);uuids=[l.split(',')[1].strip()for l in apps.splitlines()if l.split(',')[0].strip()==str(pid)];assert len(uuids)==1
rows=[]
for i in range(8):
 rows.append(subprocess.check_output(['nvidia-smi','-i',uuids[0],'--query-gpu=timestamp,memory.used,memory.total,utilization.gpu,power.draw','--format=csv,noheader,nounits'],universal_newlines=True).strip())
 if i<7:time.sleep(1)
print(json.dumps(dict(pid=pid,own_GPU_only=True,samples=rows)))
'''.replace('__PID__',str(c['pid']))
r=subprocess.run(['srun','--overlap','--jobid='+job,'--gres=gpu:1','--ntasks=1','--cpus-per-task=1','--mem=128M','python3','-c',source],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=20);print(r.stdout)
if r.returncode:raise RuntimeError(r.stderr[-1500:])
