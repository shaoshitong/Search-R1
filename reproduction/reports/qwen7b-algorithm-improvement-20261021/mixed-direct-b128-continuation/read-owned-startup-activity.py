import json,pathlib,subprocess
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/joint-mixed-direct-official-169028-b128-v1');c=json.loads((b/'controller.json').read_bytes());job=c['job_id'];pid=c['pid']
source='''import pathlib,json,time
p=pathlib.Path('/proc')/str(__PID__);assert'reproduction.operations.qwen7b_warmup.joint_mixed_direct_b128'in(p/'cmdline').read_bytes().decode().split(chr(0))
a=[]
for i in range(2):
 st=(p/'stat').read_text().split();io={k:int(v)for k,v in(l.split(':')for l in(p/'io').read_text().splitlines())};status={k:v.strip()for k,v in(l.split(':',1)for l in(p/'status').read_text().splitlines())if k in('State','VmRSS','VmSize','Threads')};opened=[]
 for fd in(p/'fd').iterdir():
  try:
   target=str(fd.resolve())
   if target.endswith('index.jsonl'):opened.append(target)
  except OSError:pass
 a.append(dict(epoch=time.time(),CPU_ticks=int(st[13])+int(st[14]),io=io,status=status,open_feature_indexes=opened))
 if not i:time.sleep(2)
print(json.dumps(a))
'''.replace('__PID__',str(pid))
r=subprocess.run(['srun','--overlap','--jobid='+job,'--gres=gpu:0','--ntasks=1','--cpus-per-task=1','--mem=128M','python3','-c',source],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=12);print(r.stdout)
if r.returncode:raise RuntimeError(r.stderr[-1200:])
