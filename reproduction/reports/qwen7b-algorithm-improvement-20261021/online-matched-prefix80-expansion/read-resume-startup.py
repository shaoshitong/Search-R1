"""Read this job's log and process IO inside its existing allocation only."""
import pathlib,json,re,subprocess,time
S=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/online-mixed-layer-prefix-20261022-2gpu-v1');b=S/'layer18-prefix80';job=json.loads((b/'root-submission.json').read_bytes())['job_id'];entry=pathlib.Path(json.loads((b/'root-submission.json').read_bytes())['entry']['path']).read_text();out={'time':time.time(),'job':job}
for kind in ['output','error']:
 m=re.search(r'^#SBATCH --'+kind+r'=(.+)$',entry,re.M)
 if m:
  p=pathlib.Path(m.group(1).replace('%j',job));out[kind+'_path']=str(p)
  if p.exists():
   with p.open('rb')as f:f.seek(max(0,p.stat().st_size-4500));out[kind]=f.read().decode(errors='replace')
code='''import pathlib,json,os,time,subprocess
rows=[]
for p in pathlib.Path('/proc').glob('[0-9]*'):
 try:
  cg=(p/'cgroup').read_text();cmd=(p/'cmdline').read_bytes().replace(b'\\0',b' ').decode()
  if 'job_JOBID' not in cg or 'online_mixed_depth_' not in cmd:continue
  row={'pid':p.name,'cmd':cmd,'stat':(p/'stat').read_text(),'io':(p/'io').read_text(),'fds':[]}
  for fd in(p/'fd').iterdir():
   try:
    target=os.readlink(str(fd))
    if 'inputs/'in target:row['fds'].append({'path':target,'fdinfo':(p/'fdinfo'/fd.name).read_text()})
   except OSError:pass
  rows.append(row)
 except(OSError,UnicodeError):pass
g=subprocess.run(['nvidia-smi','--query-gpu=index,name,utilization.gpu,memory.used','--format=csv,noheader'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=8)
print(json.dumps({'time':time.time(),'processes':rows,'GPU_readonly':{'returncode':g.returncode,'stdout':g.stdout,'stderr':g.stderr}}))'''.replace('JOBID',job)
compile(code,'own-allocation-reader','exec')
p=subprocess.run(['srun','--jobid='+job,'--overlap','--nodes=1','--ntasks=1','--cpus-per-task=1','--gres=gpu:2','python3','-c',code],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=20);out['allocation_read']={'returncode':p.returncode,'stdout':p.stdout[-12000:],'stderr':p.stderr[-1000:]}
print(json.dumps(out))
