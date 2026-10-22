import pathlib,json,subprocess,time,shutil
S=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/online-mixed-layer-prefix-20261022-2gpu-v1');out={}
for layer in(15,18):
 b=S/('layer%d-prefix80'%layer);job=json.loads((b/'root-submission.json').read_bytes())['job_id'];cp=json.loads((b/'latest.json').read_bytes());assert cp['path'].endswith('epoch-02.pt');ps=[json.loads((b/('progress-rank-%d.json'%k)).read_bytes())for k in(0,1)];assert all(p['cursor']==128 for p in ps)
 code="""import pathlib,json
rows=[]
for p in pathlib.Path('/proc').glob('[0-9]*'):
 try:
  cmd=(p/'cmdline').read_bytes().replace(b'\\0',b' ').decode();cg=(p/'cgroup').read_text();stat=(p/'stat').read_text().rsplit(')',1)[1].split()
  if 'job_JOB' in cg and 'python' in cmd and '-m reproduction.operations.qwen7b_warmup.online_mixed_depth_expansion ' in cmd and '--local_rank'not in cmd and 'import pathlib'not in cmd:rows.append({'pid':p.name,'state':stat[0],'cmd':cmd,'cgroup':cg,'status':(p/'status').read_text(),'freezer':pathlib.Path('/sys/fs/cgroup/freezer/slurm/uid_21930/job_JOB/step_0/freezer.state').read_text().strip()})
 except(OSError,UnicodeError):pass
print(json.dumps(rows))""".replace('JOB',job)
 p=subprocess.run(['srun','--jobid='+job,'--overlap','--nodes=1','--ntasks=1','--cpus-per-task=1','--gres=gpu:2','python3','-c',code],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=20)
 o={'time':time.time(),'job_id':job,'rc':p.returncode,'stdout':p.stdout,'stderr':p.stderr,'checkpoint':cp}
 if p.returncode==0:
  rows=json.loads(p.stdout.strip().splitlines()[-1]);assert len(rows)>=2 and all(r['state']in('T','t')or r['freezer']=='FROZEN'for r in rows),rows
  h={'time':time.time(),'job_id':job,'epoch':2,'cursor':128,'checkpoint':cp,'actual_GPU_rank_processes':rows,'actual_GPU_step_stopped':True,'evaluation_pending':not(b/'evaluation-epoch-02.json').exists(),'discarded_optimizer_updates':0}
  old=b/'walltime-boundary-held.json'
  if old.exists():shutil.copy2(str(old),str(b/'walltime-boundary-held-epoch1-historical.json'))
  old.write_text(json.dumps(h,sort_keys=True)+'\n');o['held']=h
 out[str(layer)]=o
print(json.dumps(out))
