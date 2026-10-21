import pathlib,subprocess,json,os,hashlib,shlex,time
B=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/joint-balanced-native-E4-124194-v1');job='745501';v=subprocess.check_output(['scontrol','show','job','-o',job],universal_newlines=True);d={x.split('=',1)[0]:x.split('=',1)[1]for x in v.split()if'='in x};assert d['UserId'].startswith('mou176(')and d['JobState']=='RUNNING'and d['NodeList']=='ACD1-28'and d['Command']==str(B/'training.sbatch')
worker=r'''
import pathlib,os,json,time,hashlib,concurrent.futures,resource
b=pathlib.Path(BASE);pid=300719;p=pathlib.Path('/proc/%d'%pid);assert p.exists()and p.joinpath('stat').read_text().rsplit(')',1)[1].split()[19]=='116543024';assert str(b).encode()in p.joinpath('cmdline').read_bytes();assert not os.environ.get('CUDA_VISIBLE_DEVICES');assert len(os.sched_getaffinity(0))<=2
c=json.loads((b/'config.json').read_text());assert hashlib.sha256((b/'config.json').read_bytes()).hexdigest()=='5aaa63c9348f410357be92ce491ba6d18286d618ab86dae6e63f193f2b4367b6';fs=json.loads(pathlib.Path(c['features']['path']).read_text());opens=[]
for fd in p.joinpath('fd').iterdir():
 try:opens.append(str(fd.resolve()))
 except FileNotFoundError:pass
hit=[i for i,v in enumerate(fs)if any(s.startswith(str(pathlib.Path(v['path']).parent)+'/')for s in opens)];assert hit,'No actual main payload fd observed; refuse speculative benchmark';start=max(hit)+16;chosen=[]
for item in fs[start:start+96]:
 m=json.loads(pathlib.Path(item['path']).read_text());vals=sorted(m['files'].values(),key=lambda x:x['bytes'],reverse=True)
 for f in vals:
  if f['bytes']>=32*1024**2 and f['path']not in {v['path']for v in chosen}:chosen.append(f);break
 if len(chosen)==32:break
assert len(chosen)==32
resource.setrlimit(resource.RLIMIT_CPU,(100,100));resource.setrlimit(resource.RLIMIT_DATA,(2*1024**3,2*1024**3));deadline=time.monotonic()+95

def one(f):
 h=hashlib.sha256();n=0
 with open(f['path'],'rb',buffering=0)as inp:
  for _ in range(8):
   assert time.monotonic()<deadline,'Finite benchmark deadline';raw=inp.read(4*1024**2);assert len(raw)==4*1024**2;h.update(raw);n+=len(raw)
 return n,h.hexdigest()
results=[]
for trial,(threads,lo)in enumerate(((1,0),(8,8),(8,16),(1,24))):
 group=chosen[lo:lo+8];t=time.monotonic()
 if threads==1:r=[one(x)for x in group]
 else:
  with concurrent.futures.ThreadPoolExecutor(max_workers=8)as pool:r=list(pool.map(one,group))
 seconds=time.monotonic()-t;n=sum(x[0]for x in r);results.append(dict(trial=trial,threads=threads,bytes=n,seconds=seconds,MBps=n/seconds/1e6,ordered_range_sha256=hashlib.sha256(json.dumps([x[1]for x in r]).encode()).hexdigest()));print(json.dumps(dict(status='ACTUAL_READ_ONLY_IO_TRIAL_COMPLETE',result=results[-1])),flush=True)
print(json.dumps(dict(status='ALLOCATION_READ_ONLY_IO_BENCHMARK_COMPLETE',job_id=os.environ['SLURM_JOB_ID'],hostname=os.uname().nodename,main_pid=pid,main_birth='116543024',main_manifest_index=max(hit),sample_manifest_start=start,range_bytes_per_file=32*1024**2,chunk_bytes=4*1024**2,files=32,actual_affinity=sorted(os.sched_getaffinity(0)),maxRSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,results=results,main_code_and_checksums_unchanged=True,full_file_SHA_claim=False)),flush=True)
'''.replace('BASE',repr(str(B)))
args=['srun','--jobid='+job,'--overlap','--exact','--ntasks=1','--cpus-per-task=2','--mem=2G','--gres=gpu:0','--time=00:02:00','env','CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=2','MKL_NUM_THREADS=2','OPENBLAS_NUM_THREADS=2','python3','-u','-c',worker]
t=time.time();r=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=115);print(json.dumps(dict(status='BOUNDED_MAIN_ALLOCATION_IO_BENCHMARK_EXIT',exit_code=r.returncode,elapsed=time.time()-t,stdout=r.stdout,stderr=r.stderr[-1800:])))
