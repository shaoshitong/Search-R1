import pathlib,json,hashlib,subprocess,os,time,fcntl
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/direct-count-large-smallCP1-independent-v2');target=b.parent/'direct-count-frozenE1-fit30016-hold5120-v2'
def proof(p):
 p=pathlib.Path(p);raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def read(p):return json.loads(pathlib.Path(p).read_bytes())
lock=(b/'operator.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
if(b/'independent-CP1-acceptance.json').exists():print(json.dumps(dict(status='EXISTING_REAL_CP1_ACCEPTANCE_PRESERVED',result=read(b/'independent-CP1-acceptance.json'),receipt=proof(b/'independent-CP1-acceptance.json'))));raise SystemExit(0)
pub=read(b/'source-publication.json');assert pub['commit']=='8fce2e4683131d1d7c8de35c0e6a40a4fc693af6'
for item in pub['files'].values():assert proof(item['path'])==item
assert(b/'fixture-complete.json').exists()
s=subprocess.check_output(['scontrol','show','job','-o','743978'],universal_newlines=True);j={k:s.split(k+'=',1)[1].split()[0]for k in ('JobId','JobName','UserId','JobState','Command','WorkDir')}
assert j['JobId']=='743978'and j['JobName']=='q7b-frozenE1-head30016'and j['JobState']=='RUNNING'and j['UserId'].endswith('('+str(os.getuid())+')')and j['WorkDir']==str(target)and j['Command']==str(target/'capacity.sbatch')
cp=target/'checkpoints/cursor-000001.pt';assert 0<cp.stat().st_size<=64*1024**2
runtime=pub['runtime_root'];module=pub['files']['direct_count_large_small_checkpoint.py']['path'];intent=b/'audit-inner-env-intent.json';assert not intent.exists(),'Preserve unknown prior attempt; no duplicate'
intent.write_text(json.dumps(dict(epoch=time.time(),job=j,source=proof(b/'source-publication.json'),CP1_actual_bytes=cp.stat().st_size,launcher_fix='Set empty CUDA_VISIBLE_DEVICES using env INSIDE srun, without relaxing module guard'))+'\n')
cmd=['srun','--jobid=743978','--overlap','--gres=gpu:0','--ntasks=1','--cpus-per-task=2','--mem=3G','env','CUDA_VISIBLE_DEVICES=','QWEN7B_CODE_ROOT='+runtime,'OMP_NUM_THREADS=2','MKL_NUM_THREADS=2','OPENBLAS_NUM_THREADS=2','PYTHONDONTWRITEBYTECODE=1','bash',runtime+'/reproduction/operations/qwen7b_warmup/runtime.sh','-u',module,'--base',str(target),'--config-sha256','fb637c02812dd070cf392a58df75e0d219bbb10d1a2d24cd6e201f3638147767','--output',str(b/'independent-CP1-acceptance.json')]
start=time.time();log=b/'audit-inner-env-cpu.log'
with log.open('w')as stream:r=subprocess.run(cmd,stdout=stream,stderr=subprocess.STDOUT,timeout=40,universal_newlines=True)
text=log.read_text();rows=[json.loads(x)for x in text.splitlines()if x.startswith('{')];out=dict(status='REAL_CP1_INNER_ENV_LAUNCH_COMPLETED',returncode=r.returncode,seconds=time.time()-start,job=j,log=proof(log),result=rows[-1]if rows else None)
(b/'audit-inner-env-step-complete.json').write_text(json.dumps(out,sort_keys=True)+'\n');print(json.dumps(out));assert r.returncode==0,text[-2500:]
