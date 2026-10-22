"""Exclude the already-observed CUDA-init-failing node; preserve failed attempt."""
import pathlib,json,subprocess,shutil,time,fcntl
S=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/online-mixed-layer-prefix-20261022-2gpu-v1')
guard=(S/'prefix80-submit.lock').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
def run(cmd):
 p=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=20)
 assert p.returncode==0,p.stdout+p.stderr
 return p.stdout.strip()
out={'time':time.time()}
b=S/'layer18-prefix80';r=json.loads((b/'root-submission.json').read_bytes());job=r['job_id']
state=run(['squeue','-h','-j',job,'-o','%T|%N'])
if state.startswith('PENDING|'):
 run(['scontrol','update','JobId='+job,'ExcNodeList=ACD1-13']);out['existing_pending_updated']={'job':job,'excluded':'ACD1-13','actual':run(['scontrol','show','job',job,'-o'])}
else:out['existing_job_unchanged']={'job':job,'state':state}
b=S/'layer15-prefix80';receipt=b/'root-submission.json';r=json.loads(receipt.read_bytes())
if r['job_id']!='749684':out['replacement_already_recorded']=r
else:
 actual=run(['sacct','-X','-n','-P','-j','749684','--format=JobIDRaw,State,ExitCode,NodeList']);assert actual=='749684|FAILED|1:0|ACD1-13',actual
 assert not(b/'controller.json').exists()and not(b/'latest.json').exists()and not(b/'progress-rank-0.json').exists()
 failures=list(b.glob('failure-rank-*.json'));assert failures
 for f in failures:
  v=json.loads(f.read_bytes());assert v['type']=='RuntimeError'and'cuda_setDevice'in v['traceback']and'CUDA unknown error'in v['message']
 assert not run(['squeue','-h','-u','mou176','-n',r['name'],'-o','%i|%T'])
 archive=b/'attempts/cuda-init-failed-749684';archive.mkdir(parents=True,exist_ok=True)
 for f in failures+[receipt,b/'root-submission-intent.json']:
  if f.exists():shutil.move(str(f),str(archive/f.name))
 intent=b/'root-submission-intent.json';intent.write_text(json.dumps({'time':time.time(),'reason':'CUDA init on previously observed bad ACD1-13; zero optimizer updates','exclude':'ACD1-13'})+'\n')
 entry=b/'training.sbatch';job=run(['sbatch','--parsable','--exclude=ACD1-13',str(entry)]).split(';')[0];assert job.isdigit()
 r.update(job_id=job,time=time.time(),previous_failed_attempt='749684',exclude='ACD1-13',replacement_not_optimizer_resume=True,optimizer_updates_lost=0)
 receipt.write_text(json.dumps(r,sort_keys=True)+'\n');out['replacement']=r
(S/'prefix80-CUDA-recovery.json').write_text(json.dumps(out,sort_keys=True)+'\n');print(json.dumps(out))
