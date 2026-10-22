"""Root-only submission of the two immutable, matched P80 pilot entries."""
import pathlib,json,subprocess,time,fcntl,hashlib
S=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/online-mixed-layer-prefix-20261022-2gpu-v1')
guard=(S/'prefix80-submit.lock').open('a+');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
out=[]
for layer in (15,18):
 b=S/('layer%d-prefix80'%layer);ready=json.loads((b/'prefix80-ready.json').read_bytes());receipt=b/'root-submission.json';name=ready['job_name']
 if receipt.exists():out.append(json.loads(receipt.read_bytes()));continue
 queued=subprocess.run(['squeue','-h','-u','mou176','-n',name,'-o','%i|%j|%T'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=15)
 assert queued.returncode==0 and not queued.stdout.strip(),'Existing or unknown submission: '+queued.stdout+queued.stderr
 assert not(b/'controller.json').exists()and not(b/'latest.json').exists(),'Existing training state must be inspected'
 entry=pathlib.Path(ready['entry']['path']);assert hashlib.sha256(entry.read_bytes()).hexdigest()==ready['entry']['sha256']
 intent=b/'root-submission-intent.json';assert not intent.exists(),'Unknown previous outcome: inspect accounting before retry'
 intent.write_text(json.dumps(dict(name=name,entry=ready['entry'],time=time.time()))+'\n')
 p=subprocess.run(['sbatch','--parsable','--exclude=ACD1-13',str(entry)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=30)
 assert p.returncode==0,p.stdout+p.stderr
 job=p.stdout.strip().split(';')[0];assert job.isdigit()
 result=dict(job_id=job,name=name,entry=ready['entry'],runtime_commit=ready['runtime_commit'],time=time.time(),resources=ready['resources'],no_dependencies=True)
 receipt.write_text(json.dumps(result,sort_keys=True)+'\n');out.append(result)
print(json.dumps(dict(submissions=out,time=time.time())))
