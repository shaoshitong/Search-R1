"""Follow owned final job metadata after actual freeze; never choose using final scores."""
import base64,hashlib,json,pathlib,subprocess,time
A=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021')
B=A/'frozen-final4096-full169028-L15P80-MSE-20261023-v2';C=A/'fullP80-selected-root-control-20261023-v1'
def read(p):return json.loads(p.read_bytes())
def command(args):
 r=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=20)
 return dict(returncode=r.returncode,stdout=r.stdout.strip(),stderr=r.stderr[-700:])
out=dict(time=time.time(),read_only=True,checkpoint_tensors_loaded=False,files={})
freeze=C/'candidate-freeze.json';raw=freeze.read_bytes();assert hashlib.sha256(raw).hexdigest()=='2606a04636e282df2e011c962fe47c29c00be4460f3581f3fc03b58fd5d428ca'
out['actual_freeze_raw_b64']=base64.b64encode(raw).decode()
for n in('final-submission.json','final-submit-intent.json'):
 p=C/n
 if p.is_file():out['files'][n]=read(p)
for n in('controller.json','progress-rank-0.json','progress-rank-1.json','failure-rank-0.json','failure-rank-1.json','final-use-started.json','confirmation2048-complete.json','stability2048-complete.json','complete.json'):
 p=B/n
 if p.is_file():out['files'][n]=read(p)
sub=out['files']['final-submission.json'];job=str(sub['job_id']);assert job=='750481'
out['squeue']=command(['squeue','-h','-j',job,'-o','%i|%T|%M|%R'])
out['sacct']=command(['sacct','-X','-n','-P','-j',job,'--format=JobIDRaw,State,ExitCode,Elapsed,NodeList'])
out['scontrol']=command(['scontrol','show','job',job,'-o'])
log=B/('score-'+job+'.log')
if log.is_file():
 with log.open('rb')as f:
  f.seek(max(0,log.stat().st_size-6000));out['log_tail']=f.read().decode(errors='replace').splitlines()[-14:]
print(json.dumps(out))
