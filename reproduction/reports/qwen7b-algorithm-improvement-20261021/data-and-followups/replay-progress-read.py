import pathlib,json,subprocess,time
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/replay-layer-qualification-20261021');j=(b/'job.id').read_text().strip();assert j=='742884';o=dict(time=time.time(),job_id=j)
for k,c in [('queue',['squeue','-h','-j',j,'-o','%i|%T|%P|%M|%N|%R']),('accounting',['sacct','-X','-n','-P','-j',j,'-o','JobID,State,ExitCode,Elapsed'])]:
 o[k]=subprocess.check_output(c,universal_newlines=True,timeout=15)
for n in ['progress.json','complete.json','native-layer26-comparison.json']:
 p=b/'qualification-output'/n
 if p.exists():o[n]=json.loads(p.read_text())
p=b/'logs'/('replay-'+j+'.log')
if p.exists():
 o['log_bytes']=p.stat().st_size
 with p.open('rb') as f:f.seek(max(0,p.stat().st_size-4500));o['log_tail']=f.read().decode(errors='replace')
print(json.dumps(o))
