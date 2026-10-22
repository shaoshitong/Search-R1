import json,subprocess,time
from pathlib import Path
base=Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-retrieval-batch-wait16-service-r1-20261023')
job=(base/'job.id').read_text().strip()
def tail(path,size=4500):
 if not path.exists():return None
 with path.open('rb') as f:f.seek(max(0,path.stat().st_size-size));return f.read().decode(errors='replace')
print(json.dumps(dict(epoch=time.time(),job=job,queue=subprocess.run(['squeue','-h','-j',job,'-o','%i|%T|%N|%R'],capture_output=True,text=True,timeout=10).stdout,accounting=subprocess.run(['sacct','-n','-P','-X','-j',job,'-o','JobID,State,ExitCode,Elapsed,NodeList'],capture_output=True,text=True,timeout=10).stdout,status=json.loads((base/'status.json').read_text()) if (base/'status.json').exists() else None,ready=(base/'logs/service-ready.json').exists(),log_tail=tail(base/('logs/service-'+job+'.log')))))
