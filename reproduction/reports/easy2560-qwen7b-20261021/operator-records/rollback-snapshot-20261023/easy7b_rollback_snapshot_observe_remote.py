import json,subprocess,time,re
from pathlib import Path
root=Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fastpath-probe-20261022/diagnostics/native4-rollback-snapshot-r1-20261023');job=(root/'job.id').read_text().strip();r={'epoch':time.time(),'job':job,'root':str(root),'production_modified':False}
for key,cmd in {'queue':['squeue','-h','-j',job+',749598,745198','-o','%i|%u|%T|%N|%R'],'accounting':['sacct','-n','-X','-j',job,'-o','JobIDRaw,User,State,ExitCode,Elapsed,NodeList','-P']}.items():
 p=subprocess.run(cmd,capture_output=True,text=True,timeout=15);r[key]={'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
p=root/('slurm-'+job+'.log');tail=''
if p.is_file():
 with p.open('rb') as stream:stream.seek(max(0,p.stat().st_size-96000));tail=stream.read().decode(errors='replace')
 r['log_bytes']=p.stat().st_size;r['log_mtime']=p.stat().st_mtime
r['snapshot_progress']={}
for line in tail.splitlines():
 if 'ROLLBACK_SNAPSHOT_TIMING=' not in line:continue
 try:
  v=json.loads(line.split('ROLLBACK_SNAPSHOT_TIMING=',1)[1]);r['snapshot_progress'][str(v['rank'])]=v
 except json.JSONDecodeError:pass
r['actual_error_lines']=[line[-1500:] for line in tail.splitlines() if re.search(r'Traceback|CUDA out of memory|RuntimeError:|AssertionError:|ValueError:',line)][-10:]
r['reports']={}
for rank in range(4):
 p=root/('snapshot-timing-rank-%d.json'%rank)
 if p.is_file():r['reports'][str(rank)]=json.loads(p.read_text())
p=root/'result.json'
if p.is_file():r['benchmark_result']=json.loads(p.read_text())
r['log_tail']=tail[-4500:]
print(json.dumps(r),flush=True)
