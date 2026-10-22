import json,subprocess,time,re
from pathlib import Path
root=Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fastpath-probe-20261022/diagnostics/native4-clip-geometry-remaining-measurements-r1-20261023');job=(root/'job.id').read_text().strip();value=dict(epoch=time.time(),root=str(root),job=job,production_modified=False)
for key,cmd in {'queue':['squeue','-h','-j',job+',749598,745198','-o','%i|%u|%T|%N|%R'],'accounting':['sacct','-n','-X','-j',job,'-o','JobIDRaw,User,State,ExitCode,Elapsed,NodeList','-P']}.items():
 result=subprocess.run(cmd,capture_output=True,text=True,timeout=15);value[key]=dict(returncode=result.returncode,stdout=result.stdout,stderr=result.stderr)
p=root/('slurm-'+job+'.log');tail=''
if p.is_file():
 with p.open('rb') as stream:stream.seek(max(0,p.stat().st_size-48000));tail=stream.read().decode(errors='replace')
 value.update(log_bytes=p.stat().st_size,log_mtime=p.stat().st_mtime)
value['actual_error_lines']=[line[-1200:] for line in tail.splitlines() if re.search(r'Traceback|CUDA out of memory|RuntimeError:|AssertionError:|ValueError:',line)][-8:]
value['progress']=[json.loads(line.split('CLIP_MEASUREMENT_PROGRESS=',1)[1]) for line in tail.splitlines() if 'CLIP_MEASUREMENT_PROGRESS=' in line]
p=root/'result.json'
if p.is_file():value['result']=json.loads(p.read_text())
value['log_tail']=tail[-1800:]
print(json.dumps(value),flush=True)
