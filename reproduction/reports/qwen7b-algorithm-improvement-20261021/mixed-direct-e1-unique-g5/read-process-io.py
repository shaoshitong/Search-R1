import subprocess,json
script="""import pathlib,json,time
pid=1017537;p=pathlib.Path('/proc')/str(pid);cmd=(p/'cmdline').read_bytes().replace(b'\\x00',b' ').decode(errors='replace');assert 'joint_mixed_direct_unique'in cmd
x=dict(time=time.time(),pid=pid,io=(p/'io').read_text(),status=[line for line in (p/'status').read_text().splitlines()if line.startswith(('State:','VmRSS:','VmSize:','Threads:'))],stat=(p/'stat').read_text());fds=[]
for f in (p/'fd').iterdir():
 try:
  name=str(f.resolve())
  if any(n in name for n in ('index.jsonl','manifest','checkpoint','prototype')):fds.append(dict(fd=f.name,path=name,position=(p/'fdinfo'/f.name).read_text()))
 except OSError:pass
x['current_files']=fds[:12];print(json.dumps(x))
"""
r=subprocess.run(['srun','--jobid=749015','--overlap','--nodes=1','--ntasks=1','--cpus-per-task=1','--mem=512M','--time=00:00:20','python3','-c',script],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=28);print(json.dumps(dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr[-1000:])))
