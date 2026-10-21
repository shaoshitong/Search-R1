import json,pathlib,time,subprocess,fcntl
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/joint-balanced-native-E4-qualification-v3')
out=dict(epoch=time.time(),base=str(b))
for n in ('submission-identity','controller','reader-selection','progress','first-update-replay','first-cp-independent-audit','durable-next-update-replay','latest','failure','interrupted','complete'):
 p=b/(n+'.json')
 if p.exists():
  try:out[n]=json.loads(p.read_text())
  except FileNotFoundError:pass
for args,key in [(['squeue','-h','-u','mou176','-n','q7b-native-joint-qual','-o','%i|%j|%T|%M|%R'],'squeue'),(['sacct','-X','-n','-P','-j',out.get('submission-identity',{}).get('job_id','745177'),'--format=JobIDRaw,State,ExitCode,Elapsed,NodeList'],'sacct')]:
 r=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=12);out[key]=dict(returncode=r.returncode,stdout=r.stdout.strip(),stderr=r.stderr[-400:])
c=out.get('controller',{});pid=c.get('pid')
if pid:
 p=pathlib.Path('/proc')/str(pid)
 if p.exists():
  stat=(p/'stat').read_text().rsplit(')',1)[1].split();cmd=(p/'cmdline').read_bytes().replace(bytes([0]),b' ').decode(errors='replace');io={}
  try:io={k:int(v)for k,v in (x.split(':',1)for x in(p/'io').read_text().splitlines())if k in ('rchar','read_bytes','write_bytes')}
  except PermissionError:pass
  out['observed_PID']=dict(pid=pid,start_ticks=int(stat[19]),state=stat[0],user_ticks=int(stat[11]),system_ticks=int(stat[12]),RSS_pages=int(stat[21]),expected_native_module_in_command='joint_balanced_native_train'in cmd,expected_namespace_in_command=str(b)in cmd,IO=io)
 else:out['observed_PID']=dict(pid=pid,exists=False,note='Controller PID may run on compute node; login /proc absence is not process exit')
print(json.dumps(out))
