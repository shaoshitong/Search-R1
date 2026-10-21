import pathlib,json,subprocess
base=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/joint-balanced-native-E4-qualification-v3')
c=json.loads((base/'controller.json').read_text());pid=c['pid']
code="""import pathlib,json,os,time,socket
p=pathlib.Path('/proc')/str(PID)
out=dict(epoch=time.time(),host=socket.gethostname(),PID=PID,exists=p.exists())
if p.exists():
 stat=(p/'stat').read_text().rsplit(')',1)[1].split();cmd=(p/'cmdline').read_bytes().replace(bytes([0]),b' ').decode(errors='replace')
 out.update(start_ticks=int(stat[19]),state=stat[0],user_ticks=int(stat[11]),system_ticks=int(stat[12]),RSS_pages=int(stat[21]),expected_module_in_command='joint_balanced_native_train'in cmd,expected_namespace_in_command=BASE in cmd)
 try:out['IO']={k:int(v)for k,v in (x.split(':',1)for x in(p/'io').read_text().splitlines())if k in ('rchar','read_bytes','write_bytes')}
 except PermissionError:out['IO_permission']=False
print(json.dumps(out))
"""
code='PID='+repr(pid)+';BASE='+repr(str(base))+bytes([10]).decode()+code
r=subprocess.run(['srun','--jobid=745177','--overlap','--ntasks=1','--cpus-per-task=1','--mem=512M','--gres=gpu:0','python3','-c',code],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=25)
print(json.dumps(dict(returncode=r.returncode,stdout=r.stdout.strip(),stderr=r.stderr[-700:])))
