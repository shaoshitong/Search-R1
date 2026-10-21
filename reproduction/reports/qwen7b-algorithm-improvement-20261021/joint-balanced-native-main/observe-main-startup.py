"""Read only this accepted main job and, while running, its own allocation PID."""
import base64,importlib.util,json,shlex,subprocess,zlib
from pathlib import Path
CONTROL=Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5')
PROC='''import pathlib,json,os,socket,time
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/joint-balanced-native-E4-124194-v1');c=json.loads((b/'controller.json').read_bytes());pid=c['pid'];p=pathlib.Path('/proc')/str(pid);out=dict(epoch=time.time(),host=socket.gethostname(),allocation_job=os.environ.get('SLURM_JOB_ID'),controller_pid=pid)
if p.exists():
 cmd=(p/'cmdline').read_bytes().replace(bytes([0]),b' ').decode();assert str(b)in cmd and'joint_balanced_native_main'in cmd,'Unexpected process identity';env=(p/'environ').read_bytes().split(bytes([0]));assert b'SLURM_JOB_ID=745501'in env,'PID does not belong to accepted main allocation';stat=(p/'stat').read_text().rsplit(')',1)[1].split();io={k:int(v)for line in(p/'io').read_text().splitlines()for k,v in[line.split(':')]};rss=[line for line in(p/'status').read_text().splitlines()if line.startswith(('VmRSS:','Threads:'))];out.update(process_exists=True,birth_ticks=int(stat[19]),state=stat[0],CPU_user_ticks=int(stat[11]),CPU_system_ticks=int(stat[12]),io=io,status=rss,cmdline_matches=True)
else:out['process_exists']=False
print(json.dumps(out))
'''
BODY='''import pathlib,json,subprocess,time
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/joint-balanced-native-E4-124194-v1');identity=json.loads((b/'submission-result.json').read_bytes());assert identity['job_id']=='745501';out=dict(epoch=time.time(),job_id='745501')
for key,args in [('queue',['squeue','-h','-j','745501','-o','%i|%T|%M|%l|%R']),('accounting',['sacct','-n','-P','-j','745501','--format=JobIDRaw,State,ExitCode,Elapsed,NodeList'])]:
 r=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=12);out[key]=dict(returncode=r.returncode,rows=r.stdout.strip().splitlines(),stderr=r.stderr[-400:])
for name in('controller','reader-selection','progress','evaluation-progress','first-update-replay','first-cp-independent-audit','durable-next-update-replay','latest','failure','interrupted','complete'):
 try:
  p=b/(name+'.json')
  if p.exists():out[name]=json.loads(p.read_bytes())
 except FileNotFoundError:pass
p=b/'logs/train-745501.log'
if p.exists():
 with p.open('rb')as f:f.seek(max(0,p.stat().st_size-3500));out['log_tail']=f.read().decode(errors='replace')
if any(row.startswith('745501|RUNNING|')for row in out['queue']['rows'])and'controller'in out:
 args=['srun','--overlap','--jobid=745501','--account=mou176','--nodes=1','--ntasks=1','--cpus-per-task=1','--mem=1G','--gres=gpu:0','python3','-c',v['process']]
 try:
  r=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=20);out['allocation_process_probe']=dict(returncode=r.returncode,stderr=r.stderr[-500:]);lines=r.stdout.strip().splitlines()
  if r.returncode==0 and lines:out['allocation_process_probe']['observation']=json.loads(lines[-1])
 except subprocess.TimeoutExpired:out['allocation_process_probe']=dict(status='BOUNDED_OBSERVATION_TIMEOUT_NO_TRAINING_ACTION')
print(json.dumps(out))
'''
spec=importlib.util.spec_from_file_location('observe_native_main_only',CONTROL/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl);ip=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0']).decode().strip();body='v='+repr(dict(process=PROC))+'\n'+BODY;packed=base64.b64encode(zlib.compress(body.encode())).decode();cmd='python3 -c '+shlex.quote('import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(packed)+')))');r=ctl.run(cmd,account='mou176',source_address=ip,timeout=50)
if r.returncode:raise RuntimeError(r.stdout[-3000:]+r.stderr[-1000:])
v=json.loads(r.stdout.splitlines()[-1]);root=Path(__file__).parent;root.joinpath('actual-main-startup-latest.json').write_text(json.dumps(v,indent=2)+'\n')
with root.joinpath('actual-main-startup-observations.jsonl').open('a')as f:f.write(json.dumps(v)+'\n')
print(json.dumps({k:v[k]for k in('epoch','job_id','queue','allocation_process_probe','progress','evaluation-progress','failure','complete')if k in v}))
