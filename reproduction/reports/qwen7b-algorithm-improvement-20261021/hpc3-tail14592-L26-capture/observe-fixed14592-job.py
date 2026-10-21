"""Bounded own-allocation PID/I/O and real byte/chunk observer; no job action."""
import base64,importlib.util,json,shlex,subprocess,zlib
from pathlib import Path
CONTROL=Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-5')
PROC='''import pathlib,os,json,time
root=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/hpc3-tail14592-independent-L26-prepared-v1');script=str(root/'code-757da0b096c5/capture_hpc3_tail14592_l26.py');result=[]
for p in pathlib.Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  if p.stat().st_uid!=os.getuid():continue
  argv=(p/'cmdline').read_bytes().split(bytes([0]))
  if script.encode()not in argv:continue
  env=(p/'environ').read_bytes().split(bytes([0]));assert b'SLURM_JOB_ID=745932'in env;stat=(p/'stat').read_text().rsplit(')',1)[1].split();io={k:int(v)for line in(p/'io').read_text().splitlines()for k,v in[line.split(':')]};result.append(dict(pid=int(p.name),birth_ticks=stat[19],state=stat[0],CPU_user_ticks=int(stat[11]),CPU_system_ticks=int(stat[12]),io=io,cmdline_matches=True,action='capture'if b'capture'in argv else'merge'))
 except(FileNotFoundError,PermissionError,ProcessLookupError):continue
print(json.dumps(dict(epoch=time.time(),allocation_job=os.environ.get('SLURM_JOB_ID'),own_processes=result)))
'''
BODY='''import pathlib,json,subprocess,time
def optional(p):
 for attempt in range(3):
  try:return json.loads(p.read_bytes())
  except FileNotFoundError:
   if attempt<2:time.sleep(.03)
 return None
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/hpc3-tail14592-independent-L26-prepared-v1');assert json.loads((b/'submission-result.json').read_bytes())['job_id']=='745932';o=dict(epoch=time.time(),job_id='745932');q=subprocess.check_output(['squeue','-h','-j','745932','-o','%i|%T|%M|%l|%R'],universal_newlines=True).strip();o['squeue']=q;o['main_sacct']=subprocess.check_output(['sacct','-X','-n','-P','-j','745932','--format=JobIDRaw,State,ExitCode,Elapsed'],universal_newlines=True).strip();shards={}
for i in range(4):
 cap=b/('shard-%02d'%i)/'capture';x=dict(complete_chunks=len(list(cap.glob('chunk-*/complete.json'))),actual_packed_bytes=sum(p.stat().st_size for p in cap.glob('chunk-*/layer-26/*')if p.is_file()))
 for n in('progress.json','failure.json','complete.json','device.json'):
  p=cap/n
  value=optional(p)
  if value is not None:x[n]=value
 shards[str(i)]=x
o['shards']=shards
for n in('bank-acceptance-progress.json','bank/complete.json'):
 p=b/n
 value=optional(p)
 if value is not None:o[n]=value
if '|RUNNING|'in q:
 r=subprocess.run(['srun','--overlap','--jobid=745932','--account=mou176','--nodes=1','--ntasks=1','--cpus-per-task=1','--mem=1G','--gres=gpu:0','python3','-c',v['proc']],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=20);o['process_probe_returncode']=r.returncode
 if not r.returncode:o['process_probe']=json.loads(r.stdout.splitlines()[-1])
 else:o['process_probe_error']=r.stderr[-1000:]
o['log_tail']={p.name:p.read_text(errors='replace')[-1800:]for p in(b/'logs').glob('*.log')};print(json.dumps(o))
'''
for s in(PROC,BODY):compile(s,'own-fixed14592-observer','exec')
spec=importlib.util.spec_from_file_location('own14592_only',CONTROL/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl);ip=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0']).decode().strip();body='v='+repr(dict(proc=PROC))+'\n'+BODY;packed=base64.b64encode(zlib.compress(body.encode())).decode();cmd='python3 -c '+shlex.quote('import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(packed)+')))');r=ctl.run(cmd,account='mou176',source_address=ip,timeout=40)
if r.returncode:raise RuntimeError(r.stdout[-1500:]+r.stderr[-1500:])
v=json.loads(r.stdout.splitlines()[-1]);root=Path(__file__).parent;(root/'actual-capture-latest.json').write_text(json.dumps(v,indent=2)+'\n')
with(root/'actual-capture-observations.jsonl').open('a')as f:f.write(json.dumps(v)+'\n')
print(json.dumps(v))
