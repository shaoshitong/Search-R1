"""Compact readonly source/progress/process/cgroup observer for owned eval746247."""
import base64,hashlib,importlib.util,json,shlex,subprocess,zlib
from pathlib import Path
R=Path(__file__).resolve().parents[3];CONTROL=R.parents[1];REL='reproduction/operations/qwen7b_warmup/observe_mixed_direct_cp512.py';BRANCH='codex/mixed-direct-cp512-eval-20261022';OUT=R/'reproduction/reports/qwen7b-algorithm-improvement-20261021/mixed-direct-cp512-eval'
PROC='''import pathlib,os,json,socket
p=pathlib.Path('/proc')/str(V['pid']);argv=(p/'cmdline').read_bytes().split(bytes([0]));argv=[x.decode()for x in argv if x];env=(p/'environ').read_bytes().split(bytes([0]));assert p.stat().st_uid==os.getuid()and V['module']in argv and argv[argv.index('--base')+1]==V['base']and('SLURM_JOB_ID='+V['job']).encode()in env;fields=(p/'stat').read_text().rsplit(')',1)[1].split();io={k:int(x)for k,x in(line.split(':',1)for line in(p/'io').read_text().splitlines())};cgroup=(p/'cgroup').read_text();print(json.dumps(dict(pid=V['pid'],birth=fields[19],state=fields[0],CPU_ticks=int(fields[11])+int(fields[12]),RSS_bytes=int(fields[21])*os.sysconf('SC_PAGE_SIZE'),rchar=io['rchar'],read_bytes=io['read_bytes'],node=socket.gethostname(),uid=p.stat().st_uid,actual_cmdline_base_job_exact=True,cgroup=cgroup,cgroup_contains_job=('job_'+V['job'])in cgroup)))
'''
BODY='''import pathlib,json,hashlib,subprocess,time,os,base64
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/mixed-direct-official-cp512-eval-v1');job='746247'
def read(p):
 p=pathlib.Path(p);assert p.stat().st_size<8*1024**2 and p.suffix!='.pt';return json.loads(p.read_bytes())
def proof(p):
 p=pathlib.Path(p);assert p.stat().st_size<8*1024**2 and p.suffix!='.pt';raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
def optional(n):
 for attempt in range(3):
  try:return read(b/n)
  except FileNotFoundError:
   if attempt<2:time.sleep(.03)
 return None
assert read(b/'submission-result.json')['job_id']==job;ready=read(b/'preparation.json');assert proof(b/'evaluation-config.json')==ready['evaluation_config']and proof(b/'source-publication.json')==ready['publication'];pub=read(b/'source-publication.json');assert len(pub['files'])==26
for n,item in pub['files'].items():assert proof(pathlib.Path(pub['code_root'])/n)==dict(path=str(pathlib.Path(pub['code_root'])/n),**item)
v=read(b/'evaluation-config.json');assert proof(v['source_config']['path'])==v['source_config'];cp=v['checkpoint'];pin=read(b/'pin-complete.json');s=pathlib.Path(cp['path']).stat();assert s.st_ino==pin['pinned_stat']['inode']and s.st_size==cp['bytes']and s.st_uid==os.getuid();r=subprocess.run(['sacct','-X','-n','-P','-j',job,'--format=JobIDRaw,State,ExitCode,Elapsed,NodeList'],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=15);assert r.returncode==0;out=dict(epoch=time.time(),job_id=job,main_sacct=r.stdout.strip(),source26_fresh_SHA_passed=True,source_config=v['source_config'],evaluation_config=ready['evaluation_config'],checkpoint_cursor=512,pin_inode_size_uid_stable=True,no_checkpoint_payload_read=True,complete_exists=(b/'complete.json').exists(),prediction_files={p.name:p.stat().st_size for p in(b/'predictions').glob('*')if p.is_file()})
controller=optional('controller.json')
if controller:
 out['controller']={k:controller[k]for k in('pid','job_id','stage','time','config_sha256')};assert controller['job_id']==job
 if '|RUNNING|'in out['main_sacct']and not out['complete_exists']:
  code='V='+repr(dict(pid=controller['pid'],job=job,module='reproduction.operations.qwen7b_warmup.evaluate_mixed_direct_checkpoint512',base=str(b)))+chr(10)+PROC;packed=base64.b64encode(code.encode()).decode();inner='import base64;exec(base64.b64decode('+repr(packed)+'))'
  try:
   r=subprocess.run(['srun','--overlap','--jobid='+job,'--gres=gpu:0','--cpus-per-task=1','--mem=128M','--time=00:01:00','python3','-c',inner],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=15)
   if r.returncode==0:out['actual_process']=json.loads(r.stdout.strip().splitlines()[-1])
   else:out['process_observation_error']=dict(returncode=r.returncode,stderr_tail=r.stderr[-1000:])
  except subprocess.TimeoutExpired:out['process_observation_error']=dict(type='Bounded15sCPU0GPUObservationTimeout')
for n in('evaluation-progress.json','failure.json'):
 x=optional(n)
 if x is not None:out[n[:-5]]=x
x=optional('restore-complete.json')
if x:out['restore']={k:x[k]for k in('cursor','full_CP_fresh_SHA_verified','actual_complete_restore_passed','training_features_reader_constructed')};out['restore']['receipt']=proof(b/'restore-complete.json')
if out['complete_exists']:out['complete']=proof(b/'complete.json')
print(json.dumps(out,sort_keys=True))
'''
def main():
 commit=subprocess.check_output(['git','log','-1','--format=%H','refs/heads/'+BRANCH,'--',REL],cwd=R).decode().strip();assert Path(__file__).read_bytes()==subprocess.check_output(['git','show',commit+':'+REL],cwd=R);remote=subprocess.check_output(['git','ls-remote','origin','refs/heads/'+BRANCH],cwd=R).decode().split()[0];subprocess.check_call(['git','merge-base','--is-ancestor',commit,remote],cwd=R);source='PROC='+repr(PROC)+'\n'+BODY;compile(source,'owned_cp512_readonly','exec');spec=importlib.util.spec_from_file_location('private_cp512_observation',CONTROL/'work/prefix-schedule-cp50-20261018/reproduction/hpc3_prefix5/direct_control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl);ip=subprocess.check_output(['/usr/sbin/ipconfig','getifaddr','en0']).decode().strip();packed=base64.b64encode(zlib.compress(source.encode())).decode();command='python3 -c '+shlex.quote('import base64,zlib;exec(zlib.decompress(base64.b64decode('+repr(packed)+')))');assert len(command.encode())<16000;r=ctl.run(command,account='mou176',source_address=ip,timeout=50)
 if r.returncode:raise RuntimeError(r.stdout[-2000:]+r.stderr[-2500:])
 x=json.loads(r.stdout.splitlines()[-1]);OUT.mkdir(parents=True,exist_ok=True);(OUT/'actual-readonly-observation.json').write_text(json.dumps(x,indent=2)+'\n')
 with(OUT/'actual-readonly-observations.jsonl').open('a')as f:f.write(json.dumps(x,sort_keys=True)+'\n')
 print(json.dumps(x))
if __name__=='__main__':main()
