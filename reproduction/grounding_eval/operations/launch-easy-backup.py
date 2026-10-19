import sys,subprocess,json,pathlib,base64
sys.path.insert(0,'/Users/shaoshitong/Documents/remote_codex_control');import control
worker=pathlib.Path('work/grounding-eval-strict-20261019/reproduction/grounding_eval/backup_h100.py').read_text()
code='WORKER='+repr(worker)+'\n'+r'''
import os,json,pathlib,subprocess,time,hashlib
base=pathlib.Path('/mnt/data/shaoshitong/grounding-eval-paper-strict-v1-20261019')
base.mkdir(parents=True,exist_ok=True)
source=pathlib.Path('/mnt/shared/workspace/training_data_node_8/shaoshitong/search-g1-speed-resume5-0553ade-20261012/checkpoints/search-g1-common8/actor/global_step_200')
dest=base/'assets/easy-selection-iter200';dest.mkdir(parents=True,exist_ok=True)
control=base/'control';control.mkdir(exist_ok=True)
script=control/'backup_h100.py';script.write_text(WORKER)
launch=control/'easy-backup-launch.json'
if launch.exists():
 prior=json.loads(launch.read_text());pid=prior.get('pid')
 try: alive=os.path.exists('/proc/%d/cmdline'%pid) and 'backup_h100.py' in pathlib.Path('/proc/%d/cmdline'%pid).read_text()
 except Exception:alive=False
 status=dest/'backup-status.json'
 if alive or (status.exists() and json.loads(status.read_text()).get('state')=='COMPLETE'):
  print(json.dumps({'existing':prior,'alive':alive,'status':json.loads(status.read_text()) if status.exists() else None}));raise SystemExit
args=['/usr/bin/python3',str(script),'--source',str(source),'--destination',str(dest)]
log=open(control/'easy-backup.log','ab',buffering=0)
p=subprocess.Popen(args,stdout=log,stderr=subprocess.STDOUT,start_new_session=True,stdin=subprocess.DEVNULL)
receipt={'pid':p.pid,'args':args,'started_at':time.time(),'script_sha256':hashlib.sha256(WORKER.encode()).hexdigest(),'source_exists':source.exists(),'destination':str(dest)}
with open(launch,'w') as f:json.dump(receipt,f,indent=2);f.flush();os.fsync(f.fileno())
print(json.dumps(receipt))
'''
r=subprocess.run(control.route('h100-10','/usr/bin/python3 -'),input=code,capture_output=True,text=True,timeout=60)
p=pathlib.Path('outputs/grounding-eval-20261019/easy-backup-launch.json');p.write_text(json.dumps({'rc':r.returncode,'stdout':r.stdout,'stderr':r.stderr},indent=2));print(r.returncode,r.stdout,r.stderr[-1000:])
