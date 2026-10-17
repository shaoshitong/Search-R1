"""Read-only CP5/NB allocated metadata; no tensor-content reads."""
import importlib.util,json,shlex,sys,time
from pathlib import Path
WS=Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-6')
OUT=Path(__file__).parent
HPC_PY='/data/user/jxu957/experiments/easy2560-corollout-joint4-20261017/env/bin/python'
mode=sys.argv[1]
if mode=='nb':
 code=(WS/'work/easy2560-co-rollout-hpc3-4gpu-20261017/reproduction/easy2560_joint4/code_backup/micro16-bound-acceptance-and-storage-20261018/observe_micro16_nb_storage_metadata_20261018.py').read_text()
elif mode=='cp':
 code="import json,os,stat,subprocess,time,socket\nfrom pathlib import Path\nbase=Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-micro16-after730080-20261018')\ncp=base/'checkpoints/search-g1-common8/actor/global_step_5'\nassert os.getuid()==14090 and base.resolve()==base and cp.resolve()==cp\nstarted=time.time();rows=[];errors=[]\nfor folder,dirs,names in os.walk(cp,followlinks=False,onerror=lambda e:errors.append(str(e))):\n dirs[:]=[n for n in dirs if not (Path(folder)/n).is_symlink()]\n for name in names:\n  q=Path(folder)/name;s=q.lstat();rows.append({'path':str(q.relative_to(cp)),'mode':s.st_mode,'bytes':s.st_size,'allocated_bytes':s.st_blocks*512,'links':s.st_nlink,'device':s.st_dev,'inode':s.st_ino,'mtime_ns':s.st_mtime_ns,'is_symlink':stat.S_ISLNK(s.st_mode)})\nmeasurements=[]\nfor target in (cp,base):\n r=subprocess.run(['du','-x','-s','-B1','--',str(target)],stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=45)\n measurements.append({'path':str(target),'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr})\ncomplete=cp/'complete.json';s=complete.lstat();assert stat.S_ISREG(s.st_mode) and s.st_size<2*1024**2\nrun=json.loads((base/'logs/run-manifest.json').read_text());assert run['run_id']=='dfe9c94084bb48b2a9bb7b60f9d882da' and run['source_commit']=='8a5bac94b8230a99b81977f5290589e3d3708dfe'\nprint(json.dumps({'role':'exact_CP5_and_NB_allocated_metadata','started_unix':started,'finished_unix':time.time(),'host':socket.gethostname(),'uid':os.getuid(),'run_id':run['run_id'],'source_commit':run['source_commit'],'cp':str(cp),'measurements':measurements,'cp_files':rows,'cp_complete':json.loads(complete.read_text()),'walk_errors':errors,'symlinks_followed':False,'tensor_contents_read':False,'payload_sha_performed':False},sort_keys=True))\n"
else:raise SystemExit('nb or cp')
q=WS/'work/qprobe-sigma3-loss-20261015/reproduction/hpc3_qprobe/local_account_bridge.py'
s=importlib.util.spec_from_file_location('bridge',q);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
r=m.run('jxu957',shlex.join([HPC_PY,'-B','-c',code]),'',str(WS/'work'),login_host='10.120.48.31',expected_host_identity=dict(hostname='ACD-Login-1',boot_id='a27512a3-24a0-4ca3-aa92-98d2a5256d9a',pid_namespace='pid:[4026531836]'),observation_timeout=55)
record=dict(mode=mode,checked_local_unix=time.time(),returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
p=OUT/('cp5-storage-'+mode+'-'+str(int(time.time()))+'.json');p.write_text(json.dumps(record,sort_keys=True,indent=2)+'\n');print(str(p))
if r.returncode==0:
 d=json.loads(r.stdout)
 print(json.dumps({k:d[k] for k in ('role','started_unix','finished_unix','uid','measurements','du_returncode','du_stderr') if k in d},sort_keys=True))
else:print(json.dumps(record,sort_keys=True))
