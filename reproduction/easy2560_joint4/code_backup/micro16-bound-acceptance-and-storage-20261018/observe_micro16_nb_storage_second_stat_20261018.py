import json,os,stat,time
from pathlib import Path
b=Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-micro16-after730080-20261018');started=time.time();assert b.resolve()==b and b.stat().st_uid==os.getuid()==14090
rows=[];directories=[];errors=[]
for top in ('artifacts/mixed-candidate-pools','artifacts/behavior-traces','artifacts/mixed-selection','logs'):
 for folder,dirs,names in os.walk(b/top,followlinks=False,onerror=lambda e:errors.append(str(e))):
  path=Path(folder);s=path.lstat();directories.append({'path':str(path.relative_to(b)),'allocated_bytes':s.st_blocks*512})
  dirs[:]=[n for n in dirs if not (path/n).is_symlink() and (path/n).lstat().st_dev==b.stat().st_dev]
  for name in names:
   p=path/name;s=p.lstat()
   if stat.S_ISREG(s.st_mode):rows.append({'path':str(p.relative_to(b)),'bytes':s.st_size,'allocated_bytes':s.st_blocks*512,'links':s.st_nlink,'device':s.st_dev,'inode':s.st_ino,'mtime_ns':s.st_mtime_ns})
updates=[];p=b/'logs/actual-updates.jsonl'
if p.exists():
 for line in p.read_text().splitlines():
  r=json.loads(line);updates.append({k:r.get(k) for k in ('step','attempt_id','attempt_index','actual_optimizer_writes')})
p=b/'preparation/candidate-raw-maintenance-runner/progress.json';runner=json.loads(p.read_text()) if p.exists() else None
print(json.dumps({'role':'second_point_exact_artifacts_logs_stat_only','started_unix':started,'finished_unix':time.time(),'uid':os.getuid(),'scope':['artifacts/mixed-candidate-pools','artifacts/behavior-traces','artifacts/mixed-selection','logs'],'files':rows,'directories':directories,'metadata_errors':errors,'completed_updates':updates,'raw_runner_progress':runner,'du_executed':False,'symlink_followed':False,'tensor_contents_read':False,'other_experiment_scanned':False},sort_keys=True))
