from pathlib import Path
import hashlib,json,os,shutil,subprocess,sys,time
p=json.load(sys.stdin);b=Path(p['source_base']);target=b/'failed-job-730080';run=json.loads((b/'logs/run-manifest.json').read_text())
assert run['run_id']==p['source_run_id'] and run['source_commit']==p['source_commit']
r=subprocess.run(['sacct','-j','730080','--format=JobIDRaw,State,ExitCode,NodeList,End','--noheader','--parsable2'],capture_output=True,text=True,timeout=20)
assert r.returncode==0 and '730080|FAILED|1:0|' in r.stdout and '730080.extern|COMPLETED|0:0|' in r.stdout
assert json.loads((b/'logs/cleanup-status.json').read_text())['cleaned'] is True
assert not list((b/'checkpoints/search-g1-common8/actor').glob('global_step_*/complete.json'))
assert not target.exists(),'Inspect immutable existing failure evidence before retry'
target.mkdir(mode=0o770)
paths=[b/'logs'/n for n in ('run-manifest.json','native-allocation-admission.json','cleanup-status.json','actual-updates.jsonl','actual-attempts.jsonl','resolved-training-config.json','training-730080.log')]
paths += [b/'logs'/('optimizer-rank-%d.jsonl'%rank) for rank in range(4)]
paths += [b/'logs'/('scheduler-rank-%d.jsonl'%rank) for rank in range(4)]
if run.get('train_log'):paths.append(Path(run['train_log']))
records=[]
for i,path in enumerate(paths):
 if not path.is_file():continue
 name=path.name if path.parent==b/'logs' else str(i)+'-'+path.name;dst=target/name
 before=path.stat();shutil.copy2(path,dst);h=hashlib.sha256(path.read_bytes()).hexdigest();assert hashlib.sha256(dst.read_bytes()).hexdigest()==h
 after=path.stat();assert (before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns)==(after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns)
 with dst.open('rb') as f:os.fsync(f.fileno())
 records.append(dict(source=str(path),copy=str(dst),bytes=after.st_size,sha256=h))
value=dict(status='FAILED730080_IMMUTABLE_EVIDENCE_PRESERVED',failure_kind='actual CUDA OOM',source_run_id=run['run_id'],source_commit=run['source_commit'],source_job='730080',accounting=r.stdout,files=records,precondition=p,restore_available=False,fresh_recompute_required=True,completed_outer=1,last_retained_write_count_per_rank=21,checked_unix=time.time())
with (target/'failure-evidence.json').open('x') as f:json.dump(value,f,indent=2);f.write('\n');f.flush();os.fsync(f.fileno())
fd=os.open(target,os.O_RDONLY);os.fsync(fd);os.close(fd)
print(json.dumps(dict(status=value['status'],root=str(target),files=len(records),bytes=sum(x['bytes'] for x in records),receipt_sha256=hashlib.sha256((target/'failure-evidence.json').read_bytes()).hexdigest(),checked_unix=time.time())))
