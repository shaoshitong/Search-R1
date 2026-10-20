import hashlib,json,os,re,time
from pathlib import Path
b=Path('/mnt/data/shaoshitong/qwen7b-resample-seed20261021-g5-20261020');out=b/'preparation/shared-vram-terminal-v3-controlled-throughput';out.mkdir(parents=True,exist_ok=True);target=out/'complete.json'
if target.exists():
 v=json.loads(target.read_text());print(json.dumps(dict(status='EXISTING_COMPLETE_TERMINAL_PROOF_NOT_REHASHED',proof=str(target),proof_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),files=len(v['original_files']))));raise SystemExit(0)
RELEASE=globals()['RELEASE'];assert len(RELEASE['observations'])==2
for x in RELEASE['observations']:
 props=dict(re.findall(r'^(\w+)=(.*)$',x['unit_show'],re.M));assert props['MainPID']=='0'and props['ActiveState']not in('active','activating','deactivating')
 assert not any(x['known_own_pids_existing'].values())and not any(x['own_cgroup_members'].values())and not x['own_gpu_contexts']and x['gpu_query_returncode']==0
assert RELEASE['observations'][1]['epoch']>RELEASE['observations'][0]['epoch']
stop=b/'preparation/shared-vram-v3-controlled-throughput-stop-v1';intent=json.loads((stop/'intent.json').read_text());receipt=json.loads((stop/'receipt.json').read_text())
assert intent['run_id']=='70188b8e50fd4b45bc420ce58014f46e'and intent['main_pid']==1259750 and intent['progress']['completed_own_batches']>=2
assert receipt['intent_sha256']==hashlib.sha256((stop/'intent.json').read_bytes()).hexdigest()and receipt['status']=='CONTROLLED_V3_STOP_OBSERVED'
assert not receipt['original_births_remaining']and not receipt['cgroup_remaining']
jobs=list((b/'pool/jobs/job-shared5-1259750').glob('attempt-*'));assert len(jobs)==1;job=jobs[0];progress=json.loads((job/'progress.json').read_text());assert progress['completed_own_batches']>=2
rows=[]
for p in sorted(list(job.rglob('*.json'))+list(job.glob('*.log'))+list((b/'shared-vram/launch/native-1259750').glob('*'))+list(stop.glob('*.json'))):
 if not p.is_file():continue
 s=p.stat();h=hashlib.sha256()
 with p.open('rb')as f:
  for data in iter(lambda:f.read(8*1024**2),b''):h.update(data)
 a=p.stat();assert(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)==(a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns)
 rows.append(dict(path=str(p),device=s.st_dev,inode=s.st_ino,bytes=s.st_size,mtime_ns=s.st_mtime_ns,sha256=h.hexdigest()))
v=dict(status='OWN_NATIVE_V3_CONTROLLED_THROUGHPUT_STOP_ORIGINAL_LOG_SHA_AND_OWN_RELEASE_PASS',run_id=intent['run_id'],unit=intent['unit'],stop_reason=intent['cause'],original_stop_intent=intent,original_stop_receipt=receipt,accepted_own_batches=progress['completed_own_batches'],original_terminal= json.loads((job/'failure.json').read_text())if(job/'failure.json').exists()else None,original_files=rows,own_release=RELEASE,foreign_tasks_signalled=False,global_goal_complete=False,epoch=time.time())
with target.open('x')as f:json.dump(v,f,indent=2);f.flush();os.fsync(f.fileno())
print(json.dumps(dict(status=v['status'],proof=str(target),proof_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),files=len(rows),bytes=sum(x['bytes']for x in rows),accepted_own_batches=v['accepted_own_batches'])))
