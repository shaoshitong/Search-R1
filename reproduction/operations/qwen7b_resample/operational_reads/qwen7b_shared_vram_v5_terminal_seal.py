import hashlib,json,os,re,time
from pathlib import Path
b=Path('/mnt/data/shaoshitong/qwen7b-resample-seed20261021-g5-20261020');out=b/'preparation/shared-vram-terminal-v5';out.mkdir(parents=True,exist_ok=True);target=out/'complete.json'
if target.exists():
 v=json.loads(target.read_text());print(json.dumps(dict(status='EXISTING_V5_TERMINAL_PROOF_NOT_REHASHED',proof=str(target),proof_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),files=len(v['original_files']))));raise SystemExit(0)
RELEASE=globals()['RELEASE'];assert len(RELEASE['observations'])==2
for x in RELEASE['observations']:
 props=dict(re.findall(r'^(\w+)=(.*)$',x['unit_show'],re.M));assert props['MainPID']=='0'and props['ActiveState']=='failed'and props['ExecMainStatus']=='1'
 assert not any(x['known_own_pids_existing'].values())and not any(x['own_cgroup_members'].values())and not x['own_gpu_contexts']and x['gpu_query_returncode']==0
assert RELEASE['observations'][1]['epoch']>RELEASE['observations'][0]['epoch']
jobs=sorted((b/'pool/jobs').glob('job-shared5-1437188*/attempt-*'));assert len(jobs)==3
parent=[p for p in jobs if p.parent.name=='job-shared5-1437188'][0];bad=[p for p in jobs if p.parent.name.endswith('-engine-01')][0]
ooms=[json.loads(p.read_text())for p in bad.glob('sampler-launches/launch-*/typed-model-initialization-oom.json')]
assert len(ooms)==3 and all(x['actual_typed_cuda_oom']and x['before_first_new_sampling_request']and '520.00 MiB'in x['error']for x in ooms)
progress=json.loads((parent/'progress.json').read_text());assert progress['completed_own_batches']==0
files=set()
for job in jobs:files.update(job.rglob('*.json'));files.update(job.glob('*.log'))
files.update((b/'shared-vram/launch/native-1437188').glob('*'));rows=[]
for p in sorted(files):
 if not p.is_file():continue
 s=p.stat();h=hashlib.sha256()
 with p.open('rb')as f:
  for data in iter(lambda:f.read(8*1024**2),b''):h.update(data)
 a=p.stat();assert(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns)==(a.st_dev,a.st_ino,a.st_size,a.st_mtime_ns)
 rows.append(dict(path=str(p),device=s.st_dev,inode=s.st_ino,bytes=s.st_size,mtime_ns=s.st_mtime_ns,sha256=h.hexdigest()))
v=dict(status='OWN_NATIVE_V5_TERMINAL_ORIGINAL_LOG_SHA_AND_OWN_RELEASE_PASS',run_id='70188b8e50fd4b45bc420ce58014f46e',unit='sst-q7b-g5-sharedvram-v5-70188b8e50fd.service',original_parent_failure=json.loads((parent/'failure.json').read_text()),actual_constructor_ooms=ooms,accepted_own_batches=0,full_original_four_gpu_index_initialized=True,original_files=rows,own_release=RELEASE,foreign_tasks_signalled=False,global_goal_complete=False,epoch=time.time())
with target.open('x')as f:json.dump(v,f,indent=2);f.flush();os.fsync(f.fileno())
print(json.dumps(dict(status=v['status'],proof=str(target),proof_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),files=len(rows),bytes=sum(x['bytes']for x in rows),accepted_own_batches=0,actual_constructor_oom_attempts=3)))
