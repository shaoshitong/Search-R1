"""Finite, read-only CPU prefetch beside one immutable native probe job.

The training process continues all of its own SHA checks. No tensor is decoded.
"""
import argparse,concurrent.futures,fcntl,hashlib,json,os,pathlib,resource,time

def sha(p):
 h=hashlib.sha256()
 with p.open('rb',buffering=0)as f:
  for x in iter(lambda:f.read(4*1024**2),b''):h.update(x)
 return h.hexdigest()
def fp(p):return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p))
def atomic(p,v):
 q=p.with_name(p.name+'.%d.tmp'%os.getpid());q.write_text(json.dumps(v,sort_keys=True)+'\n');q.replace(p)
def read(p):return json.loads(p.read_bytes())
def birth(p):
 try:return p.joinpath('stat').read_text().rsplit(')',1)[1].split()[19]
 except FileNotFoundError:return None

def run(config):
 c=read(config);b=config.parent;pid=c['main_pid'];proc=pathlib.Path('/proc/%d'%pid)
 assert os.environ['SLURM_JOB_ID']==str(c['job_id'])and os.uname().nodename==c['node'] and not os.environ.get('CUDA_VISIBLE_DEVICES')and len(os.sched_getaffinity(0))<=2
 assert c['window_bytes']==1024**3 and c['threads']==8 and c['seconds']<=600
 assert birth(proc)==c['main_birth']and c['main_base'].encode()in proc.joinpath('cmdline').read_bytes()
 main=pathlib.Path(c['main_base']);assert sha(main/'config.json')==c['main_config_sha256'];mc=read(main/'config.json');pub=read(main/'source-publication.json');assert pub['commit']==c['main_commit']
 features=pathlib.Path(mc['features']['path']);assert fp(features)==mc['features'];refs=read(features);parents={str(pathlib.Path(v['path']).parent):i for i,v in enumerate(refs)}
 guard=(b/'prefetch.lock').open('a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not(b/'complete.json').exists()
 resource.setrlimit(resource.RLIMIT_DATA,(2*1024**3,2*1024**3));deadline=time.monotonic()+c['seconds'];started=time.time();accepted=set();count=0;total=0;last_hit=time.monotonic();last_current=None;reason='FINITE_TIME_BUDGET'
 def current():
  hits=[]
  for f in proc.joinpath('fd').iterdir():
   try:s=str(f.resolve());i=parents.get(str(pathlib.Path(s).parent))
   except FileNotFoundError:continue
   if i is not None:hits.append(i)
  return max(hits)if hits else None
 def one(item):
  p=pathlib.Path(item['path']);assert str(p).startswith('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/')and p.resolve()==p and p.stat().st_size==item['bytes'];h=hashlib.sha256();n=0
  with p.open('rb',buffering=0)as f:
   for x in iter(lambda:f.read(4*1024**2),b''):
    if time.monotonic()>=deadline:raise TimeoutError('Finite prefetch deadline')
    h.update(x);n+=len(x)
  assert n==item['bytes']and h.hexdigest()==item['sha256'],'Independent prefetch full-file SHA mismatch'
  return item['path'],n
 atomic(b/'process.json',dict(pid=os.getpid(),main_pid=pid,main_birth=c['main_birth'],job_id=c['job_id'],affinity=sorted(os.sched_getaffinity(0)),started=started,config=fp(config),main_config=mc['features']))
 with concurrent.futures.ThreadPoolExecutor(max_workers=8)as pool:
  while time.monotonic()<deadline:
   if birth(proc)!=c['main_birth']:reason='MAIN_PROCESS_ENDED_OR_IDENTITY_CHANGED';break
   progress=main/'progress.json'
   if progress.exists():
    try:v=read(progress)
    except json.JSONDecodeError:v={}
    if max(v.get('cursor',0),v.get('accepted_updates',0),v.get('student_updates',0))>0:reason='MAIN_ACCEPTED_UPDATE_EXISTS';break
   i=current()
   if i is None:
    if time.monotonic()-last_hit>20:reason='MAIN_NO_LONGER_READING_FEATURE_PAYLOADS';break
    time.sleep(.5);continue
   last_hit=time.monotonic();last_current=i;items=[];window=0
   # At most 128 future manifests and 1GiB, never arbitrary look-ahead.
   for ref in refs[i+1:i+129]:
    p=pathlib.Path(ref['path']);assert fp(p)==ref;m=read(p)
    values=list(m['files'].values());size=sum(x['bytes']for x in values)
    if window+size>c['window_bytes']:break
    window+=size
    for item in values:
     if item['path']not in accepted:items.append(item)
    if len(items)>=32:break
   if not items:time.sleep(.5);continue
   try:
    for path,n in pool.map(one,items):accepted.add(path);count+=1;total+=n
   except TimeoutError:reason='FINITE_TIME_BUDGET';break
   atomic(b/'progress.json',dict(status='ACTUAL_READ_ONLY_FULL_SHA_PREFETCH_ADVANCING',files=count,bytes=total,main_manifest_index=i,main_manifest_count=len(refs),ahead_window_bytes=window,max_window_bytes=c['window_bytes'],elapsed=time.time()-started,time=time.time(),maxRSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,main_checksums_unchanged=True))
 result=dict(status='FINITE_READ_ONLY_PREFETCH_FINISHED',reason=reason,files=count,bytes=total,last_main_manifest_index=last_current,elapsed=time.time()-started,time=time.time(),maxRSS_KiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,main_checksums_unchanged=True,full_file_SHA_verified=True,actor_updates=0,probe_updates=0);atomic(b/'complete.json',result);return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--config',type=pathlib.Path,required=True);a=p.parse_args()
 try:print(json.dumps(run(a.config)),flush=True)
 except BaseException as e:atomic(a.config.parent/'failure.json',dict(type=type(e).__name__,error=str(e),time=time.time()));raise
