"""Private HF archives -> H100 shared storage, streaming safe extraction + fresh SHA."""
import concurrent.futures, fcntl, hashlib, json, os, re, subprocess, sys, tarfile, threading, time
from pathlib import Path
import requests

ROOT=Path('/mnt/data/shaoshitong/qwen7b-probe-assets-20261020')
REPO='Sutoonq/Qwen25-7B-Probe-H100-Transfer-20261020'
PROXY='http://10.33.0.18:3128'
REV='a09a35458c702b33eeacc393d103063234e8bc28'

def atomic(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.%d-%d.tmp'%(os.getpid(),threading.get_ident()));t.write_text(json.dumps(v,indent=2));t.replace(p)
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def main(token):
 ctl=ROOT/'data-receiver';ctl.mkdir(exist_ok=True);lock=(ctl/'receiver.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if(ROOT/'complete.json').exists():return
 local=threading.local()
 def session():
  if not hasattr(local,'s'):
   local.s=requests.Session();local.s.trust_env=False;local.s.proxies={'https':PROXY,'http':PROXY}
  return local.s
 def get_json(path):
  u='https://huggingface.co/datasets/'+REPO+'/resolve/main/'+path
  r=session().get(u,headers={'Authorization':'Bearer '+token},timeout=(15,45))
  if r.status_code==404:return None
  r.raise_for_status();return r.json()
 atomic(ctl/'controller.json',dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],repo_id=REPO,private_credentials_saved=False,source='HPC3 via private HF',target=str(ROOT),started=time.time(),GPU_used=False,mac_data_transit=False))
 plan=None
 while plan is None:
  try:plan=get_json('PLAN-SUMMARY.json')
  except requests.RequestException as e:
   status=getattr(getattr(e,'response',None),'status_code',None);atomic(ctl/'metadata-retry.json',dict(error_type=type(e).__name__,http_status=status,epoch=time.time()))
   if status in(401,403):raise
  if plan is None:time.sleep(15)
 assert plan['revision']==REV and len(plan['groups'])==34 and len(set(plan['groups']))==34
 atomic(ctl/'plan-summary.json',plan)
 def chunk(url,start,end):
  deadline=time.monotonic()+90
  with session().get(url,headers={'Authorization':'Bearer '+token,'Range':'bytes=%d-%d'%(start,end),'Accept-Encoding':'identity'},stream=True,timeout=(15,25))as r:
   r.raise_for_status();assert r.status_code==206 and r.headers['Content-Range'].startswith('bytes %d-%d/'%(start,end))
   blocks=[];n=0
   for b in r.iter_content(1024**2):
    if time.monotonic()>deadline:raise TimeoutError('Whole bounded range deadline')
    blocks.append(b);n+=len(b)
   assert n==end-start+1;return b''.join(blocks)
 def receive(name):
  assert re.fullmatch('[a-zA-Z0-9-]+',name)
  done=ctl/'verified'/(name+'.json')
  if done.exists():return json.loads(done.read_text())
  receipt=get_json('receipts/'+name+'.json')
  if receipt is None:return None
  assert receipt['group']==name and receipt['private'] and receipt['hf_lfs_sha256']==receipt['archive_sha256']
  incoming=ROOT/'incoming';incoming.mkdir(exist_ok=True);archive=incoming/(name+'.tar.zst');partial=archive.with_suffix('.zst.part');size=receipt['archive_bytes'];url='https://huggingface.co/datasets/'+REPO+'/resolve/'+receipt['hf_archive_commit']+'/'+receipt['hf_archive_path']
  fs=os.statvfs(ROOT);usable=(fs.f_bfree if os.geteuid()==0 else fs.f_bavail)*fs.f_frsize
  assert usable>sum(x['bytes']for x in receipt['files'])+2*size+32*1024**3,'Insufficient actual writable physical headroom'
  progress=ctl/'progress'/(name+'.json')
  if not archive.exists():
   offset=partial.stat().st_size if partial.exists()else 0;assert offset<=size
   with partial.open('ab')as out,concurrent.futures.ThreadPoolExecutor(max_workers=4)as pool:
    while offset<size:
     block=16*1024**2;ranges=[(s,min(s+block,size)-1)for s in range(offset,min(offset+4*block,size),block)];futures=[pool.submit(chunk,url,s,e)for s,e in ranges]
     for(s,e),f in zip(ranges,futures):
      raw=f.result();assert s==offset;out.write(raw);out.flush();offset+=len(raw);atomic(progress,dict(stage='DOWNLOADING',bytes=offset,total_bytes=size,epoch=time.time()))
    out.flush();os.fsync(out.fileno())
   assert sha(partial)==receipt['archive_sha256'];partial.replace(archive)
  else:assert archive.stat().st_size==size and sha(archive)==receipt['archive_sha256']
  expected={x['path']:x for x in receipt['files']};assert len(expected)==len(receipt['files']);dest=ROOT/'imported';dest.mkdir(exist_ok=True);seen=set();manifest=None
  atomic(progress,dict(stage='EXTRACTING_HASHED_ARCHIVE',archive_bytes=size,epoch=time.time()))
  z=subprocess.Popen(['zstd','-d','-q','-c',str(archive)],stdout=subprocess.PIPE)
  try:
   with tarfile.open(fileobj=z.stdout,mode='r|',bufsize=1024**2)as tar:
    for member in tar:
     if member.name=='MANIFEST.json':manifest=json.load(tar.extractfile(member));continue
     assert member.isfile()and member.name in expected and not member.name.startswith('/')and'..'not in Path(member.name).parts
     row=expected[member.name];assert member.size==row['bytes'];p=dest/member.name;assert p.resolve().is_relative_to(dest.resolve());p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.migration-part');h=hashlib.sha256();n=0
     with tar.extractfile(member)as source,tmp.open('wb')as out:
      for b in iter(lambda:source.read(8*1024**2),b''):out.write(b);h.update(b);n+=len(b)
      out.flush();os.fsync(out.fileno())
     assert n==row['bytes']and h.hexdigest()==row['sha256'];tmp.replace(p);seen.add(member.name)
   assert z.wait()==0
  finally:
   if z.poll()is None:z.terminate();z.wait()
  assert seen==set(expected)and manifest['group']==name and manifest['revision']==REV and manifest['files']==receipt['files']
  atomic(progress,dict(stage='FRESH_TARGET_FULL_FILE_SHA_READBACK',files=len(expected),epoch=time.time()))
  for row in receipt['files']:
   p=dest/row['path'];assert p.stat().st_size==row['bytes']and sha(p)==row['sha256']
  result=dict(status='SOURCE_HF_TARGET_ARCHIVE_AND_FRESH_PAYLOAD_SHA_PASS',group=name,source_bytes=receipt['source_bytes'],archive_bytes=size,archive_sha256=receipt['archive_sha256'],hf_archive_commit=receipt['hf_archive_commit'],files=receipt['files'],target_root=str(dest),fresh_all_payload_sha256=True,completed=time.time());atomic(done,result);atomic(progress,{k:v for k,v in result.items()if k!='files'})
  # Only this verified task staging archive; source/HF and all extracted originals stay retained.
  archive.unlink();return result
 pending=list(plan['groups']);running={};attempts={}
 with concurrent.futures.ThreadPoolExecutor(max_workers=2)as pool:
  while pending or running:
   while pending and len(running)<2:
    name=pending.pop(0);running[pool.submit(receive,name)]=name
   for f in list(running):
    if f.done():
     name=running.pop(f)
     try:
      result=f.result()
      if result is None:pending.append(name)
     except Exception as e:
      attempts[name]=attempts.get(name,0)+1;atomic(ctl/'progress'/(name+'-retry.json'),dict(attempt=attempts[name],error_type=type(e).__name__,http_status=getattr(getattr(e,'response',None),'status_code',None),epoch=time.time()))
      if attempts[name]>=12:raise
      pending.append(name)
   verified=list((ctl/'verified').glob('*.json'))if(ctl/'verified').exists()else[];proofs=[json.loads(p.read_text())for p in verified]
   atomic(ctl/'progress.json',dict(groups_verified=len(verified),groups_expected=len(plan['groups']),source_payload_bytes_verified=sum(x['source_bytes']for x in proofs),total_source_payload_bytes=plan['payload_bytes'],archive_bytes_verified=sum(x['archive_bytes']for x in proofs),active_groups=[n for f,n in running.items()if f.running()],epoch=time.time()))
   if pending or running:time.sleep(8)
 assert len(verified)==34 and sum(len(x['files'])for x in proofs)==plan['files']
 original=ROOT/'imported/source';bank=json.loads((original/'bank/complete.json').read_text());assert bank['train_questions']==169028 and bank['diagnostic_questions']==512
 assert sha(original/'bank/groups.json')==bank['groups']['sha256']and sha(original/'bank/features.json')==bank['features']['sha256']
 feature_sources=json.loads((original/'bank/features.json').read_text());old='/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019';new=[str(original/Path(p).relative_to(old))for p in feature_sources];assert len(new)==1336 and all(Path(p).is_file()for p in new)
 atomic(ROOT/'H100-PATHS.json',dict(source_base=old,imported_original_base=str(original),official_model_dir=str(ROOT/'official-qwen2.5-7b-instruct-a09a35458c70'),features_h100=new,groups=str(original/'bank/groups.json'),best_checkpoint=str(original/'probe/epoch-0002.pt'),canonical_train=str(ROOT/'imported/canonical/train.parquet'),diagnostic=str(ROOT/'imported/canonical/validation.parquet'),original_metadata_bytes_unchanged=True,training_not_started=True))
 while not(ROOT/'official-qwen2.5-7b-instruct-a09a35458c70/complete.json').exists()or not(ROOT/'code/complete.json').exists():time.sleep(15)
 model=json.loads((ROOT/'official-qwen2.5-7b-instruct-a09a35458c70/complete.json').read_text());assert model['status']=='OFFICIAL_SNAPSHOT_ALL_FILES_FRESH_SHA_VERIFIED'and model['revision']==REV
 atomic(ROOT/'complete.json',dict(status='MIGRATION_ALL_SELECTED_ASSETS_VERIFIED',revision=REV,source_run_id=bank['run_id'],groups=34,payload_files=plan['files'],payload_bytes=plan['payload_bytes'],fresh_all_files_SHA256=True,train_questions=169028,train_trajectories=845140,diagnostic_questions=512,diagnostic_trajectories=2560,unseen_questions=2048,unseen_trajectories=10240,best_checkpoint_sha256='893e2d71e9b45abb92a9573a6091a1a0fb0a5f7ffd3e4e1201d12061317095ad',official_model_bytes=model['total_bytes'],code=json.loads((ROOT/'code/complete.json').read_text()),training_started=False,mac_model_or_data_transit=False,credentials_saved=False,completed=time.time()))

if __name__=='__main__':
 token=json.loads(sys.stdin.readline())['hf_token']
 try:main(token)
 except BaseException as e:atomic(ROOT/'data-receiver/failure.json',dict(error_type=type(e).__name__,http_status=getattr(getattr(e,'response',None),'status_code',None),epoch=time.time()));raise SystemExit(1)
