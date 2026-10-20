"""Read-only source bank export, bounded independently resumable archives."""
import argparse, concurrent.futures, fcntl, hashlib, io, json, os, stat, subprocess, tarfile, threading, time
from pathlib import Path

REV='a09a35458c702b33eeacc393d103063234e8bc28'

def read(p):return json.loads(Path(p).read_text())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb')as f:
  for b in iter(lambda:f.read(8*1024**2),b''):h.update(b)
 return h.hexdigest()
def atomic(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.%d-%d.tmp'%(os.getpid(),threading.get_ident()));t.write_text(json.dumps(v,indent=2));t.replace(p)

def plan(base):
 base=Path(base);bank=read(base/'bank/complete.json')
 assert bank['run_id']=='bdbb550fd9324cefab979728acb0f5e1'and bank['model_revision']==REV and bank['train_questions']==169028 and bank['diagnostic_questions']==512
 groups=[]
 def entries():return {}
 def add(dst,p,expected=None,alias=None):
  p=Path(p);rel=alias or 'source/'+str(p.relative_to(base));s=p.stat();assert stat.S_ISREG(s.st_mode) and not p.is_symlink()
  row=dict(source=str(p),path=rel,bytes=s.st_size,expected_sha256=expected,source_stat=[s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns]);old=dst.get(rel)
  assert old is None or old==row
  dst[rel]=row
 def metadata(dst,b):
  for rel in ('config.json','bank/complete.json','bank/groups.json','bank/features.json','inputs/manifest.json'):
   p=b/rel
   if p.is_file():add(dst,p)
 def batches(dst,b,shard,paths):
  for item in paths:
   p=Path(item);v=read(p);add(dst,p);root=p.parent;inp=b/'inputs'/('shard-%02d'%shard)/(v['batch']+'.json');add(dst,inp,v['input_sha256'])
   for phase,keys in (('prefix',('states','prefix','actor_hidden')),('full',('full','own_em'))):
    for key in keys:
     x=v[phase][key];add(dst,x['path'],x['sha256'])
    for name in ('complete.json','rank-0-complete.json','scan.json'):
     q=root/'outers/outer-000001'/phase/name
     if q.is_file():add(dst,q)
   feature=read(v['prefix']['actor_hidden']['path'])
   for x in feature['files'].values():add(dst,x['path'],x['sha256'])
   for rel in ('config.json','outers/outer-000001/questions.json','outers/outer-000001/probe-score/selection.json'):
    p=root/rel
    if p.is_file():add(dst,p)
 meta=entries();metadata(meta,base)
 cp=base/'probe/epoch-0002.pt';add(meta,cp,'893e2d71e9b45abb92a9573a6091a1a0fb0a5f7ffd3e4e1201d12061317095ad')
 if (base/'probe/epoch-0002.json').is_file():add(meta,base/'probe/epoch-0002.json')
 c=read(base/'config.json')
 for key,name in [('canonical_train','train.parquet'),('validation_questions','validation.parquet')]:add(meta,c[key],c['input_sha256']['train'if key=='canonical_train'else'diagnostic'],alias='canonical/'+name)
 groups.append(dict(group='metadata-and-best-probe-cp2',files=list(meta.values())))
 for s in range(8):
  paths=read(base/('sampling/shard-%02d'%s)/'complete.json')['batches'];assert len(paths)==167
  for k in range(0,len(paths),42):
   dst=entries();batches(dst,base,s,paths[k:k+42]);groups.append(dict(group='train-shard%02d-part%02d'%(s,k//42),files=list(dst.values())))
 held=base/'selection-2048-20261020';dst=entries();metadata(dst,held)
 for s in range(4):batches(dst,held,s,read(held/('sampling/shard-%02d'%s)/'complete.json')['batches'])
 groups.append(dict(group='unseen2048-validation',files=list(dst.values())))
 allpaths=[r['path']for g in groups for r in g['files']];assert len(set(allpaths))==len(allpaths)
 return dict(status='SOURCE_EXPORT_PLAN',source_base=str(base),revision=REV,source_run_id=bank['run_id'],train_questions=169028,train_trajectories=845140,diagnostic_questions=512,diagnostic_trajectories=2560,unseen_questions=2048,unseen_trajectories=10240,groups=groups,payload_bytes=sum(r['bytes']for g in groups for r in g['files']),files=len(allpaths),code_and_official_model_exported_separately=True)

class HashReader:
 def __init__(self,p):self.f=Path(p).open('rb');self.h=hashlib.sha256();self.count=0
 def read(self,n=-1):
  b=self.f.read(n);self.h.update(b);self.count+=len(b);return b
 def close(self):self.f.close()

def main():
 a=argparse.ArgumentParser();a.add_argument('--base',required=True);a.add_argument('--out',required=True);a.add_argument('--workers',type=int,default=2);args=a.parse_args();out=Path(args.out);out.mkdir(parents=True,exist_ok=True)
 lock=(out/'export.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 if (out/'complete.json').exists():print('EXISTING_COMPLETE_NOT_REPLAYED');return
 p=out/'plan.json'
 if not p.exists():atomic(p,plan(args.base))
 spec=read(p);atomic(out/'controller.json',dict(pid=os.getpid(),start_ticks=Path('/proc/self/stat').read_text().rsplit(')',1)[1].split()[19],started=time.time(),source_unchanged=True,no_GPU=True,groups=len(spec['groups']),files=spec['files'],payload_bytes=spec['payload_bytes']))
 def pack(g):
  name=g['group'];receipt=out/(name+'.json');target=out/(name+'.tar.zst')
  if receipt.exists():
   v=read(receipt);s=target.stat();assert [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns]==v['archive_stat'];return v
  partial=out/(name+'.tar.zst.part');records=[];t=time.time()
  with partial.open('wb')as f:
   z=subprocess.Popen(['zstd','-1','-T2','-q','-c'],stdin=subprocess.PIPE,stdout=f)
   try:
    with tarfile.open(fileobj=z.stdin,mode='w|',bufsize=1024**2)as tar:
     tar.copybufsize=1024**2
     for i,r in enumerate(g['files']):
      s=Path(r['source']).stat();assert [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns]==r['source_stat']
      h=HashReader(r['source']);info=tar.gettarinfo(r['source'],arcname=r['path']);assert info.isfile()and info.size==r['bytes'];tar.addfile(info,h);h.close();digest=h.h.hexdigest();assert h.count==r['bytes'] and (r['expected_sha256']is None or digest==r['expected_sha256']),r['path']
      s=Path(r['source']).stat();assert [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns]==r['source_stat']
      records.append(dict(path=r['path'],bytes=r['bytes'],sha256=digest,source=r['source'],source_stat=r['source_stat']))
      atomic(out/(name+'-progress.json'),dict(stage='PACKING_SOURCE_HASH_VERIFIED',files=i+1,total_files=len(g['files']),source_bytes=sum(x['bytes']for x in records),archive_current_bytes=partial.stat().st_size,epoch=time.time()))
     manifest=dict(group=name,source_base=spec['source_base'],revision=REV,files=records)
     raw=json.dumps(manifest,sort_keys=True).encode();info=tarfile.TarInfo('MANIFEST.json');info.size=len(raw);tar.addfile(info,io.BytesIO(raw))
    z.stdin.close();assert z.wait()==0
   except BaseException:
    z.stdin.close();z.wait();raise
   f.flush();os.fsync(f.fileno())
  digest=sha(partial);partial.replace(target);s=target.stat();v=dict(status='SOURCE_ARCHIVE_AND_EACH_PAYLOAD_SHA_PASS',group=name,archive=str(target),archive_bytes=s.st_size,archive_sha256=digest,archive_stat=[s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns],files=records,source_bytes=sum(r['bytes']for r in records),seconds=time.time()-t,completed=time.time());atomic(receipt,v);return v
 with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers)as pool:
  futures={pool.submit(pack,g):g['group']for g in spec['groups']}
  for f in concurrent.futures.as_completed(futures):
   try:v=f.result();print(json.dumps({k:v[k]for k in ('group','archive_bytes','source_bytes','seconds')}),flush=True)
   except BaseException as e:atomic(out/'failure.json',dict(group=futures[f],error_type=type(e).__name__,epoch=time.time()));raise
 atomic(out/'complete.json',dict(status='SOURCE_ALL_ARCHIVES_AND_PAYLOAD_SHA_COMPLETE',groups=len(spec['groups']),files=spec['files'],source_bytes=spec['payload_bytes'],completed=time.time()))

if __name__=='__main__':main()
