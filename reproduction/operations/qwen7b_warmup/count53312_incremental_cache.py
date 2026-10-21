"""Append-only frozen-E1 postblock cache; existing351360 rows are read-only.

All raw arrays remain remote CPU mmap. Only a bounded requested batch goes CUDA.
No full-bank CUDA tensor is constructed, and only added232960 rows go native.
"""
import hashlib,os,shutil,time
from pathlib import Path
from . import direct_count_large_capacity as shared
proof,checked,read,atomic,immutable,digest=shared.proof,shared.checked,shared.read,shared.atomic,shared.immutable,shared.digest


def mapped(files,n,*,create=False,dimension=3584,prefix=20):
 import torch
 spec={'hidden':(n*prefix*dimension*2,torch.bfloat16,(n,prefix,dimension)),'mask':(n*prefix,torch.bool,(n,prefix)),'logits':(n*4,torch.float32,(n,))};out={}
 for key,(size,dtype,shape)in spec.items():
  path=Path(files[key]['path']if isinstance(files[key],dict)else files[key])
  if create and not path.exists():
   with path.open('xb')as f:f.truncate(size);f.flush();os.fsync(f.fileno())
  if path.stat().st_size!=size:raise ValueError('Exact remote segment file length required')
  out[key]=torch.from_file(str(path),shared=create,size=size//(2 if key=='hidden'else 4 if key=='logits'else 1),dtype=dtype).reshape(shape)
 return out


def gather(segments,locations,qids,device):
 """Bit-exact CPU copies preserve source row order; bounded to<=32questions."""
 import torch
 if not 0<len(qids)<=32 or len(set(qids))!=len(qids):raise ValueError('At most32 unique questions per gather')
 shape=segments[0]['hidden'].shape[1:];n=len(qids)*10
 h=torch.empty((n,)+tuple(shape),dtype=torch.bfloat16);m=torch.empty((n,shape[0]),dtype=torch.bool);z=torch.empty(n,dtype=torch.float32)
 for i,q in enumerate(qids):
  segment,row=locations[q];source=segments[segment]
  if row<0 or row+10>source['hidden'].shape[0]:raise ValueError('Own G10 source rows out of range')
  for dst,key in ((h,'hidden'),(m,'mask'),(z,'logits')):dst[i*10:i*10+10].copy_(source[key][row:row+10])
 return h.to(device),m.to(device),z.to(device)


def build(c,source0,head,lookup,layout,base,device,stop):
 import torch
 from . import direct_hidden_count as d,probe_ablation_train as u
 from .model import Probe
 from .replay_feature_reader import ReplayFeatureReader
 base=Path(base);config_sha=proof(base/'config.json')['sha256'];old=read(checked(c['baseline_cache']));old_tids=[t for q in layout['old_qids']for t in lookup[q]['trajectory_ids']]
 if old['trajectory_ids_sha256']!=digest(old_tids)or old['frozen_legacy_before']!=source0['frozen_legacy_hashes']or old['frozen_legacy_after']!=source0['frozen_legacy_hashes']or old['legacy_updates']!=0 or old['actor_updates']!=0:raise ValueError('Accepted original cache/source identities differ')
 for key,ref in old['files'].items():
  atomic(base/'progress.json',dict(stage='FRESH_SHA_VERIFY_OLD_READ_ONLY_NATIVE_CACHE',file=key,bytes=ref['bytes'],time=time.time()))
  checked(ref)
 oldmaps=mapped(old['files'],len(old_tids));new_qids=layout['added_qids'];tids=[t for q in new_qids for t in lookup[q]['trajectory_ids']];n=len(tids)
 if len(new_qids)!=23296 or n!=232960 or set(old_tids)&set(tids):raise ValueError('Only23296 additional real G10 questions may go native')
 root=base/'added-cache';root.mkdir(exist_ok=True);files={k:root/v for k,v in {'hidden':'hidden.bf16','mask':'mask.bool','logits':'logits.f32'}.items()};complete=root/'complete.json'
 if not complete.exists()and shutil.disk_usage(base).free<n*(20*3584*2+20+4)+8*1024**3:raise ValueError('Insufficient remote disk for added cache; preserve original cache')
 newmaps=mapped(files,n,create=not complete.exists());frozen=source0['frozen_legacy_hashes']
 if not complete.exists():
  # Only this cache phase may load the original E1 block; it creates no Adam.
  source=torch.load(checked(c['source_checkpoint']),map_location='cpu',mmap=True,weights_only=False)
  if source['steps']!=10464 or source['epoch']!=1 or not source['completed_epoch']:raise ValueError('Exact frozen original E1 required')
  legacy=Probe(c['model_dir'],273,20).to(device);legacy.load_state_dict(source['raw'],strict=True);u.restore_buffers(legacy,source['raw_buffers']);legacy.eval().requires_grad_(False)
  if dict(raw=u.model_hash(legacy),buffers=u.tensor_hash(u.buffers(legacy)))!=frozen:raise ValueError('Native new-row source differs from accepted old cache E1')
  if len(source['rank_states'])!=4 or any(x['model_sha256']!=frozen['raw']for x in source['rank_states']):raise ValueError('Original four-rank E1 archive differs')
  del source
  refs=read(checked(c['inputs']['merged_features_layer26']));need=set(tids);selected=[]
  for ref in refs:
   manifest=read(checked(ref));index=Path(manifest['files']['index.jsonl']['path'])
   with index.open()as f:hit=any(__import__('json').loads(line)['trajectory_id']in need for line in f)
   if hit:selected.append(ref)
  reader=ReplayFeatureReader(selected,26)
  if not need<=set(reader.records):raise ValueError('New true own trajectories missing from accepted feature source')
  class Recorder:
   def encode(self,h,m):self.hidden=h.detach();self.mask=m.detach();return head.encode(h,m)
  recorder=Recorder();cursor=0;chunks=[];progress=root/'progress.json'
  if progress.exists():
   prior=read(progress)
   if prior['config_sha256']!=config_sha or prior['trajectory_ids_sha256']!=digest(tids):raise ValueError('Partial new-cache identity changed')
   cursor=prior['completed_trajectories'];chunks=prior['completed_chunks'];expected=0
   for ref in chunks:
    item=read(checked(ref))
    if item['start']!=expected or item['trajectory_ids_sha256']!=digest(tids[item['start']:item['end']]):raise ValueError('Partial cache ordered prefix differs')
    for key,span in item['spans'].items():
     with files[key].open('rb')as f:f.seek(span['offset']);raw=f.read(span['bytes'])
     if hashlib.sha256(raw).hexdigest()!=span['sha256']:raise ValueError('Accepted partial cache bytes changed; do not reset silently')
    expected=item['end']
   if expected!=cursor:raise ValueError('Partial cache cursor has no complete contiguous chunks')
  started=time.time();golden=(root/'same-api-golden.json').exists()
  for start in range(cursor,n,256):
   end=min(n,start+256);contexts=[]
   for offset in range(start,end,8):
    ids=tids[offset:min(end,offset+8)];batch=reader.get(ids,device)
    with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):z,e=d.native_hidden_forward(legacy,batch,recorder)
    if recorder.hidden.dtype!=torch.bfloat16:raise ValueError('Native postblock hidden must remain BF16')
    newmaps['hidden'][offset:offset+len(ids)].copy_(recorder.hidden.cpu());newmaps['mask'][offset:offset+len(ids)].copy_(recorder.mask.cpu());newmaps['logits'][offset:offset+len(ids)].copy_(z.cpu())
    if not golden:
     h0=recorder.hidden.clone();m0=recorder.mask.clone();z0=z.clone();e0=e.clone()
     with torch.no_grad(),torch.autocast('cuda',dtype=torch.bfloat16):z1,e1=d.native_hidden_forward(legacy,batch,recorder)
     if not all((torch.equal(h0,recorder.hidden),torch.equal(m0,recorder.mask),torch.equal(z0,z1),torch.equal(e0,e1),torch.equal(newmaps['hidden'][offset:offset+len(ids)].to(device),h0),torch.equal(newmaps['mask'][offset:offset+len(ids)].to(device),m0))):raise ValueError('Actual same-shape new cache/native replay differs')
     immutable(root/'same-api-golden.json',dict(status='PASSED',IDs_sha256=digest(ids),mask_positions_sha256=u.tensor_hash({k:batch[k]for k in ('attention_mask','position_ids')}),dtype='bfloat16',micro=8,old_cached_trajectory_rows_recomputed=0));golden=True
    contexts.append(dict(ids_sha256=digest(ids),mask_positions_sha256=u.tensor_hash({k:batch[k]for k in ('attention_mask','position_ids')})));del batch,z,e
   spans={}
   for key,itemsize,width in [('hidden',2,20*3584),('mask',1,20),('logits',4,1)]:
    with files[key].open('rb')as f:os.fsync(f.fileno());f.seek(start*width*itemsize);raw=f.read((end-start)*width*itemsize)
    spans[key]=dict(offset=start*width*itemsize,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
   item=dict(start=start,end=end,trajectory_ids_sha256=digest(tids[start:end]),spans=spans,contexts_sha256=digest(contexts));p=root/('chunk-%06d.json'%len(chunks));immutable(p,item);chunks.append(proof(p))
   atomic(progress,dict(config_sha256=config_sha,trajectory_ids_sha256=digest(tids),completed_trajectories=end,completed_chunks=chunks,time=time.time()));atomic(base/'progress.json',dict(stage='ONLY_ADDED_FROZEN_E1_NATIVE_CACHE_ADVANCING',trajectories=end,total=n,old_rows_recomputed=0,elapsed=time.time()-started,time=time.time()))
   if stop():reader.close();return None
  reader.close();del reader,recorder
  if dict(raw=u.model_hash(legacy),buffers=u.tensor_hash(u.buffers(legacy)))!=frozen:raise ValueError('Frozen E1 modified during added cache generation')
  del legacy;torch.cuda.empty_cache()
  immutable(complete,dict(status='ONLY_ADDED23296_FROZEN_E1_NATIVE_CACHE_COMPLETE',files={k:proof(p)for k,p in files.items()},chunks=chunks,config_sha256=config_sha,trajectory_ids_sha256=digest(tids),old_rows_recomputed=0,added_questions=23296,added_trajectories=n,frozen_legacy_before=frozen,frozen_legacy_after=frozen,same_api_golden=proof(root/'same-api-golden.json'),actual_native_hidden_dtype='bfloat16',legacy_updates=0,actor_updates=0))
 added=read(complete)
 if added['config_sha256']!=config_sha or added['trajectory_ids_sha256']!=digest(tids)or added['frozen_legacy_before']!=frozen or added['frozen_legacy_after']!=frozen or added['old_rows_recomputed']!=0:raise ValueError('New-cache exact lineage differs')
 for ref in added['files'].values():checked(ref)
 composite=dict(status='COUNT53312_CPU_MMAP_TWO_SEGMENT_NATIVE_CACHE_COMPLETE',old_cache=c['baseline_cache'],added_cache=proof(complete),old_rows_recomputed=0,old_trajectories=len(old_tids),added_trajectories=n,total_trajectories=len(old_tids)+n,layout_sha256=digest(layout),frozen_legacy_hashes=frozen,full_bank_CUDA_resident=False,batch_max_trajectories=320)
 immutable(base/'cache-complete.json',composite)
 return [oldmaps,newmaps],proof(base/'cache-complete.json')
