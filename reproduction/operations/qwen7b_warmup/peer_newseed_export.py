"""CPU-only immutable export of genuine sealed peer new-seed Prefix20 G5 batches.

No GPU/model/tensor/HF dependency. Producer files/services are read-only. The
byte-identical peer CPU auditor writes only into the new snapshot's own proof
cache; a producer proof may be copied in, then is accepted only if every original
fstat binding still matches. All copied payload bytes receive fresh SHA anyway.
"""
import argparse,fcntl,gzip,hashlib,importlib.util,json,os,re,stat,sys,tarfile,time,uuid
from pathlib import Path,PurePosixPath
from types import SimpleNamespace

RUN_ID='70188b8e50fd4b45bc420ce58014f46e'
SEED=20261021
ATTEMPT='independent-seed20261021-g5-20261020'
REVISION='a09a35458c702b33eeacc393d103063234e8bc28'
SOURCE='/mnt/data/shaoshitong/qwen7b-resample-seed20261021-g5-20261020'
RECORDED='/mnt/shared/workspace/training_data_node_10/shaoshitong/qwen7b-resample-seed20261021-g5-20261020'
OUTPUT='/mnt/data/shaoshitong/qwen7b-newseed-to-hpc3-20261021'
AUDIT_SHA='3def055adebfdb38e9326b9db385338c44800160c1b0a64e1ee78d8c3b68b906'
STATUS='SEALED_CANONICAL_NEWSEED_PREFIX20_G5_SNAPSHOT_EXPORT_COMPLETE'
FORBIDDEN=('.bin','.pt','.pth','.safetensors','.npy','.npz','.pkl','.pickle','.ckpt')


def canonical(v):return json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()
def read(p):return json.loads(Path(p).read_bytes())
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for raw in iter(lambda:f.read(1024*1024),b''):h.update(raw)
 return h.hexdigest()
def fs(p):
 s=Path(p).lstat();return dict(bytes=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns,inode=s.st_ino,device=s.st_dev,mode=s.st_mode)
def fp(p):
 p=Path(p);return dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p),source_stat=fs(p))
def atomic(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_name(p.name+'.tmp-'+uuid.uuid4().hex)
 with tmp.open('xb') as f:f.write(canonical(v)+b'\n');f.flush();os.fsync(f.fileno())
 os.replace(str(tmp),str(p))
def immutable(p,v):
 p=Path(p)
 if p.exists():
  if read(p)!=v:raise ValueError('Conflicting immutable snapshot artifact: '+str(p))
 else:atomic(p,v)

def safe_name(name):
 p=PurePosixPath(name)
 if not name or p.is_absolute() or '..' in p.parts or '.' in p.parts or '\\' in name or any(x in name for x in ('\x00','\r','\n')):raise ValueError('Unsafe relative tar/snapshot path')
 if str(p)!=name:raise ValueError('Noncanonical relative path')
 return name

def regular(root,path):
 root,path=Path(root),Path(path)
 try:relative=path.relative_to(root)
 except ValueError:raise ValueError('Source file escapes approved base')
 if '..' in relative.parts:raise ValueError('Source traversal is forbidden')
 current=root
 if current.is_symlink():raise ValueError('Symlink source root forbidden')
 for part in relative.parts:
  current=current/part
  if current.is_symlink():raise ValueError('Symlink source component forbidden')
 if not stat.S_ISREG(path.lstat().st_mode):raise ValueError('Only regular source files are exportable')
 if path.suffix.lower() in FORBIDDEN:raise ValueError('Tensor/model artifacts cannot enter this snapshot')
 return path

def mapped(base,value):
 base=Path(base);value=str(value)
 for prefix in (RECORDED,SOURCE,str(base)):
  if value==prefix or value.startswith(prefix+'/'):
   candidate=base/value[len(prefix):].lstrip('/');regular(base,candidate);return candidate
 raise ValueError('Recorded source path is outside the approved source base')

def contract(c):
 expected=dict(run_id=RUN_ID,seed=SEED,model_revision=REVISION,sampling_attempt_id=ATTEMPT,warmup_only=True,actor_hidden_native=False,actor_optimizer_updates=0)
 if any(c.get(k)!=v for k,v in expected.items()):raise ValueError('Requires the genuine peer NEW-seed official7B G5 source, never old HPC3 lite data')

def discover(base):
 """Read-only small seal metadata; no payload parsing and no source writes."""
 base=Path(base)
 if base.resolve()!=base or base.is_symlink():raise ValueError('Canonical physical source path required')
 c=read(regular(base,base/'config.json'));contract(c)
 manifest=read(regular(base,base/'inputs/manifest.json'));declared={};sealed=[];unsealed=[]
 for item in manifest['batches']:
  inp=mapped(base,item['path']);rel=str(inp.relative_to(base));safe_name(rel)
  if rel in declared:raise ValueError('Duplicate canonical input batch')
  declared[rel]=item;seal=base/'sampling'/inp.parent.name/'batches'/inp.stem/'complete.json'
  if not seal.exists():unsealed.append(rel);continue
  regular(base,seal);v=read(seal)
  expected=dict(run_id=RUN_ID,input_sha256=item['sha256'],batch=inp.stem,questions=item['questions'],trajectories=5*item['questions'],hidden_cached=False,model_revision=REVISION,actor_optimizer_updates=0)
  if any(v.get(k)!=x for k,x in expected.items()):raise ValueError('Claimed sealed batch metadata differs: '+str(seal))
  sealed.append(dict(input=rel,input_sha256=item['sha256'],input_stat=fs(inp),seal=str(seal.relative_to(base)),seal_sha256=sha(seal),seal_stat=fs(seal),questions=v['questions'],trajectories=v['trajectories']))
 actual={str(p.relative_to(base)) for p in (base/'inputs').glob('shard-*/batch*.json')}
 if actual!=set(declared):raise ValueError('Input manifest/files membership differs')
 known={str((base/'sampling'/Path(r).parent.name/'batches'/Path(r).stem/'complete.json').relative_to(base)) for r in declared}
 unknown={str(p.relative_to(base)) for p in (base/'sampling').glob('shard-*/batches/batch*/complete.json')}-known
 if unknown:raise ValueError('Sealed source batch absent from canonical input manifest')
 sealed.sort(key=lambda x:x['input'])
 return dict(source_base=str(base),source_node='h100-10',source_run_id=RUN_ID,seed=SEED,revision=REVISION,source_config=fp(base/'config.json'),input_manifest=fp(base/'inputs/manifest.json'),
  selected_batches=sealed,unsealed_inputs_excluded=sorted(unsealed),source_input_batches=len(declared),source_all_inputs_sealed=not unsealed,
  questions=sum(x['questions'] for x in sealed),trajectories=sum(x['trajectories'] for x in sealed),cohort_seal_digest=hashlib.sha256(canonical(sealed)).hexdigest(),
  discovery_verified_payloads=False,selection_rule='all canonical batches with an already-present valid producer complete.json at snapshot freeze; no label/probe selection')

def summary(plan):
 result={k:plan[k] for k in ('source_base','source_node','source_run_id','seed','revision','source_input_batches','source_all_inputs_sealed','questions','trajectories','cohort_seal_digest','discovery_verified_payloads','selection_rule')}
 result.update(claimed_sealed_batches=len(plan['selected_batches']),unsealed_batches_excluded=len(plan['unsealed_inputs_excluded']));return result


def load_auditor(base,snapshot):
 plan=read(Path(snapshot)/'plan.json')
 if any(x.get('audit_dispatch')for x in plan['selected_batches']):
  from .peer_newseed_source_audit_dispatch import Dispatch
  return Dispatch(base,snapshot,plan)
 path=Path(__file__).with_name('peer_newseed_audit_reference.py')
 if sha(path)!=AUDIT_SHA:raise ValueError('Exact peer standalone CPU auditor is required')
 spec=importlib.util.spec_from_file_location('pinned_peer_newseed_cpu_auditor',path);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
 auditor=module.Audit(SimpleNamespace(base=str(base),audit_dir=str(snapshot/'audit-proof-cache'),path_map=[],max_new_batches=0))
 auditor.root_config=read(Path(base)/'config.json');auditor.root_config_hash=sha(Path(base)/'config.json')
 return auditor

def batch_files(base,inp):
 dest=Path(base)/'sampling'/inp.parent.name/'batches'/inp.stem;outer=dest/'outers/outer-000001'
 return dict(input=inp,batch_seal=dest/'complete.json',config=dest/'config.json',questions=outer/'questions.json',prefix_seal=outer/'prefix/complete.json',
  prefix_scan=outer/'prefix/scan.json',prefix_rank=outer/'prefix/rank-0-complete.json',states=outer/'prefix/rank-0-states.jsonl.gz',prefix=outer/'prefix/rank-0-prefix.jsonl.gz',
  full_seal=outer/'full/complete.json',full_rank=outer/'full/rank-0-complete.json',full=outer/'full/rank-0-full.jsonl.gz',own_em=outer/'full/rank-0-own-em.jsonl',selection=outer/'probe-score/selection.json')

def own_groups(inp,batch,files):
 """Exact own G5 identity/EM correspondence; never substitute another qid."""
 by={row['qid']:row for row in batch['rows']};labels={}
 if len(by)!=len(batch['rows']):raise ValueError('Duplicate input question within source batch')
 with files['own_em'].open() as f:
  for line in f:
   row=json.loads(line);tid=row['trajectory_id'];qid=row['qid'];label=row['strict_em']
   if qid not in by or tid in labels or type(label) is not int or label not in (0,1):raise ValueError('Invalid/duplicate/foreign own-EM identity')
   labels[tid]=(qid,label)
 groups=[]
 for qid,row in by.items():
  tids=['%s-outer000001-c%04d-g%d'%(RUN_ID,row['candidate_ordinal'],j) for j in range(5)]
  if any(t not in labels or labels[t][0]!=qid for t in tids):raise ValueError('Same question must have its own complete five true labels')
  groups.append(dict(qid=qid,question_hash=row['question_hash'],candidate_ordinal=row['candidate_ordinal'],source_ordinal=row['source_ordinal'],trajectory_ids=tids,labels=[labels[t][1] for t in tids],source_input=str(inp)))
 if len(labels)!=5*len(groups):raise ValueError('Extra own labels outside canonical G5')
 return groups

def stable_copy(base,source,dest,expected=None):
 source=regular(base,source);before=fs(source);dest=Path(dest);dest.parent.mkdir(parents=True,exist_ok=True)
 flags=os.O_RDONLY|getattr(os,'O_NOFOLLOW',0);fd=os.open(str(source),flags)
 temporary=dest.with_name(dest.name+'.partial-'+uuid.uuid4().hex);h=hashlib.sha256();size=0
 try:
  with os.fdopen(fd,'rb') as inp,temporary.open('xb') as out:
   s=os.fstat(inp.fileno())
   if (s.st_ino,s.st_dev,s.st_size)!=(before['inode'],before['device'],before['bytes']):raise ValueError('Source identity raced before copy')
   for raw in iter(lambda:inp.read(1024*1024),b''):h.update(raw);size+=len(raw);out.write(raw)
   out.flush();os.fsync(out.fileno());after_fd=os.fstat(inp.fileno())
  after=fs(source)
  if before!=after or (after_fd.st_size,after_fd.st_mtime_ns,after_fd.st_ctime_ns)!=(before['bytes'],before['mtime_ns'],before['ctime_ns']):raise ValueError('Sealed source changed during export copy')
  value=dict(bytes=size,sha256=h.hexdigest())
  if expected and (value['bytes']!=expected['bytes'] or value['sha256']!=expected['sha256']):raise ValueError('Copied source differs from producer SHA/size')
  if dest.exists():
   regular(dest.parent,dest)
   if dest.stat().st_size!=size or sha(dest)!=value['sha256']:raise ValueError('Different existing immutable snapshot file')
  else:os.link(str(temporary),str(dest));dest.chmod(0o444)
  temporary.unlink()
  return dict(value,source_path=str(source),source_stat=before)
 except BaseException:
  # Owned partial files remain outside every accepted tar membership list.
  raise


def tar_group(root,paths,archive,prefix):
 """Only explicitly listed regular files; deterministic, unique safe prefix."""
 root,archive=Path(root),Path(archive);safe_name(prefix);members={}
 for relative,item in paths.items():
  safe_name(relative);p=regular(root,root/relative)
  if p.stat().st_size!=item['bytes'] or sha(p)!=item['sha256']:raise ValueError('Snapshot payload SHA differs before tar')
  members[prefix+'/'+relative]=item
 if len(members)!=len(paths):raise ValueError('Duplicate tar members')
 temporary=archive.with_name(archive.name+'.partial-'+uuid.uuid4().hex);archive.parent.mkdir(parents=True,exist_ok=True)
 with tarfile.open(temporary,'w',format=tarfile.PAX_FORMAT) as tar:
  for name,item in sorted(members.items()):
   relative=name[len(prefix)+1:];p=regular(root,root/relative);before=fs(p)
   info=tarfile.TarInfo(name);info.size=item['bytes'];info.mode=0o444;info.mtime=0;info.uid=info.gid=0
   fd=os.open(str(p),os.O_RDONLY|getattr(os,'O_NOFOLLOW',0))
   with os.fdopen(fd,'rb') as f:tar.addfile(info,f)
   if fs(p)!=before:raise ValueError('Snapshot payload changed while archiving')
 with temporary.open('rb') as f:os.fsync(f.fileno())
 with tarfile.open(temporary,'r:') as tar:
  found=tar.getmembers()
  if len(found)!=len(members) or {x.name for x in found}!=set(members):raise ValueError('Tar membership differs')
  for info in found:
   safe_name(info.name)
   if not info.isfile() or info.issym() or info.islnk():raise ValueError('Nonregular tar member forbidden')
   h=hashlib.sha256();size=0
   with tar.extractfile(info) as f:
    for raw in iter(lambda:f.read(1024*1024),b''):h.update(raw);size+=len(raw)
   if (size,h.hexdigest())!=(members[info.name]['bytes'],members[info.name]['sha256']):raise ValueError('Tar payload SHA differs')
 result=dict(archive=str(archive),archive_bytes=temporary.stat().st_size,archive_sha256=sha(temporary))
 if archive.exists():
  regular(archive.parent,archive)
  if archive.stat().st_size!=result['archive_bytes'] or sha(archive)!=result['archive_sha256']:raise ValueError('Different existing immutable archive')
 else:os.link(str(temporary),str(archive));archive.chmod(0o444)
 temporary.unlink();return result


def verify_frozen_source(base,plan):
 for key in ('source_config','input_manifest'):
  proof=plan[key];p=regular(base,Path(proof['path']))
  if fs(p)!=proof['source_stat'] or sha(p)!=proof['sha256']:raise ValueError('Frozen global source changed')
 for item in plan['selected_batches']:
  inp=regular(base,Path(base)/item['input']);seal=regular(base,Path(base)/item['seal'])
  if fs(inp)!=item['input_stat'] or fs(seal)!=item['seal_stat'] or sha(seal)!=item['seal_sha256']:raise ValueError('Frozen selected batch/input/seal changed')


def export(base,output_root,snapshot_id,*,batches_per_archive=16):
 if not re.fullmatch(r'[A-Za-z0-9_-]{8,80}',snapshot_id) or '..' in snapshot_id:raise ValueError('Unique safe explicit snapshot-id required')
 if not 16<=batches_per_archive<=32:raise ValueError('Use16..32 complete source batches per archive group')
 base,output_root=Path(base),Path(output_root);root=output_root/snapshot_id
 if output_root.resolve()!=output_root or output_root.is_symlink() or base.resolve()!=base:raise ValueError('Physical canonical source/output roots required')
 if root==base or str(root).startswith(str(base)+'/'):raise ValueError('New independent snapshot root required')
 if root.exists() and root.is_symlink():raise ValueError('Snapshot symlink forbidden')
 root.mkdir(parents=True,exist_ok=True)
 with (root/'export.lock').open('a') as lock:
  fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
  if (root/'complete.json').exists():
   done=read(root/'complete.json')
   if done['snapshot_id']!=snapshot_id or done['source_run_id']!=RUN_ID:raise ValueError('Existing snapshot identity differs')
   for group in done['groups']:
    p=regular(root,Path(group['archive']))
    if p.stat().st_size!=group['archive_bytes'] or sha(p)!=group['archive_sha256']:raise ValueError('Accepted archive changed')
   return done
  if (root/'plan.json').exists():plan=read(root/'plan.json')
  else:
   if any(p.name not in ('export.lock',) for p in root.iterdir()):raise ValueError('Unowned partial snapshot requires reconciliation')
   plan=discover(base)
   if not plan['selected_batches']:raise ValueError('No genuine sealed batch available')
   plan.update(snapshot_id=snapshot_id,batches_per_archive=batches_per_archive,exporter_sha256=sha(__file__),auditor_sha256=AUDIT_SHA)
   immutable(root/'plan.json',plan)
  if plan['snapshot_id']!=snapshot_id or plan['source_base']!=str(base) or plan['batches_per_archive']!=batches_per_archive or plan['exporter_sha256']!=sha(__file__):raise ValueError('Frozen snapshot/exporter/selection identity differs')
  verify_frozen_source(base,plan);auditor=load_auditor(base,root);admitted={};seen_tids=set();validated=[];source_stats={};deduplicated=0
  atomic(root/'controller.json',dict(status='VALIDATING_FROZEN_SEALED_SOURCE_BATCHES',pid=os.getpid(),snapshot_id=snapshot_id,source_run_id=RUN_ID,epoch=time.time()))
  try:
   for number,item in enumerate(plan['selected_batches']):
    inp=regular(base,base/item['input']);batch=read(inp)
    if sha(inp)!=item['input_sha256']:raise ValueError('Canonical input SHA changed')
    files={key:regular(base,path) for key,path in batch_files(base,inp).items()}
    before={str(p):fs(p) for p in files.values()}
    cache=base/'audit/batches'/inp.parent.name/(inp.stem+'.json');dest=root/'audit-proof-cache/batches'/inp.parent.name/(inp.stem+'.json')
    if cache.is_file() and not dest.exists() and not item.get('audit_dispatch'):stable_copy(base,cache,dest)
    proof=auditor.batch(inp,batch,item['input_sha256'])
    if not proof or proof.get('deferred'):raise ValueError('Frozen batch is not genuinely fully sealed/verified')
    if before!={str(p):fs(p) for p in files.values()}:raise ValueError('Source files changed during real data audit')
    groups=own_groups(inp,batch,files);new=[]
    for g in groups:
     qid=g['qid'];signature={k:g[k] for k in ('qid','question_hash','candidate_ordinal','source_ordinal','trajectory_ids','labels')}
     if qid in admitted:
      if admitted[qid]!=signature:raise ValueError('Conflicting duplicate question/own G5; never silently replace source')
      deduplicated+=1;continue
     if seen_tids.intersection(g['trajectory_ids']):raise ValueError('Real trajectory ID reused by another question')
     admitted[qid]=signature;seen_tids.update(g['trajectory_ids']);new.append(signature)
    source_stats.update(before);validated.append(dict(input=item['input'],files=files,new_groups=new,source_questions=len(groups),audit_dispatch=item.get('audit_dispatch')))
    atomic(root/'controller.json',dict(status='FROZEN_SOURCE_BATCHES_ADVANCING',validated_batches=number+1,total_batches=len(plan['selected_batches']),questions=len(admitted),trajectories=len(seen_tids),epoch=time.time()))
   if len({g['source_ordinal'] for g in admitted.values()})!=len(admitted) or len({g['candidate_ordinal'] for g in admitted.values()})!=len(admitted):raise ValueError('Canonical source/candidate ordinal reused by distinct question')
   ordered=sorted(admitted.values(),key=lambda g:g['source_ordinal'])
   immutable(root/'canonical-membership.json',dict(groups=ordered,questions=len(ordered),trajectories=5*len(ordered),source_run_id=RUN_ID,seed=SEED))
   groups_out=[];common=[base/'config.json',base/'inputs/manifest.json']
   # Copy/archive fixed subsets; archive members are unique snapshot/group paths.
   for start in range(0,len(validated),batches_per_archive):
    group_number=start//batches_per_archive;name='group-%05d'%group_number;payload=root/'payload'/name;paths={};membership=[];source_bytes=0;metadata_resealed_audits={}
    for source in common:
     relative='source/'+str(source.relative_to(base));value=stable_copy(base,source,payload/relative);paths[relative]=value;source_stats[str(source)]=value['source_stat'];source_bytes+=value['bytes']
    for value in validated[start:start+batches_per_archive]:
     membership.extend(value['new_groups'])
     rank=read(value['files']['prefix_rank']);fullrank=read(value['files']['full_rank']);producer={'states':rank['states'],'prefix':rank['prefix'],'full':fullrank['full'],'own_em':fullrank['own_em']}
     for key,source in value['files'].items():
      relative='source/'+str(source.relative_to(base))
      if relative in paths:continue
      expected=producer.get(key)
      if expected and mapped(base,expected['path'])!=source:raise ValueError('Producer artifact path differs from sealed batch role')
      if fs(source)!=source_stats[str(source)]:raise ValueError('Sealed source changed between validation and copy')
      copied=stable_copy(base,source,payload/relative,expected);paths[relative]=copied;source_bytes+=copied['bytes']
    for value in validated[start:start+batches_per_archive]:
     info=value.get('audit_dispatch')
     if not info or not info.get('metadata_resealed_attestation'):continue
     inp=Path(value['input']);mapping=dict(auditor_sha256=info['auditor_sha256'],original_receipt_chain_equal_current=False)
     for role,key in (('proof','proof'),('attestation','metadata_resealed_attestation')):
      ref=info[key];src=Path(ref['path']);relative='provenance/audit/'+inp.parent.name+'/'+inp.stem+'-'+role+'.json';copied=stable_copy(src.parent,src,payload/relative,ref);paths[relative]=copied;mapping[role]=dict(path=relative,bytes=copied['bytes'],sha256=copied['sha256'])
     metadata_resealed_audits[value['input']]=mapping
    cohort=dict(source_run_id=RUN_ID,seed=SEED,sampling_attempt_id=ATTEMPT,revision=REVISION,prefix_tokens=20,groups=membership,questions=len(membership),trajectories=5*len(membership),source_base=str(base),recorded_base=RECORDED,
     metadata_resealed_audits=metadata_resealed_audits,no_label_selection=True,duplicate_policy='deduplicate only exact same question/hash/ordinal/own-five IDs and labels; conflicting duplicate fails closed',strict_em_recomputed=False,own_em_full_correspondence_verified=True,hidden_exported=False)
    immutable(payload/'cohort.json',cohort);paths['cohort.json']=dict(bytes=(payload/'cohort.json').stat().st_size,sha256=sha(payload/'cohort.json'))
    if group_number==0:
     for own_name in ('plan.json','canonical-membership.json'):
      relative='provenance/'+own_name;copied=stable_copy(root,root/own_name,payload/relative);paths[relative]=copied
    prefix=snapshot_id+'/'+name;archive=root/'archives'/(snapshot_id+'-'+name+'.tar');packed=tar_group(payload,paths,archive,prefix)
    record=dict(group=name,archive_format='tar',**packed,files=[dict(path=prefix+'/'+relative,bytes=v['bytes'],sha256=v['sha256']) for relative,v in sorted(paths.items())],source_bytes=sum(v['bytes'] for v in paths.values()),copied_source_bytes=source_bytes,questions=len(membership),trajectories=5*len(membership),source_batches=len(validated[start:start+batches_per_archive]),cohort_sha256=sha(payload/'cohort.json'))
    groups_out.append(record);atomic(root/'controller.json',dict(status='SNAPSHOT_ARCHIVE_GROUP_ADVANCING',completed_groups=len(groups_out),questions=sum(g['questions'] for g in groups_out),epoch=time.time()))
   verify_frozen_source(base,plan)
   for path,before in source_stats.items():
    if fs(regular(base,Path(path)))!=before:raise ValueError('Source changed before full snapshot acceptance')
   ordered=sorted(admitted.values(),key=lambda g:g['source_ordinal'])
   immutable(root/'canonical-membership.json',dict(groups=ordered,questions=len(ordered),trajectories=5*len(ordered),source_run_id=RUN_ID,seed=SEED))
   result=dict(status=STATUS,snapshot_id=snapshot_id,source_base=str(base),source_node='h100-10',source_run_id=RUN_ID,seed=SEED,sampling_attempt_id=ATTEMPT,groups=groups_out,revision=REVISION,questions=len(admitted),trajectories=len(seen_tids),files=sum(len(g['files']) for g in groups_out),payload_bytes=sum(sum(f['bytes'] for f in g['files']) for g in groups_out),source_bytes=sum(g['source_bytes'] for g in groups_out),
    raw_source_questions=sum(x['source_questions'] for x in validated),raw_duplicate_payloads_retained=deduplicated>0,source_input_batches=plan['source_input_batches'],frozen_sealed_batches=len(validated),source_all_inputs_sealed_at_freeze=plan['source_all_inputs_sealed'],unsealed_batches_excluded=len(plan['unsealed_inputs_excluded']),duplicates_removed=deduplicated,cohort_seal_digest=plan['cohort_seal_digest'],ordered_qid_sha256=hashlib.sha256(canonical([g['qid'] for g in ordered])).hexdigest(),canonical_membership=fp(root/'canonical-membership.json'),
    fresh_all_source_files_SHA256=True,all_copied_files_fresh_SHA_verified=True,all_archives_fresh_SHA_and_member_SHA_verified=True,source_full_fstat_pre_post_verified=True,producer_sources_modified=False,source_assets_deleted=False,hidden_exported=False,model_tensors_exported=False,actor_updates=0,probe_updates=0,strict_em_recomputed=False,strict_em_verification='own-em equals exact full trajectory labels; pinned peer auditor validates complete turn/chunk/prefix/seed provenance; tokenizer-based strict EM is not rerun',states_shards_deep_audited=False,states_payload_SHA_verified=True,auditor_sha256=AUDIT_SHA,deep_audit_cache_reused_batches=auditor.reused,deep_audit_fresh_batches=auditor.fresh,exporter_sha256=sha(__file__),plan=fp(root/'plan.json'),epoch=time.time())
   immutable(root/'complete.json',result);atomic(root/'controller.json',dict(status=STATUS,questions=result['questions'],trajectories=result['trajectories'],archives=len(groups_out),epoch=time.time()))
   return result
  except BaseException as error:
   atomic(root/'failure.json',dict(status='SNAPSHOT_NOT_ACCEPTED',error_type=type(error).__name__,error=str(error)[:1000],epoch=time.time()));raise


def main():
 p=argparse.ArgumentParser(description=__doc__);action=p.add_mutually_exclusive_group(required=True);action.add_argument('--discover',action='store_true');action.add_argument('--export',action='store_true')
 p.add_argument('--base',default=SOURCE);p.add_argument('--output-root',default=OUTPUT);p.add_argument('--snapshot-id');p.add_argument('--batches-per-archive',type=int,default=16);a=p.parse_args()
 if a.base!=SOURCE:p.error('Only the explicit new peer NAS10 source is allowed')
 if a.discover:print(json.dumps(summary(discover(Path(a.base)))),flush=True);return
 if sys.platform=='darwin':p.error('Payload snapshots must run on remote Linux, never on the Mac')
 if a.output_root!=OUTPUT or not a.snapshot_id:p.error('Explicit snapshot-id and independent declared remote output-root are required')
 print(json.dumps(export(Path(a.base),Path(a.output_root),a.snapshot_id,batches_per_archive=a.batches_per_archive)),flush=True)
if __name__=='__main__':main()
