"""After verified owned cleanup, fork zero-write51 and reuse its exact sampling."""
import json,os,time,uuid
from pathlib import Path

def prepare(source,target,canonical,commit):
 from reproduction.prefix_selection200.diverse_resume import read,sha,once,validate_imported
 from reproduction.a800.owned_processes import same
 source=Path(source);target=Path(target);old=read(source/'preparation/launch-config.json')
 assert read(source/'logs/cleanup-status.json')['owned_births_all_gone'] and not same(read(source/'preparation/submit-receipt.json')['leader'])
 rows=[];writes=[]
 for rank in range(8):
  matches=list((source/'actor/outers').glob('outer-0051-attempt-*/rank-%d-updates.jsonl'%rank));assert len(matches)==1
  records=[json.loads(line)for line in matches[0].read_text().splitlines()];assert records[0]['event']=='ACTUAL_RESTORE'
  assert records[0]['actual_adam_before']==1000 and records[0]['restored_scheduler']==50
  writes.append(sum(r['event']=='ACTUAL_ADAM_WRITE'for r in records));rows.append(dict(rank=rank,path=str(matches[0]),sha256=sha(matches[0])))
 assert writes==[0]*8 and read(source/'actor/latest.json')['completed_outer']==50
 logs=list((source/'logs').glob('*actor*.log'));assert any('Monitor must use the exact physical4/reference8 rank assignment'in f.read_text()for f in logs)
 def rebase(x):
  if isinstance(x,dict):return{k:rebase(v)for k,v in x.items()}
  if isinstance(x,list):return[rebase(v)for v in x]
  if isinstance(x,str)and(x==old['base']or x.startswith(old['base']+'/')):return canonical+x[len(old['base']):]
  return x
 c=rebase(old);c.update(base=canonical,execution_id=uuid.uuid4().hex,source_commit=commit,actor_checkpoint_binding_commit=old['source_commit'])
 proof=dict(status='EIGHT_MONITOR_MAPPING_ZERO_WRITE_REPAIR',source_base=old['base'],binding_commit=old['source_commit'],runtime_commit=commit,source_actual_writes=writes,source_rank_journals=rows,complete_outer=50,source_cleanup=read(source/'logs/cleanup-status.json'),repair='four length-strata from each original virtual16 block; world4 two blocks -> world8 one;32 global unchanged',new_sampling_calls=0,optimizer_reset=False,created_unix=time.time())
 path=target/'preparation/runtime-compatibility.json';once(path,proof)
 c['runtime_compatibility']=dict(path=canonical+'/preparation/runtime-compatibility.json',sha256=sha(path))
 def links(relative):
  src=source/relative;dst=target/relative;dst.mkdir(parents=True,exist_ok=False)
  for parent,dirs,names in os.walk(src):
   for d in dirs:(dst/Path(parent).relative_to(src)/d).mkdir()
   for n in names:
    f=Path(parent)/n;assert not f.is_symlink();os.link(f,dst/f.relative_to(src))
 # Atomic writers replace latest entries. Historical checkpoint/tensor files
 # remain immutable shared inodes, never copied via Mac or rewritten.
 for name in ('probe','question-pool'):links(name)
 (target/'actor').mkdir();(target/'actor/latest.json').write_bytes((source/'actor/latest.json').read_bytes())
 for n in range(1,51):
  src=source/'outers'/('outer-%06d'%n);dst=target/'outers'/src.name;dst.mkdir(parents=True)
  for f in src.glob('*.json'):(dst/f.name).write_bytes(f.read_bytes())
 (target/'latest-joint.json').write_bytes((source/'latest-joint.json').read_bytes())
 for name in ('resume-joint-receipt.json','actor-cp50-4to8-reshard.json'):(target/'preparation'/name).write_bytes((source/'preparation'/name).read_bytes())
 original=source/'outers/outer-000051';root=target/'outers/outer-000051';root.mkdir()
 for name in ('prefix','probe-score','full','full-pool'):links('outers/outer-000051/'+name)
 for name in ('pool-reservation.json','selection.json','questions.json'):(root/name).write_bytes((original/name).read_bytes())
 (root/'imported-original-actor-binding.json').write_bytes((original/'actor-binding.json').read_bytes())
 seals={relative:sha(original/relative)for relative in ('prefix/complete.json','probe-score/complete.json','probe-score/manifest.json','probe-score/actual-em-selection.json','full/complete.json','full-pool/complete.json')}
 c['resume_cached_sampling']=dict(source_base=old['base'],outer=51,seals=seals)
 once(target/'preparation/launch-config.json',c)
 from reproduction.prefix_selection200.diverse_cached_resume import validate_cached_sampling
 # Namespace-dependent validation executes later inside the private wrapper.
 return c
