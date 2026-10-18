"""Same-eight native recovery; no reshard, reset, or replay of completed outers."""
import gc,json,os,time
from pathlib import Path
from .diverse_resume import read,sha,once

MODE='diverse_same_eight_latest_joint_v1'
PROTECTED=('actor.py','actor_safety.py','probe.py','ab_probe.py','ab_policy.py',
 'probe_plan.py','rollout.py','actor_hidden.py','prefix_diversity.py',
 'b_progressive.py','grounding.py','interaction_protocol.py','pool.py','storage.py','launch.py')

def stopped(source):
 from .b_progressive_resume import stopped as check
 return check(Path(source))

def validate_compatibility(c,p):
 ref=c['runtime_compatibility'];root=Path(c['base'])/'repo/reproduction/prefix_selection200'
 if (sha(ref['path'])!=ref['sha256'] or p['runtime_commit']!=c['source_commit']
     or c['actor_checkpoint_binding_commit']!=p['binding_commit'] or c['world_size']!=8
     or c['actor_global_micro_batch_size']!=8 or set(p['protected_sha256'])!=set(PROTECTED)):
  raise ValueError('Native recovery runtime/scientific binding differs')
 for name,value in p['protected_sha256'].items():
  if sha(root/name)!=value:raise ValueError('Scientific runtime changed: '+name)
 return p

def qualify(source,output):
 """One full CPU read of genuine rolling state, before a new GPU writer."""
 import torch
 from .storage import actual_actor_boundary
 from .checkpoint_cpu_once import load_torch_cpu,hash_file,validate_actor_state,validate_rng,tensor_cpu
 s=Path(source);output=Path(output);stopped(s)
 j=read(s/'latest-joint.json');n=j['outer'];c=read(s/'preparation/launch-config.json')
 if not 51<=n<200 or c['world_size']!=8 or not actual_actor_boundary(j['actor'],n):
  raise ValueError('Latest completed native eight-rank joint required')
 if read(s/'actor/latest.json')!=j['actor'] or read(s/'probe/latest.json')!=j['probe']:
  raise ValueError('Uncommitted actor/probe suffix requires explicit reconciliation')
 if output.exists():
  p=read(output)
  if p['status']!='DIVERSE_NATIVE_FULL_STATE_CPU_PASS' or p['joint_sha256']!=sha(s/'latest-joint.json'):
   raise ValueError('Prior CPU qualification must be reconciled')
  return p
 once(output.with_suffix('.intent.json'),dict(source=str(s),outer=n,created_unix=time.time()))
 a=j['actor'];files=[];ranks=[]
 for rank,item in enumerate(a['rank_states']):
  saved,sealed=load_torch_cpu(item['path'],item)
  ranks.append(validate_actor_state(saved,a,dict(c,grounding_state=a['grounding_state']),n,rank))
  files.append(sealed);del saved;gc.collect()
 for name,item in a['hf_files'].items():files.append(hash_file(Path(a['hf_model_dir'])/name,item))
 ps=Path(j['probe_snapshot']);saved,sealed=load_torch_cpu(ps,j['probe']);files.append(sealed)
 steps=j['probe']['actual_adam_steps_per_rank'];assert steps==saved['actual_adam_steps_per_rank'] and len(steps)==8
 for name in ('model','ema_model'):
  for value in saved[name].values():tensor_cpu(value,finite=True)
 for state in saved['optimizer']['state'].values():
  assert int(state['step'])==steps[0]
  for name in ('exp_avg','exp_avg_sq'):tensor_cpu(state[name],dtype=torch.float32,finite=True)
 assert len(saved['rank_rng'])==8
 for rng in saved['rank_rng']:validate_rng(rng)
 del saved;gc.collect()
 assert sha(j['pool_snapshot'])==j['pool_sha256']
 p=dict(status='DIVERSE_NATIVE_FULL_STATE_CPU_PASS',source=str(s),outer=n,
  joint_sha256=sha(s/'latest-joint.json'),actor_adam=a['actual_adam_steps_per_rank'],
  probe_adam=steps,rank_checks=ranks,files=files,completed_unix=time.time())
 once(output,p);return p

def validate_imported(base,c):
 b=Path(base);r=read(b/'preparation/resume-joint-receipt.json');spec=c['resume_joint']
 source=Path(spec['source_base']);stopped(source)
 q=read(r['qualification']);j=read(source/'latest-joint.json');n=spec['completed_outer']
 if (r['mode']!=MODE or q['status']!='DIVERSE_NATIVE_FULL_STATE_CPU_PASS' or q['outer']!=n
     or sha(r['qualification'])!=r['qualification_sha256'] or sha(source/'latest-joint.json')!=q['joint_sha256']
     or j!=read(b/'latest-joint.json') or j['outer']!=n or read(b/'actor/latest.json')!=j['actor']
     or read(b/'probe/latest.json')!=j['probe'] or read(b/'question-pool/state.json')!=r['pool_state']):
  raise ValueError('Imported latest actor/probe/data boundary differs')
 for row in q['files']:
  st=Path(row['path']).stat()
  if (st.st_size,st.st_ino,st.st_mtime_ns)!=(row['bytes'],row['inode'],row['mtime_ns']):
   raise ValueError('Qualified immutable state changed')
 return n

def prepare(source,base,config,qualification):
 """Import exact immutable states plus a validated pending reservation only."""
 from .pool import reserve_for_outer,state_sha
 s=Path(source);b=Path(base);stopped(s);j=read(s/'latest-joint.json');n=j['outer'];q=read(qualification)
 assert q['outer']==n and q['joint_sha256']==sha(s/'latest-joint.json')
 def copy(src,dst):
  dst.parent.mkdir(parents=True,exist_ok=True)
  with dst.open('xb')as f:f.write(src.read_bytes())
 for i in range(1,n+1):
  root=s/'outers'/('outer-%06d'%i)
  for name in ('complete.json','pool-state-at-completion.json','grounding-state-at-completion.json','grounding-refresh.json'):
   if (root/name).exists():copy(root/name,b/'outers'/root.name/name)
 for name in ('binding.json','provenance.json'):
  if (s/'question-pool'/name).exists():copy(s/'question-pool'/name,b/'question-pool'/name)
 before=read(j['pool_snapshot']);current=read(s/'question-pool/state.json');pool=before
 if current!=before:
  old=read(s/'preparation/launch-config.json');root=s/'outers'/('outer-%06d'%(n+1));reservation=read(root/'pool-reservation.json')
  ids,cycles,after=reserve_for_outer(before,old,n+1)
  assert reservation['before_sha256']==state_sha(before) and reservation['after_sha256']==state_sha(after)
  assert reservation['indices']==ids and reservation['cycles']==cycles and reservation['after']==after and current==after
  pool=after
  for name in ('pool-reservation.json','selection.json','questions.json'):
   if (root/name).exists():copy(root/name,b/'outers'/root.name/name)
 # No old actor-binding marker, partial native generation or Adam attempt is imported.
 once(b/'question-pool/state.json',pool);once(b/'latest-joint.json',j)
 once(b/'actor/latest.json',j['actor']);once(b/'probe/latest.json',j['probe'])
 os.link(j['probe_snapshot'],b/'probe/latest.pt')
 for name in ('initialization-complete.json','initialization-intent.json'):
  if(s/'probe'/name).exists():copy(s/'probe'/name,b/'probe'/name)
 r=dict(mode=MODE,source_base=str(s),completed_outer=n,qualification=str(qualification),
  qualification_sha256=sha(qualification),pool_state=pool,actor_adam=q['actor_adam'],probe_adam=q['probe_adam'],
  pending_reservation_preserved=current!=before,completed_updates_reexecuted=0,optimizer_reset=False)
 once(b/'preparation/resume-joint-receipt.json',r);validate_imported(b,config);return r
