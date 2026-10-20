"""Retire only the obsolete full178 fallback after exact four-rank181 import.

The complete181 full CPU/SHA proof protects all recovery assets. No active
writer exists. Historical rollout-input linkage is provenance, not a requirement
to keep every previous model after a complete successor; its logs/raw/meta stay.
"""
import json,hashlib,shutil,time
from pathlib import Path

def main():
 parent=Path('/mnt/data/shaoshitong');s=parent/'search-g1-prefix-selection200-reference-replica-b-diverse10to5-cp178-shared5-recovery-20261020-r1'
 def read(p):return json.loads(p.read_text())
 def physical(x):
  prefix='/mnt/data/sihe-share5/workspace/training_data_node_10/shaoshitong/'
  return parent/x[len(prefix):]if x.startswith(prefix)else Path(x)
 j=read(s/'latest-joint.json');q=read(s/'preparation/recovery181-cpu/full-state-proof.json')
 assert j['outer']==q['outer']==181 and q['status']=='DIVERSE_NATIVE_FULL_STATE_CPU_PASS'
 assert hashlib.sha256((s/'latest-joint.json').read_bytes()).hexdigest()==q['joint_sha256']
 assert read(s/'logs/cleanup-status.json')['owned_births_all_gone']
 # Check every recorded owned birth independently, never use shared run_id.
 from reproduction.a800.owned_processes import same
 ledger=read(s/'logs/process-ledger.json')
 for row in ledger['jobs'].values():
  for who in row['ledger'].values():assert not same(who)
 protected={physical(row['path'])for row in q['files']}
 for row in q['files']:
  st=physical(row['path']).stat();assert(st.st_size,st.st_ino,st.st_mtime_ns)==(row['bytes'],row['inode'],row['mtime_ns'])
 converted=parent/'search-g1-prefix-selection200-reference-replica-b-diverse10to5-cp181-four-shared5-20261020-r1'
 ready=read(converted/'preparation/native-recovery-complete.json');assert ready['outer']==181 and ready['world_size']==4 and not ready['optimizer_reset']
 migration=read(converted/'preparation/actor-cp181-8to4-reshard.json');assert migration['all_moments_bitwise_preserved']and migration['new_optimizer_steps']==0
 assert migration['source_joint_sha256']==q['joint_sha256']
 assert not(converted/'preparation/submit-intent.json').exists()
 old=parent/'search-g1-prefix-selection200-reference-replica-b-diverse10to5-cp174-retained18-recovery-20261020-r1'
 assert read(old/'logs/cleanup-status.json')['owned_births_all_gone']
 a=read(old/'outers/outer-000178/complete.json')['actor'];root=physical(a['state_dir'])
 assert root.parent==old/'actor/outers'and a['completed_outer']==178 and j['actor']['completed_outer']==181
 assert physical(j['actor']['state_dir'])!=root
 assets=[(physical(row['path']),row)for row in a['rank_states']]
 assets += [(physical(a['hf_model_dir'])/name,row)for name,row in a['hf_files'].items()if name.endswith(('.safetensors','.bin'))]
 assert len(assets)==12
 plan=[]
 for f,r in assets:
  assert f.is_relative_to(root)and f not in protected and not f.is_symlink()
  st=f.stat();assert st.st_size==r['bytes']and st.st_nlink==1
  plan.append(dict(path=str(f),bytes=st.st_size,inode=st.st_ino,mtime_ns=st.st_mtime_ns,producer_sha256=r['sha256']))
 value=dict(status='INTENT',source=str(s),retired_outer=178,retained_complete_outer=181,assets=plan,
   obsolete_fallback_after_complete181_reshard=True,latest_milestone_or_resume_state_deleted=False,started_unix=time.time())
 intent=s/'preparation/retired-fallback178-intent.json'
 with intent.open('x')as f:json.dump(value,f)
 for row in plan:
  f=Path(row['path']);st=f.stat();assert(st.st_size,st.st_ino,st.st_mtime_ns)==(row['bytes'],row['inode'],row['mtime_ns']);f.unlink()
 value.update(status='OBSOLETE_FALLBACK178_RETIRED_KEEP_ORIGINAL_AND_CONVERTED181',removed_bytes=sum(x['bytes']for x in plan),free_bytes=shutil.disk_usage(s).free,completed_unix=time.time())
 with(s/'preparation/retired-fallback178-complete.json').open('x')as f:json.dump(value,f)
 with(old/'outers/outer-000178/retired-model-state-pruned.json').open('x')as f:json.dump(dict(audit=str(intent),weights_retired=True,metadata_and_raw_retained=True),f)
 print(json.dumps({k:v for k,v in value.items()if k!='assets'}),flush=True)

if __name__=='__main__':main()
