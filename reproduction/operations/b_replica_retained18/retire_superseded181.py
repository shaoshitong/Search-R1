"""Retire redundant nonmilestone181 payloads only after real four-rank successors.

The live writer is never stopped or modified. Historical metadata/raw/journals
remain; latest/milestone states are protected. This is exact lineage cleanup,
not a shared run_id search. A consumed intent is never replayed.
"""
import argparse,hashlib,json,time,shutil
from pathlib import Path

PARENT=Path('/mnt/data/shaoshitong')
NAME='search-g1-prefix-selection200-reference-replica-b-diverse10to5-cp181-four-shared5-20261020-r2'
OLD='search-g1-prefix-selection200-reference-replica-b-diverse10to5-cp178-shared5-recovery-20261020-r1'
FIRST='search-g1-prefix-selection200-reference-replica-b-diverse10to5-cp181-four-shared5-20261020-r1'

def main():
 p=argparse.ArgumentParser();p.add_argument('--kind',required=True,choices=['derived-four','original-eight']);a=p.parse_args()
 def read(f):return json.loads(f.read_text())
 def physical(s):
  prefix='/mnt/data/sihe-share5/workspace/training_data_node_10/shaoshitong/'
  return PARENT/s[len(prefix):]if s.startswith(prefix)else Path(s)
 b=PARENT/NAME;old=PARENT/OLD;first=PARENT/FIRST
 j=read(b/'latest-joint.json');n=j['outer'];c=read(b/'preparation/launch-config.json')
 assert n>=(182 if a.kind=='derived-four' else 183)and c['world_size']==4 and c['physical_gpu_indices']==[0,1,2,3]
 assert not(b/'logs/cleanup-status.json').exists()
 # Require committed joint and directly sealed four-rank optimizer journals.
 assert j==read(b/('outers/outer-%06d/complete.json'%n))and j['actor']==read(b/'actor/latest.json')
 assert j['probe']==read(b/'probe/latest.json')and j['actor']['scheduler_completed_outer']==n
 protected={physical(r['path'])for r in j['actor']['rank_states']}
 for rank,row in enumerate(j['actor']['rank_receipts']):
  assert row['rank']==rank and row['complete_outer']and row['opportunities_processed']==20 and row['new_adam_steps']>=1
  f=physical(row['ledger_path']);raw=f.read_bytes();assert hashlib.sha256(raw).hexdigest()==row['ledger_sha256']
  records=[json.loads(line)for line in raw.splitlines()];restore=next(r for r in records if r['event']=='ACTUAL_RESTORE')
  assert restore['restored_completed_outer']==n-1 and restore['restored_scheduler']==n-1
  assert sum(r['event']=='ACTUAL_ADAM_WRITE'for r in records)==row['new_adam_steps']
 assert len(j['actor']['rank_receipts'])==len(protected)==4
 for row in j['actor']['rank_states']:
  assert physical(row['path']).stat().st_size==row['bytes']
 for name,row in j['actor']['hf_files'].items():
  f=physical(j['actor']['hf_model_dir'])/name;assert f.stat().st_size==row['bytes'];protected.add(f)
 # New work must already use this successor, with positive real sampling.
 stage=read(b/'logs/pipeline-status.json');assert stage['outer']>=n+1
 growth=[read(f).get('completed_trajectories',0)for f in(b/('outers/outer-%06d'%(n+1))).rglob('*progress*.json')if f.stat().st_size<15000]
 assert any(x>0 for x in growth),'Do not retire before actual successor work'
 if a.kind=='derived-four':
  m=read(first/'preparation/actor-cp181-8to4-reshard.json');assert m['new_optimizer_steps']==0 and m['all_moments_bitwise_preserved']
  assets=[]
  for row in m['rank_states']:
   f=physical(row['path']);assert f.parent==first/'actor/imported-cp181-four'
   aliases=[f,b/'actor/imported-cp181-four'/f.name]
   assert len({(x.stat().st_dev,x.stat().st_ino)for x in aliases})==1
   assert all(x.stat().st_nlink==2 for x in aliases)
   assets.extend((x,row)for x in aliases)
 else:
  assert read(old/'logs/cleanup-status.json')['owned_births_all_gone']
  prior=read(old/'latest-joint.json');assert prior['outer']==181
  actor=prior['actor'];root=physical(actor['state_dir']);assert root.parent==old/'actor/outers'
  assets=[(physical(row['path']),row)for row in actor['rank_states']]
  assets += [(physical(actor['hf_model_dir'])/name,row)for name,row in actor['hf_files'].items()if name.endswith(('.safetensors','.bin'))]
  assert len(assets)==12 and all(f.is_relative_to(root)and f.stat().st_nlink==1 for f,row in assets)
 plan=[]
 for f,row in assets:
  assert f not in protected and not f.is_symlink();st=f.stat();assert st.st_size==row['bytes']
  plan.append(dict(path=str(f),dev=st.st_dev,inode=st.st_ino,bytes=st.st_size,mtime_ns=st.st_mtime_ns,producer_sha256=row['sha256']))
 value=dict(kind=a.kind,successor_complete=n,source_outer=181,latest_and_milestones_preserved=True,assets=plan,started_unix=time.time())
 audit=b/'preparation'/('retired181-'+a.kind+'-intent.json')
 with audit.open('x')as h:json.dump(value,h)
 assert read(b/'latest-joint.json')['outer']>=n
 for row in plan:
  f=Path(row['path']);st=f.stat();assert(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)==(row['dev'],row['inode'],row['bytes'],row['mtime_ns']);f.unlink()
 if a.kind=='derived-four':
  with(first/'preparation/derived181-payload-retired.json').open('x')as h:json.dump(dict(current_base=str(b),successor_complete=n,retirement_intent=str(audit),original_eight181_retained=True,derived_rank_payloads_no_longer_exist=True),h)
 unique={(r['dev'],r['inode']):r['bytes']for r in plan}
 value.update(status='SUPERSEDED181_PAYLOAD_RETIRED_NOT_REPLAYABLE',removed_unique_bytes=sum(unique.values()),free_bytes=shutil.disk_usage(b).free,completed_unix=time.time())
 with(b/'preparation'/('retired181-'+a.kind+'-complete.json')).open('x')as h:json.dump(value,h)
 print(json.dumps({k:v for k,v in value.items()if k!='assets'}),flush=True)

if __name__=='__main__':main()
