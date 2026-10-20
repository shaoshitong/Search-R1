"""Cold, one-shot retirement of Actor186 after complete187 and real188 writes.

Protect current native recovery state, all milestones, raw data and journals.
The exact namespace and outer are intentional; this cannot sweep shared run IDs.
"""
import fcntl,json,re,shutil,time
from pathlib import Path

BASE=Path('/mnt/data/shaoshitong/search-g1-prefix-selection200-reference-replica-b-diverse10to5-cp181-four-shared5-20261020-r2')
PREFIX='/mnt/data/sihe-share5/workspace/training_data_node_10/shaoshitong/'
def read(f):return json.loads(f.read_text())
def physical(s):return Path('/mnt/data/shaoshitong')/s[len(PREFIX):] if s.startswith(PREFIX) else Path(s)
def main():
 b=BASE;c=read(b/'preparation/launch-config.json');j=read(b/'latest-joint.json')
 assert c['world_size']==4 and c['physical_gpu_indices']==[0,1,2,3] and c['expected_execution_node']=='h100-5'
 assert j['outer']>=187 and j==read(b/('outers/outer-%06d/complete.json'%j['outer']))
 assert j['actor']==read(b/'actor/latest.json') and j['probe']==read(b/'probe/latest.json')
 assert len(j['actor']['rank_states'])==4 and j['actor']['scheduler_completed_outer']==j['outer']
 assert not(b/'logs/cleanup-status.json').exists() and shutil.disk_usage(b).free<38426547064
 protected=set()
 for row in j['actor']['rank_states']:
  f=physical(row['path']);assert f.stat().st_size==row['bytes'];protected.add(f)
 for name,row in j['actor']['hf_files'].items():
  f=physical(j['actor']['hf_model_dir'])/name;assert f.stat().st_size==row['bytes'];protected.add(f)
 # Directly verify that the active successor has restored187 or later on all ranks.
 active=read(b/'logs/pipeline-status.json');assert active['outer']>=188
 restores=[]
 for f in(b/'actor/outers').glob('outer-0188-attempt-*/rank-*-updates.jsonl'):
  rows=[json.loads(line) for line in f.read_bytes().splitlines(keepends=True) if line.endswith(b'\n') and line.strip()]
  restore=next(x for x in rows if x['event']=='ACTUAL_RESTORE')
  assert restore['restored_completed_outer']==187 and restore['actual_adam_before']==3740
  assert any(x['event']=='ACTUAL_ADAM_WRITE' for x in rows)
  restores.append(restore['rank'])
 assert sorted(restores)==[0,1,2,3]
 old=read(b/'outers/outer-000186/complete.json')['actor'];root=physical(old['state_dir'])
 assert root.parent==b/'actor/outers' and re.fullmatch(r'outer-0186-attempt-[0-9a-f]+',root.name)
 assert old['completed_outer']==186 and old['scheduler_completed_outer']==186
 assert old['actual_adam_steps_per_rank']==[3720]*4 and not old.get('milestone')
 assert not(b/'outers/outer-000186/retired-model-state-pruned.json').exists()
 assets=[(physical(row['path']),row) for row in old['rank_states']]
 assets += [(physical(old['hf_model_dir'])/name,row) for name,row in old['hf_files'].items() if name.endswith(('.safetensors','.bin'))]
 assert len(assets)==8
 plan=[]
 for f,row in assets:
  assert f.is_relative_to(root) and f not in protected and not f.is_symlink()
  st=f.stat();assert st.st_size==row['bytes'] and st.st_nlink==1 and re.fullmatch('[0-9a-f]{64}',row['sha256'])
  plan.append(dict(path=str(f),dev=st.st_dev,inode=st.st_ino,bytes=st.st_size,mtime_ns=st.st_mtime_ns,producer_sha256=row['sha256']))
 lock=(b/'preparation/storage-prune.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 assert read(b/'latest-joint.json')['outer']==187,'Native commit advanced; reconcile instead of racing native prune'
 assert read(b/'logs/pipeline-status.json')['stage'].startswith('outer-000188-actor-')
 value=dict(started_unix=time.time(),source_outer=186,protected_complete_outer=j['outer'],active_outer=active['outer'],active188_restored187_ranks=restores,assets=plan,latest_and_milestones_preserved=True,foreign_operations=False)
 audit=b/'preparation/retired-obsolete186-intent.json'
 with audit.open('x')as f:json.dump(value,f)
 assert read(b/'latest-joint.json')['outer']>=187
 for row in plan:
  f=Path(row['path']);st=f.stat();assert(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)==(row['dev'],row['inode'],row['bytes'],row['mtime_ns']);f.unlink()
 value.update(status='OBSOLETE186_PAYLOAD_RETIRED_NOT_REPLAYABLE',removed_unique_bytes=sum(row['bytes'] for row in plan),free_bytes=shutil.disk_usage(b).free,completed_unix=time.time())
 with(b/'preparation/retired-obsolete186-complete.json').open('x')as f:json.dump(value,f)
 with(b/'outers/outer-000186/retired-model-state-pruned.json').open('x')as f:json.dump(dict(retirement=str(audit),weight_payload_retired=True,metadata_raw_and_journals_retained=True),f)
 print(json.dumps({k:v for k,v in value.items() if k!='assets'}),flush=True)
if __name__=='__main__':main()
