"""Build the exact deletion manifest from both completed permission views."""
import collections
import hashlib
import importlib.util
import json
from pathlib import Path
import time

HERE=Path(__file__).resolve().parent
CWD=HERE.parents[4]
OUT=CWD/'outputs/easy2560-qwen7b-20261021'
spec=importlib.util.spec_from_file_location('apply_old',HERE/'apply_authorized_old_checkpoint_cleanup.py')
apply=importlib.util.module_from_spec(spec);spec.loader.exec_module(apply)


def main():
 files={};texts={};jobs={};inputs=[];invs=[]
 for user in ('jxu957','mou176'):
  source=max(OUT.glob('old-cleanup-inventory-status-'+user+'-*.json'),key=lambda p:p.stat().st_mtime)
  wrapper=json.loads(source.read_text());assert wrapper['returncode']==0
  record=json.loads(wrapper['stdout']);assert not record['same_birth']and not record.get('incomplete_output')
  inv=record['inventory'];invs.append(inv);inputs.append(dict(path=str(source),sha256=hashlib.sha256(source.read_bytes()).hexdigest(),remote_control=record['control']))
  for row in inv['files']:
   previous=files.get(row['path'])
   if previous:assert all(previous[k]==row[k]for k in('bytes','device','inode','mtime_ns','uid'))
   files[row['path']]=row
  for row in inv['preserved_text']:texts[row['path']]=row
  for job in inv['active_jobs']:jobs[job['job']]=job
 selected=[];preserved_binary=[]
 for row in files.values():
  if apply.allowed(Path(row['path'])):
   assert row['uid']in(14090,21930);row=dict(row,apply_uid=row['uid']);selected.append(row)
  else:preserved_binary.append(row)
 refs=[]
 for row in selected:
  target=Path(row['path'])
  for job in jobs.values():
   for value in job['paths']:
    ref=Path(value)
    if ref==apply.ROOT:continue
    if target==ref or ref in target.parents:refs.append(dict(job=job['job'],path=row['path'],reference=value))
 assert not refs,refs
 # All text extensions are ineligible for unlink. Recheck actual logs/results and CP sidecars after execution.
 critical_text=[row for row in texts.values()if any(x in row['path'].lower()for x in('/logs/','/results/','/evaluation','/final','/trajectory','/trajectories/','/checkpoints/','/rolling-checkpoints/'))]
 groups=collections.defaultdict(lambda:dict(files=0,path_allocated_bytes=0));links=collections.defaultdict(list)
 for row in selected:
  g=groups[row['experiment']];g['files']+=1;g['path_allocated_bytes']+=row['allocated_bytes'];links[(row['device'],row['inode'])].append(row)
 reclaim=sum(rows[0]['allocated_bytes']for rows in links.values()if len(rows)==rows[0]['nlink'])
 assert len({r['path']for r in selected})==len(selected)
 plan=dict(schema_version=1,epoch=time.time(),authorization='2026-10-21-human-delete-all-checkpoints-exact-old-experiments-preserve-text-and-eval-trajectories',
           user_authorization_summary='Delete all experiment checkpoint payloads in only DynRPO R1-R4, TOPD, old3BEasy and CoSearch; retain all text and evaluation trajectories. Overrides prior latest-two retention for these roots only.',
           scope=sorted(apply.ALLOWED_ROOTS),inputs=inputs,delete_files=sorted(selected,key=lambda r:r['path']),
           preserved_text=critical_text,preserved_binary=preserved_binary,preserved_text_total_count=len(texts),
           groups=dict(groups),expected_reclaimable_allocated_bytes=reclaim,unique_target_inodes=len(links),
           logical_unlink_path_allocated_bytes=sum(r['allocated_bytes']for r in selected),queue='\n'.join(x['queue'].strip()for x in invs),
           active_jobs=list(jobs.values()),active_retired_checkpoint_references=refs,
           feature7b_modified=False,qprobe_modified=False,current7b_modified=False,remote_archives_modified=False,
           permission_errors_by_uid={str(x['uid']):len(x['errors'])for x in invs},execution_performed=False)
 target=OUT/'authorized-old-checkpoint-cleanup-plan-20261021.json'
 raw=json.dumps(plan,ensure_ascii=False,sort_keys=True,indent=2)+'\n';target.write_text(raw)
 digest=hashlib.sha256(raw.encode()).hexdigest()
 (OUT/'authorized-old-checkpoint-cleanup-envelope-20261021.json').write_text(json.dumps(dict(plan=raw,plan_sha256=digest))+'\n')
 print(json.dumps(dict(plan=str(target),sha256=digest,target_files=len(selected),groups=dict(groups),apply_uids=dict(collections.Counter(r['apply_uid']for r in selected)),unique_target_inodes=len(links),expected_reclaimable_allocated_bytes=reclaim,preserved_text_checked=len(critical_text),preserved_binary=len(preserved_binary)),ensure_ascii=False))

if __name__=='__main__':main()
