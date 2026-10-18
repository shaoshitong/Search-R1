"""Finish metadata after the specific optional-provenance CP50 import failure.

No tensors, optimizer states, frozen runtime or source experiment are changed.
"""
import argparse,copy,json,time
from pathlib import Path

def main():
 from reproduction.prefix_selection200.diverse_resume import read,sha,once,validate_imported,ADAPTER,imported_actor_boundary
 parser=argparse.ArgumentParser();parser.add_argument('--base',required=True);args=parser.parse_args();b=Path(args.base)
 assert not(b/'preparation/submit-intent.json').exists(),'No repair after training submission'
 terminal=read(b/'preparation/import-worker-terminal.json')
 assert terminal['status']=='FAILED' and 'provenance.json' in terminal['traceback'] and 'FileNotFoundError' in terminal['traceback']
 c=read(b/'preparation/launch-config.json');source=Path(c['resume_joint']['source_base']);jp=source/'outers/outer-000050/complete.json';j=read(jp)
 assert sha(jp)==c['resume_joint']['source_joint_sha256']
 migration_path=b/'preparation/actor-cp50-4to8-reshard.json';migration=read(migration_path)
 actor=read(b/'actor/latest.json');assert imported_actor_boundary(actor,50)
 assert migration['rank_states']==actor['rank_states'] and migration['new_optimizer_steps']==0
 for f in actor['rank_states']:assert Path(f['path']).stat().st_size==f['bytes']
 probe=read(b/'probe/latest.json');assert probe['completed_outer']==50 and probe['actual_adam_steps_per_rank']==[700]*8
 assert sha(b/'probe/head-imported-cp50.pt')==probe['sha256']==sha(b/'probe/latest.pt')
 assert read(b/'question-pool/binding.json')==read(source/'question-pool/binding.json')
 assert not(source/'question-pool/provenance.json').exists()
 assert sha(j['pool_snapshot'])==j['pool_sha256']
 once(b/'preparation/optional-source-provenance-absent.json',dict(source=str(source),source_provenance_present=False,pool_binding_sha256=sha(source/'question-pool/binding.json'),pool_snapshot_sha256=j['pool_sha256'],repaired_unix=time.time()))
 (b/'question-pool/state.json').write_bytes(Path(j['pool_snapshot']).read_bytes())
 fork=copy.deepcopy(j);fork.update(actor=actor,probe=probe,probe_snapshot=str(b/'probe/head-imported-cp50.pt'),imported_boundary=True,source_cp50_joint_sha256=sha(jp))
 (b/'outers/outer-000050/parent-complete.json').write_bytes(jp.read_bytes())
 (b/'outers/outer-000050/complete.json').write_text(json.dumps(fork,sort_keys=True))
 once(b/'latest-joint.json',fork)
 result=dict(status='B_CP50_DIVERSE_EIGHT_IMPORTED',adapter=ADAPTER,completed_outer=50,source_joint_path=str(jp),source_joint_sha256=sha(jp),actor_migration=dict(path=str(migration_path),sha256=sha(migration_path)),imported_actor_adam=[1000]*8,imported_probe_adam=[700]*8,optimizer_reset=False,exact_model_and_adam_preserved=True,exact_executor_rng_resume=False,new_optimizer_steps=0,optional_source_provenance_absent=True)
 once(b/'preparation/resume-joint-receipt.json',result);validate_imported(b,c)
 once(b/'preparation/import-finalizer-terminal.json',dict(status='COMPLETE',receipt=result,finished_unix=time.time(),original_failed_terminal_preserved=True))
 print(json.dumps(result),flush=True)
if __name__=='__main__':main()
