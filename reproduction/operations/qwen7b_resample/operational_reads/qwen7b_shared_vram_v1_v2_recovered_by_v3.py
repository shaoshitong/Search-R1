"""Record historical failures recovered only by the two actually audited v3 batches."""
import hashlib,json,os,time
from pathlib import Path
b=Path('/mnt/data/shaoshitong/qwen7b-resample-seed20261021-g5-20261020')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
replacement=list((b/'pool/jobs/job-shared5-1259750').glob('attempt-*'));assert len(replacement)==1
progress=read(replacement[0]/'progress.json');assert progress['completed_own_batches']>=2 and progress['run_id']=='70188b8e50fd4b45bc420ce58014f46e'
audit=b/'audit/latest.json';a=read(audit);assert a['failed_batches']==0 and a['verified_batches']>=244 and a['epoch']>=progress['epoch']
controlled=b/'preparation/shared-vram-terminal-v3-controlled-throughput/complete.json';assert sha(controlled)=='169e0892d5084658736cd4c097f55d98b041bf7a1d0951a29fe7badaea543e57'
output=[]
for pid,version,proof_sha in[(1160466,'v1','980f3e9478e62f7c5bd0107a5e35457dd7c2286645cca943d804af27d7c93982'),(1218632,'v2','1ae48d7d35430cc2064039c8da618cc836754e8f390929fc9f395580eb614f08')]:
 attempts=list((b/'pool/jobs'/('job-shared5-'+str(pid))).glob('attempt-*'));assert len(attempts)==1
 old=attempts[0];failure=old/'failure.json';assert failure.is_file()
 proof=b/'preparation'/('shared-vram-terminal-'+version)/'complete.json';assert sha(proof)==proof_sha
 target=old/'resolution.json'
 if target.exists():
  v=read(target);assert v['original_failure_sha256']==sha(failure)and v['terminal_proof_sha256']==proof_sha and v['replacement_new_complete_batches']>=2
  output.append(dict(path=str(target),status='EXISTING_HISTORICAL_RESOLUTION_READ_NO_REPLAY'));continue
 v=dict(status='RECOVERED_FAILED_ATTEMPT_WITH_PRODUCTIVE_REPLACEMENT',run_id=progress['run_id'],original_failure_sha256=sha(failure),terminal_proof=str(proof),terminal_proof_sha256=proof_sha,replacement_directory=str(replacement[0]),replacement_new_complete_batches=progress['completed_own_batches'],replacement_new_complete_trajectories=progress['trajectories'],replacement_progress_sha256=sha(replacement[0]/'progress.json'),independent_audit=str(audit),independent_audit_sha256=sha(audit),replacement_final_state='CONTROLLED_THROUGHPUT_STOP_AFTER_REAL_PRODUCTIVE_RECOVERY',replacement_controlled_stop_proof=str(controlled),replacement_controlled_stop_proof_sha256=sha(controlled),original_failure_not_deleted_or_rewritten=True,whole_goal_complete=False,epoch=time.time())
 with target.open('x')as f:json.dump(v,f,indent=2);f.flush();os.fsync(f.fileno())
 output.append(dict(path=str(target),sha256=sha(target),replacement_batches=progress['completed_own_batches'],status=v['status']))
print(json.dumps(dict(resolutions=output,whole_goal_complete=False)))
