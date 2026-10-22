from pathlib import Path
import hashlib,json,os,time
A=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021');root=A/'fullP80-MSE-terminal-CPU-coordinators-20261023-v1';out=root/'job-750190';old=root/'code-9772c3a72e4b';base=A/'full169028-L15P80-MSE-ownG5-physical-seed19137-plan2026102302-20261023-v3';validator=A/'full169028-L15P80-MSE-terminal-CPU-acceptance-20261023-v1/code-693a81dfbbae/accept_full169028_L15P80_MSE_checkpoint.py';entry=validator.parent/'seed19137.sh';controller=json.loads((out/'controller.json').read_bytes());assert controller['pid']==3344791 and controller['start_ticks']=='1410521';p=Path('/proc/3344791')
try:live=p.joinpath('stat').read_text().rsplit(')',1)[1].split()[19]==controller['start_ticks']
except(FileNotFoundError,ProcessLookupError):live=False
assert not live;matches=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  if p.stat().st_uid!=os.getuid():continue
  argv=(p/'cmdline').read_bytes().split(bytes([0]))
  if(str(old/'fullP80_MSE_terminal_CPU_coordinator.py').encode()in argv and str(old/'job-750190.json').encode()in argv)or(str(validator).encode()in argv and str(base).encode()in argv)or str(entry).encode()in argv:matches.append(dict(pid=int(p.name),argv=[x.decode(errors='replace')for x in argv if x]))
 except(FileNotFoundError,PermissionError,ProcessLookupError):pass
failure=out/'failure.json';raw=failure.read_bytes();locks=[line for line in Path('/proc/locks').read_text().splitlines()if' 3344791 'in line];assert not matches and not locks and not(out/'active-child.json').exists();print(json.dumps(dict(status='OLD_SEED2_COORDINATOR_DEAD_NO_SAME_JOB_COORDINATOR_OR_IMMUTABLE_CPU_READER_OR_PID_LOCK',time=time.time(),old_PID=3344791,old_start_ticks='1410521',actual_same_identity_live=False,matching_same_job_processes=matches,old_PID_proc_locks=locks,active_CPU_child_receipt_exists=False,failure=dict(path=str(failure),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),receipt=json.loads(raw)),old_failure_preserved=True,workers_or_CP_readers_started=False,training_or_Slurm_changed=False,CP_or_tensor_or_final_read=False)))
