from pathlib import Path
import hashlib,json,subprocess,time
A=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021');root=A/'fullP80-MSE-terminal-CPU-coordinators-20261023-v2';code=root/'code-669427f37da0';j=json.loads((code/'staged-19137.json').read_bytes());assert len(j['results'])==1;r=j['results'][0];assert r['job_id']=='750190'
def proof(p):
 raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
for k in('worker','config','coordinator_entry','existing_CPU_entry','existing_CPU_validator','old_v1_failure_preserved'):assert proof(Path(r[k]['path']))==r[k]
subprocess.check_call(['bash','-n',r['coordinator_entry']['path']]);source=Path(r['worker']['path']).read_text();assert'def wait_for_terminal'in source and'class TrainingTerminalFailure'in source and'except(OSError,json.JSONDecodeError,subprocess.TimeoutExpired,ObservationUnavailable)'in source
old=A/'fullP80-MSE-terminal-CPU-coordinators-20261023-v1';seed1=json.loads((old/'job-750189/controller.json').read_bytes());p=Path('/proc/%d'%seed1['pid'])
try:seed1_live=p.joinpath('stat').read_text().rsplit(')',1)[1].split()[19]==seed1['start_ticks']
except(FileNotFoundError,ProcessLookupError):seed1_live=False
print(json.dumps(dict(status='SEED2_V2_SOURCE_ENTRY_CONFIG_REHASHED_OLD_FAILURE_PRESERVED_NO_START_BY_CHECK',time=time.time(),result=r,healthy_seed1_v1_identity_live=seed1_live,seed1_v1_PID=seed1['pid'],seed1_v1_start_ticks=seed1['start_ticks'],v2_controller_exists=(root/'job-750190/controller.json').exists(),workers_started_or_changed=False,CP_or_tensor_or_final_read=False,training_or_Slurm_changed=False)))
