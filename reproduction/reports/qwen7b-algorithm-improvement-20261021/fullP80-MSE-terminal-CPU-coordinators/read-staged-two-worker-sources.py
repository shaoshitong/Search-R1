import hashlib,json,subprocess,time
from pathlib import Path
A=Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021');root=A/'fullP80-MSE-terminal-CPU-coordinators-20261023-v1';code=root/'code-9772c3a72e4b';j=json.loads((code/'staged-None.json').read_bytes())
def proof(p):
 raw=p.read_bytes();return dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
for r in j['results']:
 for k in('worker','config','coordinator_entry','existing_CPU_entry','existing_CPU_validator'):assert proof(Path(r[k]['path']))==r[k]
 cfg=json.loads(Path(r['config']['path']).read_bytes());assert cfg['job_id']==r['job_id']and cfg['entry']==r['existing_CPU_entry']and cfg['validator']==r['existing_CPU_validator']and cfg['accepted']==r['accepted_path'];subprocess.check_call(['bash','-n',r['coordinator_entry']['path']]);r['actual_worker_controller_exists']=(root/('job-'+r['job_id'])/'controller.json').is_file()
print(json.dumps(dict(status='TWO_FULL_MSE_TERMINAL_COORDINATOR_SOURCE_CONFIG_ENTRY_REHASHED_NO_EXECUTION',time=time.time(),results=j['results'],coordinator_or_CPU_reader_started_by_this_check=False,checkpoint_or_tensor_or_final_read=False,Slurm_mutated=False)))
