"""Read existing CPU proxy counters inside job745371; no model/GPU calls."""
import json
import os
import subprocess
import time
from pathlib import Path

BASE = Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-exact-cp1-r2-20261021')
assert os.getuid() == 14090 and (BASE/'job.id').read_text().strip() == '745371'
code = """import json,os,socket
import requests
assert os.getuid()==14090 and os.environ.get('SLURM_JOB_ID')=='745371'
with requests.Session() as session:
 session.trust_env=False
 response=session.get('http://127.0.0.1:18010/health',timeout=10)
 response.raise_for_status()
 value=response.json()
assert value['ok'] and value['run_id']=='232f20e112ac4a978c5aba52a0367195'
assert str(value['slurm_job_id'])=='745371' and str(value['service_job_id'])=='745198'
assert value['physical_gpu_count']==2 and value['local_proxy_gpu_count']==0
print(json.dumps(dict(host=socket.gethostname(),job=os.environ['SLURM_JOB_ID'],proxy_health=value)))
"""
result = subprocess.run(['srun', '--jobid=745371', '--overlap', '--ntasks=1',
                         '--cpus-per-task=1', '--gres=none', 'python3', '-B', '-c', code],
                        capture_output=True, text=True, timeout=25)
print(json.dumps(dict(checked_unix=time.time(), job='745371', read_only=True,
                     gpu_allocation_requested=False, models_loaded=0,
                     returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)))
