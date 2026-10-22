"""Read existing CPU proxy counters inside job750029; no model/GPU calls."""
import json
import os
import subprocess
import time
from pathlib import Path

BASE = Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fa3-leanmetrics-r1-20261023')
assert os.getuid() == 14090 and (BASE/'job.id').read_text().strip() == '750029'
code = """import json,os,socket,time
import requests
assert os.getuid()==14090 and os.environ.get('SLURM_JOB_ID')=='750029'
with requests.Session() as session:
 session.trust_env=False
 response=session.get('http://127.0.0.1:18010/health',timeout=10)
 response.raise_for_status()
 value=response.json()
assert value['ok'] and value['run_id']=='232f20e112ac4a978c5aba52a0367195'
assert str(value['slurm_job_id'])=='750029' and str(value['service_job_id'])=='745198'
assert value['physical_gpu_count']==2 and value['local_proxy_gpu_count']==0
print(json.dumps(dict(host=socket.gethostname(),job=os.environ['SLURM_JOB_ID'],proxy_health=value, observed_unix=time.time())))
"""
result = subprocess.run(['srun', '--jobid=750029', '--overlap', '--exact', '--ntasks=1',
                         '--cpus-per-task=1', '--mem=256M', '--gres=none', '/data/user/jxu957/experiments/topd-20261008/runtime/env/bin/python', '-B', '-c', code],
                        capture_output=True, text=True, timeout=25)
print(json.dumps(dict(checked_unix=time.time(), job='750029', read_only=True,
                     gpu_allocation_requested=False, models_loaded=0,
                     returncode=result.returncode, stdout=result.stdout, stderr=result.stderr)))
