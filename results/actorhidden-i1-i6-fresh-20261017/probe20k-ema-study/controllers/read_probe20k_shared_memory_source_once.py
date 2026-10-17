import json,sys,datetime
from pathlib import Path
sys.path.insert(0,'/Users/shaoshitong/Documents/Search-G1/actorhidden-period20-20261016')
from reproduction.h1008_qprobe.established_controller_transport import remote
script=r'''
import pathlib,json,subprocess,time,sys
b=pathlib.Path('/mnt/data/shaoshitong/search-g1-prefix-selection200-probe20k-dataset-20261017-0bdddc20840b-r48')
p=pathlib.Path('/dataset/shaoshitong/Search-G1-common8/env/lib/python3.10/site-packages/vllm/worker/worker.py');s=p.read_text();a=s.index('def determine_num_available_blocks');z=s.index('def initialize_cache',a)
ledger=json.loads((b/'logs/process-ledger.json').read_text());sys.path.insert(0,str(b/'repo'))
from reproduction.prefix_only_correctness_ab.launch import birth_states
jobs=ledger.get('jobs',ledger);births=[v for j in jobs.values()if isinstance(j,dict)for v in j.get('ledger',{}).values()]
result=dict(memory_source=s[a:z],original_ledger_keys=list(ledger),original_birth_count=len(births),remaining_original_births=birth_states(births),gpu_memory=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.total,memory.used,memory.free','--format=csv,noheader,nounits'],text=True,capture_output=True).stdout,cgroup_present=pathlib.Path('/sys/fs/cgroup/system.slice/slurmstepd.scope/job_3468').exists(),checked_unix=time.time())
(b/'preparation/terminal3468-second-release-source-read.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
'''
p=Path('work')/('probe20k-shared-cache-source-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'.json');v=remote(script,str(p.with_name(p.stem+'.transport.json')));p.write_text(json.dumps(json.loads(v),indent=2)+'\n');print(v)
