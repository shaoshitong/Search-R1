from pathlib import Path
import sys,json,datetime
sys.path.insert(0,'/Users/shaoshitong/Documents/Search-G1/actorhidden-period20-20261016')
from reproduction.h1008_qprobe.established_controller_transport import remote
code=r'''
import pathlib,json,subprocess,shutil,socket,importlib.metadata
p=pathlib.Path('/mnt/data/shaoshitong/search-g1-prefix-selection200-actorhidden20-i1-i6-fresh-20261017-6cd6f043ef16/preparation/launch-config.json')
c=json.loads(p.read_text());keys=['environment_root','model_dir','official_model_dir','canonical_train','validation_questions','final_questions','common_manifest_path','retrieval_index','retrieval_corpus','retrieval_encoder','retriever_gpu_lib','grounding_manifest','grounding_calibration_questions']
v={'host':socket.gethostname(),'old_config_path':str(p),'assets':{}}
for k in keys:
 if k not in c:continue
 p=pathlib.Path(c[k]);v['assets'][k]={'path':str(p),'exists':p.exists(),'bytes':p.stat().st_size if p.is_file()else None}
v['gpu_inventory']=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.total,memory.used','--format=csv,noheader,nounits'],text=True)
v['disk']=dict(zip(('total','used','free'),shutil.disk_usage('/mnt/data')))
v['mount']=subprocess.run(['findmnt','-T','/mnt/data','-n','-o','SOURCE,FSTYPE,TARGET'],capture_output=True,text=True).stdout
print(json.dumps(v))
'''
tag=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');p=Path('work')/('prefix5-reference-h100-inventory-'+tag+'.transport.json');v=json.loads(remote(code,str(p)));q=Path('work/prefix5-reference-h100-inventory.json');q.write_text(json.dumps(v,indent=2)+'\n');print(json.dumps(v))
