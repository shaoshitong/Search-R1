import json,subprocess
from pathlib import Path
b=Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-fa3-batch-searchstop-r1-20261023');run=json.loads((b/'logs/run-manifest.json').read_text());r={'base':str(b),'run_id':run['run_id'],'source_commit':run['source_commit'],'checkpoints':{}}
for step in [100,150]:
 p=b/'checkpoints/search-g1-common8/actor'/('global_step_'+str(step));r['checkpoints'][step]={'exists':p.is_dir(),'hf_files':[{ 'name':x.name,'bytes':x.stat().st_size} for x in p.glob('*.safetensors')],'config':(p/'config.json').exists(),'generation_config':(p/'generation_config.json').exists(),'complete':(p/'complete.json').exists()}
for n in ['data/manifest.json','logs/service-ready.json']:
 p=b/n
 if p.is_file():r[n]=json.loads(p.read_text()) if n.startswith('data') else {'exists':True}
root=Path('/data/user/jxu957/experiments/easy2560-corollout-joint4-qwen7b-milestone100-150-native2-eval-20261025');r['existing_eval_root']={'exists':root.exists(),'files':[str(x.relative_to(root)) for x in root.glob('*')] if root.exists() else []}
r['own_queue']=subprocess.run(['squeue','-u','jxu957','-h','-o','%i|%j|%T|%P|%D|%b|%R'],capture_output=True,text=True).stdout
print(json.dumps(r))
