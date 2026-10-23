import sys,json,shlex,time,importlib.util
from pathlib import Path
w=Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-6')
spec=importlib.util.spec_from_file_location('campus_ctl',w/'work/start_cosearch_flash_hf.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
remote="""import json,time,os
from pathlib import Path
root=Path('/backup/shaoshitong/easy2560-corollout-hpc3-4gpu-20261017');job='750491'
r={'epoch':time.time(),'job':job,'controllers':{}}
for name,base,pid,start in [('core',root/'archive-controller'/('job-'+job),2001174,1050855824),('companion',root/'post-callback-boundary-control'/('job-'+job)/'future-100-200',2001175,1050855826)]:
 x={'records':{},'pid':pid,'expected_start_ticks':start};proc=Path('/proc')/str(pid)
 try:
  stat=(proc/'stat').read_text().rsplit(')',1)[1].split();x.update(state=stat[0],actual_start_ticks=int(stat[19]),same_process_birth=int(stat[19])==start)
 except FileNotFoundError:x['process_exited']=True
 for n in ('claim.json','progress.json','failure.json','result.json','waiting.json'):
  p=base/n
  if p.is_file():
   try:x['records'][n]=json.loads(p.read_text())
   except json.JSONDecodeError:x['records'][n]={'partial':True}
 r['controllers'][name]=x
print(json.dumps(r),flush=True)
"""
r=m.campus('hkust3',shlex.join(['python3','-B','-c',remote]),observation_timeout=35)
p=w/'outputs/easy2560-qwen7b-20261021'/('combined-batch-search-stop-hk-archive-observe-'+str(time.time_ns())+'.json');p.write_text(json.dumps({'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr},indent=2)+'\n');print(p)
if r.returncode==0:
 v=json.loads(r.stdout);print(json.dumps({'epoch':v['epoch'],'controllers':{k:{a:b for a,b in x.items() if a!='records'}|{'record_names':list(x['records']),'failure':x['records'].get('failure.json'),'progress':x['records'].get('progress.json')} for k,x in v['controllers'].items()}}))
else:print(r.stderr[-1800:])
sys.exit(r.returncode)
