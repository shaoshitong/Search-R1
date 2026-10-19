import sys,subprocess,json,pathlib,shlex,time
sys.path.insert(0,'/Users/shaoshitong/Documents/remote_codex_control');import control
code=r'''import json,pathlib,time,subprocess,os
root=pathlib.Path('/data/shaoshitong/grounding-eval-20261019/assets/easy-selection-iter200');s=json.loads((root/'asset-status.json').read_text());out={'at':time.time(),'status':s,'process':subprocess.run(['ps','-p',str(s['pid']),'-o','pid,ppid,etimes,args'],capture_output=True,text=True,timeout=5).stdout,'files':[]}
for q in root.iterdir():
 if q.is_file():out['files'].append({'name':q.name,'size':q.stat().st_size})
if(root/'asset-migration-complete.json').exists():out['complete']=json.loads((root/'asset-migration-complete.json').read_text())
if s['status']=='FAILED_PARTIALS_PRESERVED':out['sanitized_log_tail']=(root/'asset-fetch.log').read_text()[-2000:]
print(json.dumps(out))'''
r=subprocess.run(control.route('baai-a800','python3 -B -c '+shlex.quote(code)),capture_output=True,text=True,timeout=20);assert r.returncode==0;x=json.loads(r.stdout);p=pathlib.Path('outputs/grounding-eval-20261019/easy-a800-asset-observation-'+str(time.time_ns())+'.json');p.write_text(json.dumps(x,indent=2)+'\n');print(json.dumps({'audit':str(p),'at':x['at'],'status':x['status']['status'],'observed_bytes':x['status']['observed_bytes'],'file_status':x['status']['files'],'process':x['process']}))
