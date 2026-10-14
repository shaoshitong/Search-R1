from pathlib import Path
import json,sys,subprocess,shlex,concurrent.futures
from h100_remote import call
sys.path.insert(0,'/Users/shaoshitong/Documents/remote_codex_control');import control
D=Path(__file__).resolve().parent
code='''from pathlib import Path
import json,hashlib,datetime,re
roots=ROOTS;o={'observed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'series':[]}
for name,root in roots:
 r=Path(root);p=r/'logs/actual-updates.jsonl';rows=[]
 for l in p.open():
  v=json.loads(l);m=v.get('metrics',{});x={'step':v['step'],'run_id':v['run_id']}
  for k in ['critic/rewards/mean','critic/scores/mean','train/perf/em']:x[k]=m.get(k)
  rows.append(x)
 o['series'].append({'name':name,'source':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'rows':rows})
print(json.dumps(o))'''
def a800():
 roots=[('Soft Reward','/data/shaoshitong/Search-G1-a800-fresh200-a20f0475ec824c69ba1554f39e9ff512'),('Soft Reward -> Hard Reward','/data/shaoshitong/Search-G1-a800-fresh200-5e1f65d8e7d44363b02938c8d19852ab')];a=control.route('baai-a800','python3 -B -c '+shlex.quote(code.replace('ROOTS',repr(roots))));a[1:1]=['-o','UpdateHostKeys=no'];v=subprocess.run(a,text=True,capture_output=True,timeout=90);assert v.returncode==0,v.stderr[-800:];return json.loads(v.stdout)
def h100():
 return call('h100-8',code.replace('ROOTS',repr([('Easy Selection+Soft Reward','/mnt/data/shaoshitong/search-g1-speed-resume5-0553ade-20261012')])),90)
with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:values=list(pool.map(lambda f:f(),[a800,h100]))
z={'observed_utc':[v['observed_utc'] for v in values],'series':sum([v['series'] for v in values],[])};(D/'native-reward-means.json').write_text(json.dumps(z,indent=2));print([(s['name'],len(s['rows']),[(v['step'],v['critic/rewards/mean']) for v in s['rows'] if v['step'] in [50,100,150,200]]) for s in z['series']])
