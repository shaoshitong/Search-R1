from pathlib import Path
import sys,subprocess,shlex,json,concurrent.futures
D=Path(__file__).resolve().parent
sys.path.insert(0,'/Users/shaoshitong/Documents/remote_codex_control');import control
CODE=r'''from pathlib import Path
import json,hashlib,datetime,math
r=Path(ROOT);o={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'root':str(r),'fixed512':[],'extra':[]}
p=r/'logs/validation-identities.jsonl'
if p.exists():
 raw=p.read_bytes();o['ledger']={'path':str(p),'sha256':hashlib.sha256(raw).hexdigest(),'bytes':len(raw)}
 for line in raw.splitlines():
  d=json.loads(line);n=sum(d.get('counts',{}).values());m=d.get('metrics',{});n=m.get('val/qa_count/all',d.get('evaluated_rows',n))
  if n==512:o['fixed512'].append({'iter':d.get('step',d.get('actual_completed_outer')),'n':n,'em':d.get('weighted_em',m.get('val/em/all')),'f1':d.get('weighted_f1',m.get('val/f1/all')),'correct':d.get('correct'),'source':str(p),'record_sha256':hashlib.sha256(line).hexdigest()})
for rel in ['logs/final-acceptance-repair-result.json','logs/final-acceptance-repair-claim.json']:
 p=r/rel
 if p.exists():
  raw=p.read_bytes();d=json.loads(raw);o['extra'].append({'path':str(p),'sha256':hashlib.sha256(raw).hexdigest(),'data':d})
print(json.dumps(o))'''
def task(name,host,root):
 c='ROOT='+repr(root)+'\n'+CODE
 if host=='baai-a800':p=subprocess.run(control.route(host,'python3 -B -c '+shlex.quote(c)),capture_output=True,text=True,timeout=60)
 else:
  cm=Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control/work/read_search_g1_shared3381_direct_v1.py');p=subprocess.run([sys.executable,str(cm)],input=c,capture_output=True,text=True,timeout=90,cwd=str(cm.parent.parent))
 assert p.returncode==0,p.stderr[-1000:];d=json.loads(p.stdout);(D/(name+'-validation.json')).write_text(json.dumps(d,ensure_ascii=False,indent=2));print(name,json.dumps(d,ensure_ascii=False)[:13000])
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
 fs=[ex.submit(task,*t) for t in [('g1-soft','baai-a800','/data/shaoshitong/Search-G1-a800-fresh200-a20f0475ec824c69ba1554f39e9ff512'),('g1-soft-hard','baai-a800','/data/shaoshitong/Search-G1-a800-fresh200-5e1f65d8e7d44363b02938c8d19852ab'),('g1-mixed','h100-8','/mnt/data/shaoshitong/search-g1-speed-resume5-0553ade-20261012')]]
 for f in fs:
  try:f.result()
  except Exception as e:print(str(e))
