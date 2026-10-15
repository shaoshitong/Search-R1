from pathlib import Path
import json,subprocess,shlex,sys,datetime,concurrent.futures
D=Path(__file__).resolve().parent
sys.path.insert(0,'/Users/shaoshitong/Documents/remote_codex_control');import control
CODE=r'''from pathlib import Path
import json,hashlib,datetime,re,math,os
r=Path(ROOT);o={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'root':str(r),'training':[],'fixed512':[],'final':[],'metadata':[],'errors':[]}
for rel in ['logs/actual-updates.jsonl','logs/actual-outer-updates.jsonl','logs/pipeline-status.json','logs/deployment-pipeline-status.json','logs/run-manifest.json','logs/process-ledger.json','logs/cleanup-status.json']:
 p=r/rel
 if not p.exists():continue
 raw=p.read_bytes();h=hashlib.sha256(raw).hexdigest();v=None
 try:v=json.loads(raw) if p.suffix=='.json' else [json.loads(x) for x in raw.splitlines()]
 except Exception as e:o['errors'].append([str(p),str(e)])
 entry={'path':str(p),'bytes':len(raw),'mtime':p.stat().st_mtime,'sha256':h}
 if isinstance(v,dict):
  entry['scalars']={k:z for k,z in v.items() if isinstance(z,(int,float,str,bool,type(None))) and ('token' not in k.lower()) and len(str(z))<600}
  if rel.endswith('process-ledger.json'):entry['jobs']={k:{n:z.get(n) for n in ['state','exit_code','leader']} for k,z in v.get('jobs',{}).items()}
 if isinstance(v,list):
  entry['row_count']=len(v);entry['last_scalar']={k:z for k,z in v[-1].items() if isinstance(z,(int,float,str,bool,type(None)))} if v else None
  for z in v:
   m=z.get('metrics',{});em=m.get('train/perf/em',m.get('train_em'));i=z.get('step',z.get('outer_step'))
   if em is not None:o['training'].append({'iter':i,'em':em,'reward':m.get('critic/rewards/mean'),'source':str(p)})
 o['metadata'].append(entry)
rows={}
for rel in ['logs/trainer.log','logs/search-g1-common8-training.log','stage.log','trainer.log','train.log']:
 p=r/rel
 if not p.exists():continue
 for l in p.open(errors='replace'):
  s=re.search(r'\bstep:\s*(\d+)',l)
  if not s:continue
  i=int(s[1]);m={k:float(v) for k,v in re.findall(r'([\w/.-]+):\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)',l)}
  if 'train/perf/em' in m:rows[i]={'iter':i,'em':m['train/perf/em'],'reward':m.get('critic/rewards/mean'),'source':str(p)}
 o['metadata'].append({'path':str(p),'bytes':p.stat().st_size,'mtime':p.stat().st_mtime})
if rows:o['training']=list(sorted(rows.values(),key=lambda z:z['iter']))
ps=set()
for pat in ['stage512-step*.json','logs/*identity.json','output/validation-cost/validation-*.json','logs/full-run-acceptance.json','*final*summary*.json']:
 ps.update(r.glob(pat))
for p in sorted(ps):
 if p.stat().st_size>20*1024*1024:continue
 raw=p.read_bytes()
 try:d=json.loads(raw)
 except:continue
 m=d.get('metrics',{});n=d.get('evaluated_rows',d.get('count',sum(d.get('counts',{}).values())));n=m.get('val/qa_count/all',n)
 em=d.get('weighted_em',m.get('val/em/all'));f1=d.get('weighted_f1',m.get('val/f1/all'));step=d.get('actual_completed_outer',d.get('step'))
 if n in [512,51713]:
  z={'path':str(p),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'mtime':p.stat().st_mtime,'iter':step,'n':n,'em':em,'f1':f1,'correct':d.get('correct'),'split':d.get('split',d.get('phase'))}
  if isinstance(d.get('em'),list):z['em_array_n']=len(d['em']);z['em_array_sum']=math.fsum(d['em']);z['binary_em_verified']=all(x in (0,1) for x in d['em']);z['em_array_mean']=math.fsum(d['em'])/len(d['em'])
  if isinstance(d.get('f1'),list):z['f1_array_n']=len(d['f1']);z['f1_array_mean']=math.fsum(d['f1'])/len(d['f1'])
  (o['fixed512'] if n==512 else o['final']).append(z)
 if p.name=='full-run-acceptance.json':o['acceptance']={'path':str(p),'sha256':hashlib.sha256(raw).hexdigest(),'actual_outer_updates':d.get('actual_outer_updates'),'structural_accepted':d.get('structural_accepted'),'final_weighted_em':d.get('final_weighted_em'),'final_weighted_f1':d.get('final_weighted_f1'),'per_rank_actual_optimizer_writes':d.get('per_rank_actual_optimizer_writes')}
print(json.dumps(o))'''
def task(name,host,root):
 c='ROOT='+repr(root)+'\n'+CODE
 if host=='baai-a800':p=subprocess.run(control.route(host,'python3 -B -c '+shlex.quote(c)),capture_output=True,text=True,timeout=90)
 else:
  cm=Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control/work/read_search_g1_shared3381_direct_v1.py')
  p=subprocess.run([sys.executable,str(cm)],input=c,capture_output=True,text=True,timeout=100,cwd=str(cm.parent.parent))
 try:d=json.loads(p.stdout)
 except:d={'error':p.stderr[-1200:],'returncode':p.returncode,'stdout':p.stdout[:500]}
 (D/(name+'.json')).write_text(json.dumps(d,ensure_ascii=False,indent=2));print(name,'rows',len(d.get('training',[])),'last',d.get('training',[])[-1:] ,'fixed',d.get('fixed512'),'final',d.get('final'),'error',d.get('error'),flush=True)
if __name__=='__main__':
 tasks=[('g1-soft','baai-a800','/data/shaoshitong/Search-G1-a800-fresh200-a20f0475ec824c69ba1554f39e9ff512'),('g1-soft-hard','baai-a800','/data/shaoshitong/Search-G1-a800-fresh200-5e1f65d8e7d44363b02938c8d19852ab'),('g1-mixed','h100-8','/mnt/data/shaoshitong/search-g1-speed-resume5-0553ade-20261012')]
 with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
  fs=[ex.submit(task,*t) for t in tasks]
  for f in fs:
   try:f.result()
   except Exception as e:print(type(e).__name__,str(e),flush=True)
