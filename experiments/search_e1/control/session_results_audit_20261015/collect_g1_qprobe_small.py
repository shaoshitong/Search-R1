from pathlib import Path
import sys,json,subprocess,shlex
D=Path(__file__).resolve().parent
sys.path.insert(0,'/Users/shaoshitong/Documents/remote_codex_control');import control
c=r'''from pathlib import Path
import json,datetime,re,os,hashlib
r=Path('/data/shaoshitong/Search-G1-a800-fresh200-86cf721183014e729036924e8978a6b9');o={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'root':str(r),'logs':[],'json_files':[]}
for rel in ['logs/trainer.log','logs/search-g1-common8-training.log','logs/deployment-pipeline.log']:
 p=r/rel
 if not p.exists():continue
 with p.open('rb') as f:f.seek(max(0,p.stat().st_size-300000));raw=f.read()
 lines=[]
 for l in raw.decode(errors='replace').splitlines():
  l=re.sub(r'\x1b\[[0-9;]*m','',l)
  if any(t in l for t in ['step:','QUESTION_PROBE','QPROBE','Training Progress','teacher_warmup','current_step','rollout_time','PROGRESS=']):lines.append(l[:800])
 o['logs'].append({'path':str(p),'bytes':p.stat().st_size,'mtime':p.stat().st_mtime,'lines':lines[-15:],'tail_sha256':hashlib.sha256(raw).hexdigest(),'traceback_in_tail':'Traceback' in raw.decode(errors='replace')})
for p in (r/'artifacts/question-probe').rglob('*.json'):
 if len(o['json_files'])>=100:break
 if p.stat().st_size>2000000:continue
 raw=p.read_bytes()
 try:d=json.loads(raw)
 except:continue
 
 if isinstance(d.get('groups'),list):
  vec=[x.get('strict_correctness') for x in d['groups']];valid=all(isinstance(v,list) and len(v)==5 and all(z in (0,1) for z in v) for v in vec)
  if valid: d['rollout_strict_em_n']=sum(len(v) for v in vec);d['rollout_strict_em_correct']=sum(sum(v) for v in vec);d['rollout_strict_em']=d['rollout_strict_em_correct']/d['rollout_strict_em_n']
 scalars={k:v for k,v in d.items() if isinstance(v,(float,int,str,bool,type(None))) and len(str(v))<300}
 o['json_files'].append({'path':str(p.relative_to(r)),'bytes':len(raw),'mtime':p.stat().st_mtime,'sha256':hashlib.sha256(raw).hexdigest(),'scalars':scalars})
for rel in ['logs/actual-updates.jsonl','logs/actual-attempts.jsonl','logs/question-probe-iterations.jsonl']:
 p=r/rel
 o.setdefault('ledger_files',[]).append({'path':str(p),'exists':p.exists(),'bytes':p.stat().st_size if p.exists() else None})
o['rank_progress']=[]
for rank in range(4):
 p=r/('logs/optimizer-rank-'+str(rank)+'.jsonl')
 if not p.exists():continue
 raw=p.read_bytes();rows=[json.loads(l) for l in raw.splitlines() if l.strip()];ops=[x for x in rows if x.get('kind')=='OPPORTUNITY'];last=rows[-1] if rows else {}
 o['rank_progress'].append({'rank':rank,'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'row_count':len(rows),'opportunity_rows':len(ops),'last_opportunity':{k:v for k,v in ops[-1].items() if isinstance(v,(float,int,str,bool,type(None))) and len(str(v))<200} if ops else None,'last':{k:v for k,v in last.items() if isinstance(v,(float,int,str,bool,type(None))) and len(str(v))<200}})
print(json.dumps(o))'''
p=subprocess.run(control.route('baai-a800','python3 -B -c '+shlex.quote(c)),capture_output=True,text=True,timeout=60)
assert p.returncode==0,p.stderr[-1000:];d=json.loads(p.stdout);(D/('g1-qprobe-small-'+sys.argv[1]+'.json')).write_text(json.dumps(d,ensure_ascii=False,indent=2));print(json.dumps(d,ensure_ascii=False)[:16000])
