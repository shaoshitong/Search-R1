from pathlib import Path
import json,hashlib,datetime,subprocess,re,time
O={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'epoch':time.time(),'read_only':True,'runs':[]}
roots=['/data/user/jxu957/experiments/qprobe-teacher-warmup200-sigma3-20261015','/data/user/jxu957/experiments/cosearch-continue300-x203-20261008/runtime/cosearch-final200-four-gpu-20261015','/data/user/jxu957/experiments/cosearch-continue300-x203-20261008/runtime/cosearch-instruct200-common-em-20261011']
def read(p):
 p=Path(p)
 if not p.is_file() or p.stat().st_size>2000000:return None
 raw=p.read_bytes()
 try:d=json.loads(raw)
 except Exception:return None
 if isinstance(d,dict):
  for k in ('question_ids','data_sources','rows'):
   if isinstance(d.get(k),list):d.pop(k)
 return {'path':str(p),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'mtime':p.stat().st_mtime,'data':d}
for root in roots:
 r=Path(root);o={'root':root,'exists':r.exists(),'files':[],'train':[],'fixed512':[],'logs':[],'named':[]}
 if not r.exists():O['runs'].append(o);continue
 if 'sigma3' in root:
  p=r/'logs/actual-updates.jsonl';raw=p.read_bytes();complete=raw if raw.endswith(b'\n') else raw[:raw.rfind(b'\n')+1]
  us=[json.loads(l) for l in complete.splitlines()];o['updates_proof']={'path':str(p),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'mtime':p.stat().st_mtime,'incomplete_tail_bytes':len(raw)-len(complete)}
  o['updates']=us;o['completed_iter']=max([u['step'] for u in us]+[0]);o['head_count']=len(list(r.glob('artifacts/question-probe/completed/*.json')))
  for u in us:
   p=r/'artifacts/question-probe/attempts'/u['attempt_id']/'rollout-audit.json';v=read(p)
   if v:
    m=v['data'].get('prequential',{}).get('all',{});o['train'].append({'iter':u['step'],'em':m.get('actual_em_mean'),'n':m.get('questions'),'reward':u.get('metrics',{}).get('critic/rewards/mean'),'path':str(p),'sha256':v['sha256']})
  paths=set(r.glob('logs/*development*.json'))|set(r.glob('logs/*stage512*.json'))|set(r.glob('logs/*validation*.json'))
  for p in r.glob('artifacts/question-probe/completed-pairs/*/source-receipt.json'):
   v=read(p)
   if not v:continue
   for k,d in v['data'].get('historical_artifacts',{}).get('development',{}).items():
    q=read(d['path'])
    if q:paths.add(Path(d['path']))
  for p in paths:
   v=read(p)
   if not v:continue
   d=v['data'];m=d.get('metrics',d);n=m.get('val/qa_count/all',d.get('question_count',d.get('qa_count',d.get('rows'))));em=m.get('val/em/all',d.get('weighted_em',d.get('em')))
   if n==512 and em is not None:o['fixed512'].append({'iter':d.get('step',d.get('outer_step',d.get('actual_completed_outer'))),'em':em,'f1':m.get('val/f1/all',d.get('weighted_f1',d.get('f1'))),'n':512,'correct':d.get('correct_count'),'path':str(p),'sha256':v['sha256']})
  p=r/'logs/validation-identities.jsonl'
  if p.exists():
   raw=p.read_bytes();o['validation_ledger_proof']={'path':str(p),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()};o['validation_ledger']=[]
   for line_index,l in enumerate(raw.splitlines()):
    try:d=json.loads(l)
    except Exception:continue
    ids=d.get('question_ids',[]);sources=d.get('data_sources',[]);em=d.get('em',[]);f1=d.get('f1',[])
    n=len(ids);counts=d.get('counts',{});row={'iter':d['step'],'split':d.get('split'),'final':d.get('final'),'n':n,'correct':d.get('correct'),'em':d.get('weighted_em'),'f1':d.get('weighted_f1'),'counts':counts,'source_em':{s:sum(e for e,t in zip(em,sources) if t==s)/count for s,count in counts.items()},'source_f1':{s:sum(e for e,t in zip(f1,sources) if t==s)/count for s,count in counts.items()},'path':str(p),'line_index':line_index,'row_sha256':hashlib.sha256(l).hexdigest(),'checked_unix':d.get('checked_unix'),'run_id':d.get('run_id'),'validation_sha256':d.get('validation_sha256'),'scalar_readback_correct':sum(em)==d.get('correct'),'unique_question_ids':len(set(ids))}
    o['validation_ledger'].append(row)
    if n==512 and row['em'] is not None:o['fixed512'].append(dict(row,sha256=o['validation_ledger_proof']['sha256']))
  for p in r.glob('logs/optimizer-rank-*.jsonl'):
   with p.open('rb') as f:f.seek(max(0,p.stat().st_size-32000));tail=f.read().splitlines()
   rows=[]
   for l in tail:
    try:rows.append(json.loads(l))
    except Exception:pass
   o.setdefault('rank_optimizer',[]).append({'path':str(p),'bytes':p.stat().st_size,'mtime':p.stat().st_mtime,'latest':rows[-1] if rows else None})
  for p in r.glob('logs/*.log'):
   if p.name.startswith(('model-','retriev','bootstrap','server')):continue
   with p.open('rb') as f:f.seek(max(0,p.stat().st_size-64000));tail=f.read().decode(errors='replace')
   o['logs'].append({'path':str(p),'bytes':p.stat().st_size,'mtime':p.stat().st_mtime,'tail':tail[-1400:],'errors':[l[:1000] for l in tail.splitlines() if re.search('Traceback|CUDA out of memory|AssertionError|RuntimeError:|SEARCH_G1_FATAL',l)][-4:]})
  p=r/'logs/search-g1-common8-training.log'
  if p.exists():
   raw=p.read_bytes();o['log_eval_proof']={'path':str(p),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
   for line in p.open(errors='replace'):
    if 'val/' not in line or 'step:' not in line:continue
    m={k:float(v) for k,v in re.findall(r'([\w/.-]+):\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)',line)};s=re.search(r'\bstep:\s*(\d+)',line)
    if s:o.setdefault('log_eval',[]).append({'iter':int(s[1]),'metrics':{k:v for k,v in m.items() if k.startswith('val/')}})
 if 'instruct200' in root:
  for p in r.glob('common-scoring-v4/common-training/*/completed-outer-*.json'):
   d=json.loads(p.read_text());m=d['all_actual_generated_responses'];o['train'].append({'iter':d['step'],'em':m['em'],'correct':m['correct'],'job':p.parent.name,'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
  for p in r.glob('common-scoring-v4/common-validation/*.receipt.json'):
   v=read(p);d=v['data'];o['fixed512'].append({'iter':d['step'],'em':d['em'],'f1':d['f1'],'n':d['questions'],'path':str(p),'sha256':v['sha256']})
 paths=set()
 for pattern in ('acceptance.json','final-completion-report.json','recovery-plan.json','binding.json','*summary*.json','*complete*.json','*metrics*.json','*result*.json','progress.json','controller-state.json','logs/pipeline-status.json','logs/*stage512*.json','logs/*development*.json','results/*summary*.json','results/*metrics*.json','final/*summary*.json','evaluation/*summary*.json'):
  paths.update(r.glob(pattern))
 for p in paths:
  v=read(p)
  if v:o['files'].append(v)
 for pat in ('*','logs/*','results/*','evaluation/*'):
  o['named'] += [{'path':str(p.relative_to(r)),'bytes':p.stat().st_size if p.is_file() else None} for p in r.glob(pat) if p.name not in ('repo','source','env','cache','ray','question-features')][:160]
 O['runs'].append(o)
for argv in (['squeue','-h','-j','721485,721542,721339','-o','%i|%T|%N|%M'],['sacct','-nP','-X','-j','721485,721542,721339','--format=JobID,State,ExitCode,NodeList']):
 p=subprocess.run(argv,stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=25);O[str(argv)]={'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
print(json.dumps(O))
