import pathlib,json,time,os,hashlib
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/training-extraG5-matched6528-20261021')
r=dict(time=time.time(),base=str(b))
for n in ['launch.json','controller.json','result.json','failure.json']:
 if (b/n).exists():r[n]=json.loads((b/n).read_text())
c=r.get('controller.json') or r.get('launch.json')
if c:
 p=pathlib.Path('/proc/%d'%c['pid']);r['identity_live']=p.exists() and (p/'stat').read_text().rsplit(')',1)[1].split()[19]==c['start_ticks']
 if r['identity_live']:
  r['io']=(p/'io').read_text();r['status']={x.split(':',1)[0]:x.split(':',1)[1].strip() for x in (p/'status').read_text().splitlines() if x.split(':',1)[0] in ['State','VmRSS','Threads']}
if (b/'prepare.log').exists():
 raw=(b/'prepare.log').read_bytes();r['log_bytes']=len(raw);r['log_sha256']=hashlib.sha256(raw).hexdigest();r['last_error']=[x for x in raw.decode(errors='replace').splitlines() if 'Error' in x or 'Traceback' in x][-3:]
if (b/'materialized').exists():r['inputs_files']=len(list((b/'materialized/inputs').glob('**/*.json')))
print(json.dumps(r))
