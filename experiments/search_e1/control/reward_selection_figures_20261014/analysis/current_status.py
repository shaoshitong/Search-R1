from pathlib import Path
import sys,subprocess,shlex,json,datetime
D=Path(__file__).resolve().parent;controlbase=Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control');sys.path.insert(0,str(controlbase/'work'));import search_g1_restore5_transport as t
# Existing retained18 auth/host checks; the outer task-only compatibility options are from current connection memory.
original=subprocess.run
mode=sys.argv[-1]
def transport(args,*a,**kw):
 if isinstance(args,list) and 'mgmt-host' in args:
  args=list(args);option='ProxyCommand=/usr/bin/nc -X 5 -x 127.0.0.1:7892 %h %p'
  if option in args:
   i=args.index(option);del args[i-1:i+1]
  j=args.index('mgmt-host')
  opts=[] if mode=='plain' else ['-o','KexAlgorithms=curve25519-sha256','-o','IPQoS=none']
  if mode=='bound':
   source=original(['/usr/sbin/ipconfig','getifaddr','en0'],capture_output=True,text=True,check=True).stdout.strip();proxy='/Users/shaoshitong/Documents/Codex/2026-10-03/xia/work/recursive-bran-experiment-a-20261011/release/recursive_bran_a/h100_migration/source_bound_proxy.py';opts+=['-o','ProxyCommand='+shlex.join([sys.executable,proxy,'--source',source,'--host','52.6.216.152','--port','22'])]
  elif mode!='plain':opts+=['-o','ProxyCommand=none','-o','BindInterface=en0']
  args[j:j]=opts
 return original(args,*a,**kw)
t.subprocess.run=transport
code='''from pathlib import Path
import json,hashlib,subprocess,datetime
r=Path('/mnt/data/shaoshitong/search-g1-speed-resume5-0553ade-20261012');o={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'root':str(r),'files':{},'final_files':[],'gpu':subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.total,utilization.gpu','--format=csv,noheader'],text=True,capture_output=True).stdout,'apps':subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_memory','--format=csv,noheader'],text=True,capture_output=True).stdout}
keep=['run_id','step','split','stage','status','phase','correct','weighted_em','weighted_f1','counts','final_counts','final_correct','final_weighted_em','final_weighted_f1','structural_accepted','actual_outer_updates','validation_sha256']
for n in ['full-run-acceptance.json','latest-validation-identity.json','final-identity.json','pipeline-status.json']:
 p=r/'logs'/n
 if p.exists():
  raw=p.read_bytes();v=json.loads(raw);o['files'][n]={'sha256':hashlib.sha256(raw).hexdigest(),'keys':list(v),'scalars':{k:v[k] for k in keep if k in v},'ordered_identity_sha256':{k:hashlib.sha256(json.dumps(v[k],ensure_ascii=False,separators=(',',':')).encode()).hexdigest() for k in ['question_ids','data_sources'] if k in v}}
for p in (r/'logs').iterdir():
 if 'final' in p.name:o['final_files'].append({'name':p.name,'size':p.stat().st_size})
print(json.dumps(o))'''
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%H%M%S%f')
try:
 raw=t.remote('/usr/bin/python3 -c '+shlex.quote(code),str(D/('transport-'+mode+'-'+stamp+'.json')),emit=False);v=json.loads(raw);(D/'mixed-current.json').write_text(json.dumps(v,ensure_ascii=False,indent=2));print(json.dumps(v))
except Exception as e: print('Connection failed via '+mode+': '+str(e)[-450:]);sys.exit(1)
