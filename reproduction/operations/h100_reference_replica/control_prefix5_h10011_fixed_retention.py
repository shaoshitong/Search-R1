"""Explicit once start or metadata-only status of retained fixed-input copying."""
from pathlib import Path
import argparse,datetime,hashlib,importlib.util,json,subprocess
ROOT=Path(__file__).resolve().parents[1]
REPO=Path('/Users/shaoshitong/Documents/Search-G1/h100-prefix5-reference-replica-20261018')
p=argparse.ArgumentParser();p.add_argument('action',choices=['start','status']);p.add_argument('--cache-retry',action='store_true');a=p.parse_args()
spec=importlib.util.spec_from_file_location('ctl','/Users/shaoshitong/Documents/remote_codex_control/control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl)
base='/mnt/shared/workspace/training_data_node_10/shaoshitong/search-g1-prefix5-h10011-retained-fixed-assets-20261018'
source_root='/mnt/shared/workspace/training_data_node_8/shaoshitong/'
inventory=json.loads((ROOT/'work/prefix5-h10011-retained-assets-inspect-20261017T215645873050Z.json').read_text())
worker=(REPO/'reproduction/operations/h100_reference_replica/retain_fixed_assets.py').read_text();worker_sha=hashlib.sha256(worker.encode()).hexdigest()
rows=[]
for r in sorted(inventory['source_inventory'],key=lambda x:(x['role'],x['relative'])):
 if r['path'].endswith('.lock') or (a.cache_retry and r['role']!='cache'):continue
 assert r['path'].startswith(source_root) and not r['symlink']
 row=dict(source=r['path'],target_relative='shaoshitong/'+r['path'][len(source_root):],identity=[r['device'],r['inode'],r['bytes'],r['mtime_ns']])
 if r['relative']=='wiki_dump.jsonl':row['expected_sha256']='43d7d3f58d01d711d95b00b70584211eea639fa46802905a4b7e11cf0617752d'
 rows.append(row)
proof=inventory['backup_metadata']['RESULT.json']
reused=[]
for r in proof['data']['files']:
 if r['roles'] in [['retrieval_index'],['retrieval_corpus']]:
  assert r['original_path'].startswith('/mnt/data/shaoshitong/')
  reused.append(dict(proof_path=proof['path'],proof_sha256=proof['sha256'],relative_path=r['relative_path'],target_relative='shaoshitong/'+r['original_path'][len('/mnt/data/shaoshitong/'):]))
assert len(reused)==2
config=dict(destination=base,bytes_per_second=20*1024**2,files=rows,reused_backup_payloads=[]if a.cache_retry else reused,record_prefix='cache2-'if a.cache_retry else '')
script=r'''import json,os,time,hashlib,subprocess
from pathlib import Path
B=Path(__BASE__);ACTION=__ACTION__;PREFIX=__PREFIX__
def rec(name):return B/(PREFIX+name)
def status():
 out={'checked_unix':time.time(),'base':str(B),'files':{}}
 for name in ['launch-intent.json','launch-receipt.json','INTENT.json','PROGRESS.json','RESULT.json']:
  p=rec(name)
  if p.exists():
   data=p.read_bytes();d=json.loads(data)
   if name=='RESULT.json':d={k:v for k,v in d.items()if k!='files'};d['all_payload_rows']=len(json.loads(data)['files'])
   out['files'][name]={'data':d,'sha256':hashlib.sha256(data).hexdigest()}
 r=out['files'].get('launch-receipt.json',{}).get('data');out['worker_same_birth_now']=False
 if r:
  p=Path('/proc/'+str(r['pid'])+'/stat')
  try:
   fields=p.read_text().rsplit(')',1)[1].split();out['worker_same_birth_now']=int(fields[19])==r['start_ticks'] and fields[0]!='Z';out['worker_state']=fields[0]
  except FileNotFoundError:pass
 log=rec('worker.log')
 if log.exists():out['worker_log_tail']=log.read_text()[-1600:]
 return out
if ACTION=='start':
 B.mkdir(parents=True,exist_ok=True)
 if not rec('launch-intent.json').exists():
  WORKER=__WORKER__;SHA=__WORKER_SHA__;CONFIG=__CONFIG__
  assert hashlib.sha256(WORKER.encode()).hexdigest()==SHA
  c=B/'control'/SHA;c.mkdir(parents=True,exist_ok=True)
  script=c/'retain_fixed_assets.py';script.write_text(WORKER);assert hashlib.sha256(script.read_bytes()).hexdigest()==SHA
  if PREFIX:
   previous=json.loads((B/'launch-receipt.json').read_text());q=Path('/proc/'+str(previous['pid'])+'/stat')
   assert not q.exists() or int(q.read_text().rsplit(')',1)[1].split()[19])!=previous['start_ticks'], 'Old copy must be terminal'
   prior=B/'completed-files.json';data=prior.read_bytes();CONFIG['prior_completed_manifest']={'path':str(prior),'sha256':hashlib.sha256(data).hexdigest()}
  cp=c/(PREFIX+'spec.json');cp.write_text(json.dumps(CONFIG,indent=2)+'\n')
  intent={'worker_sha256':SHA,'spec_sha256':hashlib.sha256(cp.read_bytes()).hexdigest(),'requested_unix':time.time(),'live_training_changed':False}
  with rec('launch-intent.json').open('x')as f:json.dump(intent,f);f.flush();os.fsync(f.fileno())
  with rec('worker.log').open('xb')as f:
   proc=subprocess.Popen(['nice','-n','10','/usr/bin/python3',str(script),'--spec',str(cp)],stdin=subprocess.DEVNULL,stdout=f,stderr=subprocess.STDOUT,start_new_session=True,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''))
  fields=Path('/proc/'+str(proc.pid)+'/stat').read_text().rsplit(')',1)[1].split()
  receipt=dict(intent,pid=proc.pid,start_ticks=int(fields[19]))
  with rec('launch-receipt.json').open('x')as f:json.dump(receipt,f);f.flush();os.fsync(f.fileno())
print(json.dumps(status()))
'''.replace('__BASE__',repr(base)).replace('__ACTION__',repr(a.action)).replace('__WORKER_SHA__',repr(worker_sha)).replace('__WORKER__',repr(worker)).replace('__CONFIG__',repr(config)).replace('__PREFIX__',repr('cache2-'if a.cache_retry else ''))
compile(script,'retention_remote_control','exec')
r=subprocess.run(ctl.route('h100-11','/usr/bin/python3 -'),input=script,text=True,capture_output=True,timeout=70);assert r.returncode==0,r.stderr;d=json.loads(r.stdout);stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');out=ROOT/('work/prefix5-h10011-fixed-retention-'+('cache2-'if a.cache_retry else '')+a.action+'-'+stamp+'.json');out.write_text(json.dumps(d,indent=2)+'\n');print('actual',out,'sha256',hashlib.sha256(out.read_bytes()).hexdigest());print(json.dumps(d))
