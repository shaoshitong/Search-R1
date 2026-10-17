# Archived actual operation. Reconcile existing intent/receipt before any use.
import json,subprocess,concurrent.futures,datetime,importlib.util
from pathlib import Path
spec=importlib.util.spec_from_file_location('ctl','/Users/shaoshitong/Documents/remote_codex_control/control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl)
arms=json.loads(Path('work/prefix5-ab-assigned-arms.json').read_text())
stop=r'''import json,os,signal,datetime,time
from pathlib import Path
ARMS=__ARMS__
out={}
for slot,a in ARMS.items():
 b=Path(a['base']);r=json.loads((b/'preparation/submit-receipt.json').read_text());leader=r['leader'];pid=leader['pid']
 proof=b/'preparation/migrate-h10011-human-stop.json'
 if not proof.exists():
  record={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'human_instruction':'那就在那个空闲最多的上面跑呗','scope':'Only own A/B native processes on h100-8; migrate to h100-11; preserve outputs and goal','leader':leader,'joint_at_request':json.loads((b/'latest-joint.json').read_text()).get('outer',0) if (b/'latest-joint.json').exists() else 0}
  with proof.open('x')as f:json.dump(record,f)
  try:
   fd=os.pidfd_open(pid);s=Path('/proc/%d/stat'%pid).read_text().rsplit(')',1)[1].split();assert int(s[19])==leader['start_ticks'];signal.pidfd_send_signal(fd,signal.SIGTERM);os.close(fd);record['signal_sent']=True
  except ProcessLookupError:record['already_gone']=True
  out[slot]=record
 else:out[slot]={'existing_stop_request':str(proof),'replayed':False}
print(json.dumps(out))
'''.replace('__ARMS__',repr(arms))
inspect=r'''import os,json,socket,subprocess,datetime
from pathlib import Path
paths=['/dataset/shaoshitong/Search-G1-common8/env/bin/python','/dataset/shaoshitong/Search-G1-common8/bootstrap-assets/instruct300/assets/qwen2.5-3b-instruct-aa8e725','/mnt/data/shaoshitong/search-g1-prefix5-reference-assets-20261018-model-only','/mnt/data/sihe-share5/workspace/training_data_node_10/shaoshitong','/mnt/shared/workspace/training_data_node_10/shaoshitong','/mnt/data/shaoshitong/search-r1-rl/gigpo/assets-root/assets/wiki/e5_Flat.index','/dataset/shaoshitong/Search-G1-common8/retriever-gpu-lib','/mnt/data/shaoshitong/search-g1-prefix20-online1000-20261016-ca1c60f/retrieval-cache']
def run(a):
 p=subprocess.run(a,text=True,capture_output=True,timeout=15);return {'rc':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
print(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'host':socket.gethostname(),'paths':{s:{'exists':Path(s).exists(),'realpath':os.path.realpath(s),'bytes':Path(s).stat().st_size if Path(s).exists() else None}for s in paths},'gpu':run(['nvidia-smi','--query-gpu=index,uuid,memory.free,utilization.gpu','--format=csv,noheader,nounits']),'mounts':run(['findmnt','-t','nfs,nfs4','-o','TARGET,SOURCE,FSTYPE']),'disk':run(['df','-B1','/mnt/data','/dataset']), 'ports':run(['ss','-ltn'])}))
'''
def do(h,s,label):
 p=subprocess.run(ctl.route(h,'/usr/bin/python3 -'),input=s,text=True,capture_output=True,timeout=65)
 tag=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');v={'target':h,'rc':p.returncode,'stdout':p.stdout,'stderr':p.stderr};Path('work/prefix5-ab-'+label+'-'+tag+'.json').write_text(json.dumps(v,indent=2));print(label,json.dumps(v))
with concurrent.futures.ThreadPoolExecutor(max_workers=2)as p:
 list(p.map(lambda x:do(*x),[('h100-8',stop,'migrate11-stop-request'),('h100-11',inspect,'migrate11-target-inspect')]))
