"""Stage a source8-independent path wrapper; optionally inspect paths CPU-only.

Never starts/stops/resumes a trainer or replaces its live mounts/UUID leases.
Inspection is for retained-input recovery readiness, not a training gate.
"""
from pathlib import Path
import argparse
import datetime
import hashlib
import importlib.util
import json
import subprocess

ROOT = Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control')
REPO = Path('/Users/shaoshitong/Documents/Search-G1/h100-prefix5-reference-replica-20261018')
p = argparse.ArgumentParser()
p.add_argument('action', choices=['stage', 'inspect'])
a = p.parse_args()
wrapper = (REPO / 'reproduction/operations/h100_reference_replica/h10011_retained_paths.sh').read_text()
wrapper_sha = hashlib.sha256(wrapper.encode()).hexdigest()
arms = json.loads((ROOT / 'work/prefix5-ab-assigned-arms.json').read_text())
inner = r'''
import json,hashlib,time
from pathlib import Path
ARMS=__ARMS__
NAS=Path('/mnt/data/sihe-share5/workspace/training_data_node_10/shaoshitong')
ROOT=NAS/'search-g1-prefix5-h10011-retained-fixed-assets-20261018'
PROOF=ROOT/'retained-paths-cpu-once.json'
if PROOF.exists():
 raw=PROOF.read_bytes();print(json.dumps({'existing_proof':json.loads(raw),'sha256':hashlib.sha256(raw).hexdigest()}));raise SystemExit(0)
def sha_small(path):
 raw=Path(path).read_bytes();return hashlib.sha256(raw).hexdigest()
def mount_for(path):
 p=str(Path(path).resolve());matches=[]
 for l in Path('/proc/self/mountinfo').read_text().splitlines():
  before,after=l.split(' - ',1);b=before.split();v=after.split();target=b[4]
  if p==target or p.startswith(target.rstrip('/')+'/'):matches.append((len(target),{'target':target,'root':b[3],'filesystem':v[0],'source':v[1]}))
 return max(matches,key=lambda x:x[0])[1]
keys=['canonical_train','validation_questions','final_questions','common_manifest_path','composite_manifest_path','source_question_pool','fixed_test_identity_path','grounding_manifest','grounding_calibration_questions','probe_source_checkpoint','probe_source_cpu_proof','retrieval_corpus','retrieval_index','retrieval_data_cache','official_model_dir','retrieval_encoder','environment_root','retriever_gpu_lib']
rows=[]
for arm,entry in ARMS.items():
 config=Path(entry['base'])/'preparation/launch-config.json';c=json.loads(config.read_text())
 for key in keys:
  path=Path(c[key]);assert path.exists(),(arm,key,str(path));m=mount_for(path)
  assert m['filesystem']=='nfs4' and m['source'].split(':',1)[0] in ['h100-10','10.32.0.18'],(str(path),m)
  st=path.stat();rows.append({'arm':arm,'role':key,'path':str(path),'bytes':st.st_size if path.is_file()else None,'mount':m})
 # Grounding manifest paths must resolve in the same retained view.
 g=json.loads(Path(c['grounding_manifest']).read_text())
 for key in ['boundary','commit']:
  path=Path(g[key]);assert path.is_file();assert mount_for(path)['source'].split(':',1)[0]in ['h100-10','10.32.0.18']
retention=ROOT/'cache2-RESULT.json'
record=json.loads(retention.read_text());counts={};retained_bytes=0
old_prefix='/mnt/shared/workspace/training_data_node_10'
new_prefix='/mnt/data/sihe-share5/workspace/training_data_node_10'
def retained_path(value):
 assert value.startswith(old_prefix+'/')
 return Path(new_prefix+value[len(old_prefix):])
for row in record['files']:
 path=retained_path(row['target']);assert path.is_file() and not path.is_symlink()
 kind=row['kind'];counts[kind]=counts.get(kind,0)+1
 if kind=='NEW_COPY_SOURCE_TARGET_FULL_SHA_PASS':
  assert row['source_sha256']==row['destination_sha256'] and len(row['destination_sha256'])==64
  size=row['bytes'];assert path.stat().st_size==size
 else:
  assert kind=='HARDLINK_TO_EXISTING_IMMUTABLE_FULL_SHA_BACKUP'
  original=retained_path(row['retained_payload']);a,b=path.stat(),original.stat()
  assert (a.st_dev,a.st_ino)==(b.st_dev,b.st_ino)
  size=row['identity'][2];assert a.st_size==size and len(row['proven_sha256'])==64
 retained_bytes+=size
proof={'retention_rows_verified':len(record['files']),'retention_kind_counts':counts,'retained_payload_bytes':retained_bytes,'status':'RETAINED_FIXED_PATHS_CPU_ONLY_PASS','checked_unix':time.time(),'wrapper_sha256':__WRAPPER_SHA__,'retention_proof_sha256':sha_small(retention),'canonical_paths':rows,'training_handoff_performed':False,'live_runs_source_independent':False,'live_uuid_lock_paths_unchanged':True,'model_forward_performed':False,'new_large_file_hashes_performed':False,'model_bytes_via_Mac':False}
with PROOF.open('x')as f:json.dump(proof,f,indent=2)
raw=PROOF.read_bytes();print(json.dumps({'proof':proof,'path':str(PROOF),'sha256':hashlib.sha256(raw).hexdigest()}))
'''.replace('__ARMS__', repr(arms)).replace('__WRAPPER_SHA__', repr(wrapper_sha))
remote = r'''
from pathlib import Path
import json,hashlib,subprocess,os
WRAPPER=__WRAPPER__;SHA=__SHA__;ACTION=__ACTION__
folder=Path.home()/'.local/share/search-g1/prefix5-retained-bootstrap'/SHA
folder.mkdir(parents=True,exist_ok=True);p=folder/'h10011_retained_paths.sh'
if p.exists():assert hashlib.sha256(p.read_bytes()).hexdigest()==SHA
else:
 with p.open('x')as f:f.write(WRAPPER)
p.chmod(0o700)
r={'status':'STAGED_NOT_ACTIVATED','wrapper':str(p),'wrapper_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'live_training_changed':False}
if ACTION=='inspect':
 q=subprocess.run(['unshare','--mount','--propagation','private','bash',str(p),'/usr/bin/python3','-c',__INNER__],env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),text=True,capture_output=True,timeout=100)
 r.update(exit_code=q.returncode,stdout=q.stdout,stderr=q.stderr)
print(json.dumps(r))
'''.replace('__WRAPPER__', repr(wrapper)).replace('__SHA__', repr(wrapper_sha)).replace('__ACTION__', repr(a.action)).replace('__INNER__', repr(inner))
compile(inner, 'retained_metadata_inspection', 'exec')
compile(remote, 'retained_remote_control', 'exec')
spec = importlib.util.spec_from_file_location('ctl', '/Users/shaoshitong/Documents/remote_codex_control/control.py')
ctl = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ctl)
r = subprocess.run(ctl.route('h100-11', '/usr/bin/python3 -'), input=remote,
                   text=True, capture_output=True, timeout=120)
assert r.returncode == 0, r.stderr
d = json.loads(r.stdout)
tag = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
out = ROOT / ('work/prefix5-retained-bootstrap-' + a.action + '-' + tag + '.json')
out.write_text(json.dumps(d, indent=2) + '\n')
print('actual', str(out), 'SHA', hashlib.sha256(out.read_bytes()).hexdigest())
print(json.dumps(d))
