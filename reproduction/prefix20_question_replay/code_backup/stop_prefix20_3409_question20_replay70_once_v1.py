"""Once scoped native stop for the current human question20 replay70 replacement."""
from pathlib import Path
import datetime,json,hashlib,sys
W=Path(__file__).parent.resolve();R=Path('/Users/shaoshitong/Documents/Search-G1/qprobe-prefix20-online1000-h1008-20261016');sys.path.insert(0,str(R))
from reproduction.h1008_qprobe.management_transport import remote
AUTH={'request': 'Stop current training, resume the latest complete checkpoint, include question plus 20 policy-token representations, flatten S*L to one correctness logit; replay the saved 70 data batches for 70 iterations and evaluate fixed-test mixed-group accuracy only.', 'confirmed_data_protocol': 'Replay the saved 70 batches sequentially, one 5120-question G5 batch per replay iteration, global batch256.', 'target_job_id': 3409, 'source_base': '/mnt/data/shaoshitong/search-g1-prefix20-online1000-20261016-4bc6838-lr3e5', 'scope': 'Stop only current native3409 for this architecture/data-protocol replacement; preserve source data and all foreign tasks; new70replay work authorized.', 'original_individual_message_timestamp_unavailable': True, 'request_recorded_utc': '2026-10-16T05:22:23.060362+00:00', 'supersedes': 'Previous1000 fresh-rollout continuation; do not restart it from old heartbeat.'}
local_intent=W/'prefix20-3409-question20-replay70-stop-once-v1-local-intent.json'
if local_intent.exists():raise FileExistsError('Stop intent already exists; query actual result, never replay')
local_intent.write_text(json.dumps(AUTH,ensure_ascii=False,indent=2)+'\n')
script=r'''
from pathlib import Path
import json,subprocess,datetime,os,sys,hashlib
A=AUTH
B=Path('/mnt/data/shaoshitong/search-g1-prefix20-online1000-20261016-4bc6838-lr3e5');sys.path.insert(0,str(B/'repo'))
from reproduction.common8.native_allocation import fields
P=B/'preparation';I=P/'question20-replay70-stop-once-v1-intent.json';O=P/'question20-replay70-stop-once-v1-receipt.json'
def once(p,x):
 with p.open('x')as f:json.dump(x,f,ensure_ascii=False,sort_keys=True);f.flush();os.fsync(f.fileno())
if I.exists():
 print(json.dumps(dict(status='EXISTING_STOP_INTENT_NO_REPLAY',intent=json.loads(I.read_text()),receipt=json.loads(O.read_text())if O.exists()else None)));raise SystemExit
c=json.loads((P/'launch-config.json').read_text());sub=json.loads((P/'submit-receipt.json').read_text())
assert str(sub['job_id'])=='3409' and c['run_id']=='8a6b24dc19664cb3a3fb97101f21c54e'
raw=subprocess.check_output(['scontrol','show','job','3409','-o'],text=True,stderr=subprocess.PIPE,timeout=30);n=fields(raw)
assert n['JobId']=='3409'and n['Comment']=='g1-prefix20-online:'+c['run_id']and n['Command']==str(P/'native-launch.sh')
assert n['NodeList']=='h100-8'and n['JobState']=='RUNNING'
latest=json.loads((B/'train/latest.json').read_text());pool=json.loads((B/'question-pool/state.json').read_text())
intent=dict(human_authorization=A,source_base=str(B),native_before=n,native_raw_before=raw,latest_before=latest,pool_before=pool,command=['scancel','3409'],created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
once(I,intent)
try:
 r=subprocess.run(['scancel','3409'],capture_output=True,text=True,timeout=30)
 receipt=dict(status='SCANCEL_REQUESTED'if r.returncode==0 else'SCANCEL_ERROR_INSPECT_BEFORE_RETRY',returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,intent_sha256=hashlib.sha256(I.read_bytes()).hexdigest(),completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),latest_before=latest)
except BaseException as e:
 receipt=dict(status='SCANCEL_OUTCOME_UNKNOWN_NO_REPLAY',error=type(e).__name__,message=str(e),intent_sha256=hashlib.sha256(I.read_bytes()).hexdigest())
once(O,receipt);print(json.dumps(receipt,ensure_ascii=False))
'''.replace('AUTH',repr(AUTH))
transport=W/'prefix20-3409-question20-replay70-stop-once-v1-transport.json';out=json.loads(remote(script,transport,control_path=None,target='h100-8'));p=W/'prefix20-3409-question20-replay70-stop-once-v1-result.json';p.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),result=out),ensure_ascii=False))
