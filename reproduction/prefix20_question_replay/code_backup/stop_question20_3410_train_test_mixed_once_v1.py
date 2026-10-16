"""Once scoped native stop for the current human question20 replay70 replacement."""
from pathlib import Path
import datetime,json,hashlib,sys
W=Path(__file__).parent.resolve();R=Path('/Users/shaoshitong/Documents/Search-G1/qprobe-prefix20-online1000-h1008-20261016');sys.path.insert(0,str(R))
from reproduction.h1008_qprobe.management_transport import remote
AUTH={'request': 'Report mixed-group decision accuracy for both the training set and test set.', 'implementation_scope': 'Add post-epoch frozen-model full-train and fixed-test inference; necessary clean continuation of current frozen runtime3410 from its latest complete replay state; total70 and LR3e-6 unchanged, no hotedit/reset/new rollout.', 'target_job_id': 3410, 'source_base': '/mnt/data/shaoshitong/search-g1-question20-replay70-20261016-5534d25', 'original_individual_message_timestamp_unavailable': True, 'request_recorded_utc': '2026-10-16T05:55:06.601121+00:00', 'earlier_authorization': 'Human authorized stop/latestcheckpoint resume for question+20 and seventy cached batches; later explicitly set LR3e-6.'}
local_intent=W/'question20-3410-train-test-mixed-stop-once-v1-local-intent.json'
if local_intent.exists():raise FileExistsError('Stop intent already exists; query actual result, never replay')
local_intent.write_text(json.dumps(AUTH,ensure_ascii=False,indent=2)+'\n')
script=r'''
from pathlib import Path
import json,subprocess,datetime,os,sys,hashlib
A=AUTH
B=Path('/mnt/data/shaoshitong/search-g1-question20-replay70-20261016-5534d25');sys.path.insert(0,str(B/'repo'))
from reproduction.common8.native_allocation import fields
P=B/'preparation';I=P/'train-test-mixed-stop-once-v1-intent.json';O=P/'train-test-mixed-stop-once-v1-receipt.json'
def once(p,x):
 with p.open('x')as f:json.dump(x,f,ensure_ascii=False,sort_keys=True);f.flush();os.fsync(f.fileno())
if I.exists():
 print(json.dumps(dict(status='EXISTING_STOP_INTENT_NO_REPLAY',intent=json.loads(I.read_text()),receipt=json.loads(O.read_text())if O.exists()else None)));raise SystemExit
c=json.loads((P/'launch-config.json').read_text());sub=json.loads((P/'submit-receipt.json').read_text())
assert str(sub['job_id'])=='3410' and c['run_id']=='80cb2d8592304bc197614be9b04911c9'
raw=subprocess.check_output(['scontrol','show','job','3410','-o'],text=True,stderr=subprocess.PIPE,timeout=30);n=fields(raw)
assert n['JobId']=='3410'and n['Comment']=='g1-q20-replay70:'+c['run_id']and n['Command']==str(P/'native-launch.sh')
assert n['NodeList']=='h100-8'and n['JobState']=='RUNNING'
latest=json.loads((B/'train/latest.json').read_text());pool={}
intent=dict(human_authorization=A,source_base=str(B),native_before=n,native_raw_before=raw,latest_before=latest,pool_before=pool,command=['scancel','3410'],created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
once(I,intent)
try:
 r=subprocess.run(['scancel','3410'],capture_output=True,text=True,timeout=30)
 receipt=dict(status='SCANCEL_REQUESTED'if r.returncode==0 else'SCANCEL_ERROR_INSPECT_BEFORE_RETRY',returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,intent_sha256=hashlib.sha256(I.read_bytes()).hexdigest(),completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),latest_before=latest)
except BaseException as e:
 receipt=dict(status='SCANCEL_OUTCOME_UNKNOWN_NO_REPLAY',error=type(e).__name__,message=str(e),intent_sha256=hashlib.sha256(I.read_bytes()).hexdigest())
once(O,receipt);print(json.dumps(receipt,ensure_ascii=False))
'''.replace('AUTH',repr(AUTH))
transport=W/'question20-3410-train-test-mixed-stop-once-v1-transport.json';out=json.loads(remote(script,transport,control_path=None,target='h100-8'));p=W/'question20-3410-train-test-mixed-stop-once-v1-result.json';p.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),result=out),ensure_ascii=False))
