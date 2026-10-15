"""Once scoped native stop for the current explicit human LR-reduction request."""
from pathlib import Path
import datetime,json,hashlib,sys
W=Path(__file__).parent.resolve();R=Path('/Users/shaoshitong/Documents/Search-G1/qprobe-prefix20-online1000-h1008-20261016');sys.path.insert(0,str(R))
from reproduction.h1008_qprobe.management_transport import remote
AUTH=dict(request='是不是学习率太大了呢？我发现准确率没上去了？你试着从最新的结果resume，然后学习率降低为现在的1/1。',clarification='降为1/10：3e-5（推荐）',instruction_scope='stop only current3408 for latest-complete recovery at target3e-5; keep1000goal active; preserveforeign',current_turn_started_unix=1792097072,original_individual_message_timestamp_unavailable=True,request_recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),target_job_id=3408,source_learning_rate=3e-4,target_learning_rate=3e-5)
local_intent=W/'prefix20-3408-lr3e5-stop-once-v1-local-intent.json'
if local_intent.exists():raise FileExistsError('Stop intent already exists; query actual result, never replay')
local_intent.write_text(json.dumps(AUTH,ensure_ascii=False,indent=2)+'\n')
script=r'''
from pathlib import Path
import json,subprocess,datetime,os,sys,hashlib
A=AUTH
B=Path('/mnt/data/shaoshitong/search-g1-prefix20-online1000-20261016-ca1c60f');sys.path.insert(0,str(B/'repo'))
from reproduction.common8.native_allocation import fields
P=B/'preparation';I=P/'lr3e5-stop-once-v1-intent.json';O=P/'lr3e5-stop-once-v1-receipt.json'
def once(p,x):
 with p.open('x')as f:json.dump(x,f,ensure_ascii=False,sort_keys=True);f.flush();os.fsync(f.fileno())
if I.exists():
 print(json.dumps(dict(status='EXISTING_STOP_INTENT_NO_REPLAY',intent=json.loads(I.read_text()),receipt=json.loads(O.read_text())if O.exists()else None)));raise SystemExit
c=json.loads((P/'launch-config.json').read_text());sub=json.loads((P/'submit-receipt.json').read_text())
assert sub['job_id']==3408 and c['run_id']=='8a6b24dc19664cb3a3fb97101f21c54e'
raw=subprocess.check_output(['scontrol','show','job','3408','-o'],text=True,stderr=subprocess.PIPE,timeout=30);n=fields(raw)
assert n['JobId']=='3408'and n['Comment']=='g1-prefix20-online:'+c['run_id']and n['Command']==str(P/'native-launch.sh')
assert n['NodeList']=='h100-8'and n['JobState']=='RUNNING'
latest=json.loads((B/'train/latest.json').read_text());pool=json.loads((B/'question-pool/state.json').read_text())
intent=dict(human_authorization=A,source_base=str(B),native_before=n,native_raw_before=raw,latest_before=latest,pool_before=pool,command=['scancel','3408'],created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
once(I,intent)
try:
 r=subprocess.run(['scancel','3408'],capture_output=True,text=True,timeout=30)
 receipt=dict(status='SCANCEL_REQUESTED'if r.returncode==0 else'SCANCEL_ERROR_INSPECT_BEFORE_RETRY',returncode=r.returncode,stdout=r.stdout,stderr=r.stderr,intent_sha256=hashlib.sha256(I.read_bytes()).hexdigest(),completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),latest_before=latest)
except BaseException as e:
 receipt=dict(status='SCANCEL_OUTCOME_UNKNOWN_NO_REPLAY',error=type(e).__name__,message=str(e),intent_sha256=hashlib.sha256(I.read_bytes()).hexdigest())
once(O,receipt);print(json.dumps(receipt,ensure_ascii=False))
'''.replace('AUTH',repr(AUTH))
transport=W/'prefix20-3408-lr3e5-stop-once-v1-transport.json';out=json.loads(remote(script,transport,control_path=None,target='h100-8'));p=W/'prefix20-3408-lr3e5-stop-once-v1-result.json';p.write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n');print(json.dumps(dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),result=out),ensure_ascii=False))
