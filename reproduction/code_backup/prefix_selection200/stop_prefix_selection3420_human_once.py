"""Scoped human stop; preserve all remote experiment artifacts."""
import datetime
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
REPO = Path('/Users/shaoshitong/Documents/Search-G1/prefix-selection200-20261016')
sys.path.insert(0, str(REPO))
from reproduction.h1008_qprobe.management_transport import remote

auth = dict(
    request='这个已经不用跑了，这个已经问题很大，原因估计很简单，那就是选出来的512题*5的轨迹本身就有bias',
    recorded_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    original_message_exact_timestamp_unavailable=True,
    target_job_id=3420,
    run_id='5cba22ca409a4571a1e6b87d89c86b09',
    scope='Stop current formal prefix-selection actor/probe training; do not restart. Preserve results and independent backup. Answer probe questions.',
    later_resume_authorization=False,
)
intent = ROOT / 'prefix-selection3420-human-stop-local-intent.json'
with intent.open('x') as f:
    json.dump(auth, f, ensure_ascii=False, indent=2)
script = r'''
import json,pathlib,os,subprocess,sys,datetime,hashlib
A=AUTH
b=pathlib.Path('/mnt/data/shaoshitong/search-g1-prefix-selection200-20261016-5cba22ca409a-resume4');p=b/'preparation'
sys.path.insert(0,str(b/'repo'))
from reproduction.common8.native_allocation import fields
def read(x):return json.loads(x.read_text())
def once(x,v):
 with x.open('x') as f:json.dump(v,f,ensure_ascii=False,sort_keys=True);f.flush();os.fsync(f.fileno())
i=p/'human-stop-3420-once-intent.json';o=p/'human-stop-3420-once-receipt.json'
if i.exists():
 print(json.dumps(dict(status='EXISTING_STOP_NO_REPLAY',intent=read(i),receipt=read(o) if o.exists() else None)));raise SystemExit
c=read(p/'launch-config.json');sub=read(p/'submit-receipt.json')
assert c['run_id']==A['run_id'] and str(sub['job_id'])=='3420'
raw=subprocess.check_output(['scontrol','show','job','3420','-o'],text=True,stderr=subprocess.PIPE,timeout=20);n=fields(raw)
assert n['JobId']=='3420' and n['Comment']=='g1-prefix200:'+c['run_id'] and n['Command']==str(p/'native-launch.sh')
assert n['NodeList']=='h100-8'
before=dict(native=n,stage=read(b/'logs/stage.json') if (b/'logs/stage.json').exists() else None,latest=read(b/'joint/latest.json') if (b/'joint/latest.json').exists() else None)
once(i,dict(human_authorization=A,before=before,native_raw=raw,command=['scancel','3420']))
if n['JobState'] in ('RUNNING','PENDING','CONFIGURING','COMPLETING','SUSPENDED'):
 r=subprocess.run(['scancel','3420'],capture_output=True,text=True,timeout=20)
 v=dict(status='SCANCEL_REQUESTED' if r.returncode==0 else 'SCANCEL_ERROR',returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
else:v=dict(status='ALREADY_TERMINAL_NO_CANCEL',native=n)
v.update(before=before,completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),intent_sha256=hashlib.sha256(i.read_bytes()).hexdigest());once(o,v);print(json.dumps(v))
'''.replace('AUTH', repr(auth))
tag = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
v = json.loads(remote(script, ROOT / ('prefix-selection3420-human-stop-'+tag+'-transport.json'), target='h100-8'))
out = ROOT / ('prefix-selection3420-human-stop-'+tag+'-actual.json')
out.write_text(json.dumps(v, ensure_ascii=False, indent=2)+'\n')
print(json.dumps(dict(path=str(out), result=v), ensure_ascii=False))
