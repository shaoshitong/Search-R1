from pathlib import Path
import sys,json,datetime,time,hashlib
ROOT=Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control')
sys.path.insert(0,'/Users/shaoshitong/Documents/Search-G1/actorhidden-period20-20261016')
from reproduction.h1008_qprobe.established_controller_transport import remote
arms=json.loads((ROOT/'work/prefix5-ab-assigned-arms.json').read_text())
repo=json.loads((ROOT/'work/prefix5-reference-code-stage.json').read_text())['repo']
verified={k:0 for k in arms};last=None;proofs={k:[] for k in arms}
while True:
 stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
 inner='import sys,json\nsys.path.insert(0,'+repr(repo)+')\nfrom reproduction.prefix_selection200.ab_control import observe\n'
 inner+='ARMS='+repr(arms)+'\nVERIFIED='+repr(verified)+'\nprint(json.dumps({s:observe(v["base"],VERIFIED[s]) for s,v in ARMS.items()}))\n'
 code='import subprocess,os\nr=subprocess.run(["/dataset/shaoshitong/Search-G1-common8/env/bin/python","-c",'+repr(inner)+'],env=dict(os.environ,CUDA_VISIBLE_DEVICES="",PYTHONDONTWRITEBYTECODE="1"),text=True,capture_output=True,timeout=120)\nassert r.returncode==0,r.stderr\nprint(r.stdout)'
 try:
  raw=json.loads(remote(code,ROOT/('work/prefix5-ab-watch-'+stamp+'.transport.json')))
  path=ROOT/('work/prefix5-ab-watch-'+stamp+'-actual.json');path.write_text(json.dumps(raw,indent=2)+'\n')
  compact={}
  terminal=[]
  for slot,v in raw.items():
   for fact in v['facts']:proofs[slot].append(fact);verified[slot]=max(verified[slot],fact['outer'])
   compact[slot]={k:v[k] for k in ('base','mode','complete_outer','actor_adam','probe_adam','stage','native','prefix','continuation','cleanup','pipeline_complete')}
   if v['cleanup'] and not v['pipeline_complete']:terminal.append(slot)
  value=dict(observed_utc=stamp,arms=compact,original=str(path),original_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),verified=verified,terminal_attention=terminal)
  (ROOT/'work/prefix5-ab-latest-compact.json').write_text(json.dumps(value,indent=2)+'\n')
  key=[(s,v['complete_outer'],(v['stage']or{}).get('stage')) for s,v in raw.items()]
  if key!=last:
   print(json.dumps(value),flush=True);last=key
  if terminal:raise SystemExit(3)
  if all(n>=1 for n in verified.values()):
   (ROOT/'work/prefix5-ab-first-complete-actual-facts.json').write_text(json.dumps(dict(facts=proofs,last_observation=str(path)),indent=2)+'\n')
   print('BOTH_ARMS_FIRST_OUTER_ACTUALLY_COMPLETE',flush=True);break
 except (RuntimeError,ValueError,KeyError,AssertionError) as error:
  print(json.dumps(dict(observation_error=repr(error),tag=stamp,training_modified=False)),flush=True)
 time.sleep(45)
