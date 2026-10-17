# Archived readonly observer. Never duplicate a confirmed live handle.
from pathlib import Path
import sys,json,datetime,time,hashlib,importlib.util,subprocess
ROOT=Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control')
spec=importlib.util.spec_from_file_location('ctl','/Users/shaoshitong/Documents/remote_codex_control/control.py');ctl=importlib.util.module_from_spec(spec);spec.loader.exec_module(ctl)
arms=json.loads((ROOT/'work/prefix5-ab-assigned-arms.json').read_text());assert all(x.get('node')=='h100-11'for x in arms.values())
commit='2e3976230ac9425d66d8553afa9962821e964932'
wrapper='/mnt/shared/workspace/training_data_node_8/shaoshitong/search-g1-prefix5-reference-preparation-20261018/code/'+commit+'/repo/reproduction/operations/h100_reference_replica/h10011_private_paths.sh'
verified={s:2 for s in arms};proofs={s:[]for s in arms};last=None
while True:
 stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
 inner='import sys,json\nfrom pathlib import Path\nARMS='+repr(arms)+'\nVERIFIED='+repr(verified)+'\nsys.path.insert(0,next(iter(ARMS.values()))["base"]+"/repo")\nfrom reproduction.prefix_selection200.ab_control import observe\nr={s:observe(v["base"],VERIFIED[s])for s,v in ARMS.items()}\n'
 inner+='\nfor s,v in r.items():\n b=Path(ARMS[s]["base"]);v["partial_actor"]=[]\n for p in sorted((b/"actor/outers").glob("outer-*-attempt-*/rank-0-updates.jsonl")):\n  rows=[json.loads(l)for l in p.read_text().splitlines()];w=[x for x in rows if x.get("event")=="ACTUAL_ADAM_WRITE"];v["partial_actor"].append({"attempt":p.parent.name,"writes":len(w),"actual":w[-1].get("actual_adam_steps")if w else None})\n if v["cleanup"]:\n  p=b/"logs/native-pipeline.log"\n  with p.open("rb")as f:f.seek(max(0,p.stat().st_size-6000));v["terminal_log_tail"]=f.read().decode(errors="replace")\nprint(json.dumps(r))'
 script='import subprocess,os\np=subprocess.run(["unshare","--mount","--propagation","private","bash",'+repr(wrapper)+',"/dataset/shaoshitong/Search-G1-common8/env/bin/python","-c",'+repr(inner)+'],env=dict(os.environ,CUDA_VISIBLE_DEVICES="",PYTHONDONTWRITEBYTECODE="1"),text=True,capture_output=True,timeout=120)\nassert p.returncode==0,p.stderr\nprint(p.stdout)'
 try:
  p=subprocess.run(ctl.route('h100-11','/usr/bin/python3 -'),input=script,text=True,capture_output=True,timeout=155)
  evidence=ROOT/('work/prefix5-ab-h10011-watch-'+stamp+'-actual.json')
  if p.returncode:raise RuntimeError(p.stderr or p.stdout)
  raw=json.loads(p.stdout);evidence.write_text(json.dumps(raw,indent=2)+'\n')
  compact={};terminal=[]
  for s,v in raw.items():
   for fact in v['facts']:proofs[s].append(fact);verified[s]=max(verified[s],fact['outer'])
   compact[s]={k:v[k]for k in ['base','mode','complete_outer','actor_adam','probe_adam','stage','native','cleanup','pipeline_complete','partial_actor','qa']}
   if v['cleanup']and not v['pipeline_complete']:terminal.append(s)
  value=dict(observed_utc=stamp,node='h100-11',arms=compact,original=str(evidence),original_sha256=hashlib.sha256(evidence.read_bytes()).hexdigest(),verified=verified,terminal_attention=terminal)
  (ROOT/'work/prefix5-ab-latest-compact.json').write_text(json.dumps(value,indent=2)+'\n')
  key=[(s,v['complete_outer'],(v['stage']or{}).get('stage'),v['partial_actor'])for s,v in raw.items()]
  if key!=last:print(json.dumps(value),flush=True);last=key
  if terminal:
   print(json.dumps({s:raw[s].get('terminal_log_tail')for s in terminal}),flush=True);raise SystemExit(3)
  if all(x>=5 for x in verified.values()) and all('5' in r.get('qa',{}) for r in raw.values()):
   (ROOT/'work/prefix5-ab-h10011-through5-qa-actual-facts.json').write_text(json.dumps(dict(facts=proofs,last_observation=str(evidence)),indent=2)+'\n');print('BOTH_ARMS_THROUGH5_AND_QA5_ACTUALLY_COMPLETE',flush=True);break
 except (RuntimeError,ValueError,KeyError,AssertionError,subprocess.TimeoutExpired)as error:print(json.dumps(dict(observation_error=repr(error),tag=stamp,training_modified=False)),flush=True)
 time.sleep(60)
