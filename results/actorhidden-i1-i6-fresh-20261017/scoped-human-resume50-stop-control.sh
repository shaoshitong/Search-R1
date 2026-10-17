#!/bin/zsh
set -e
cd /Users/shaoshitong/Documents/Search-G1/actorhidden-period20-20261016
python3 - <<'PY'
import pathlib,json,datetime,hashlib
from reproduction.h1008_qprobe.management_transport import remote
w=pathlib.Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control/work');s=json.loads((w/'search-g1-actorhidden-period20-state.json').read_text());auth=s['latest_human_probe_lr_authorization'];scope=auth['stop_scope'];assert str(s['current_job_id'])==str(scope['job_id']) and s['base']==scope['base'] and s['run_id']==scope['run_id'],'A delayed control cannot stop a later execution';payload={'base':scope['base'],'job':str(scope['job_id']),'run_id':scope['run_id'],'authorization':auth};script='payload='+repr(payload)+'\n'+r'''
import pathlib,json,hashlib,os,subprocess,time,sys
b=pathlib.Path(payload['base']);sys.path.insert(0,str(b/'repo'));from reproduction.common8.native_allocation import fields
root=b/'preparation/human-resume50-lr1e5-stop-once-v1';out={'read_source_config':json.loads((b/'preparation/launch-config.json').read_text()),'source_config_sha256':hashlib.sha256((b/'preparation/launch-config.json').read_bytes()).hexdigest(),'source_joint50_sha256':hashlib.sha256((b/'outers/outer-000050/complete.json').read_bytes()).hexdigest(),'source_latest_outer_before_stop':json.loads((b/'latest-joint.json').read_text())['outer']}
if root.exists():
 out['stop']={'status':'EXISTING_STOP_INTENT_RECONCILE_NO_REPLAY','files':[p.name for p in root.iterdir()],'receipt':json.loads((root/'RECEIPT.json').read_text())if(root/'RECEIPT.json').exists()else None};print(json.dumps(out));raise SystemExit
r=subprocess.run(['scontrol','show','job',payload['job'],'-o'],capture_output=True,text=True,timeout=20);assert r.returncode==0,r.stderr;n=fields(r.stdout);assert n['Command']==str(b/'preparation/native-launch.sh')and n['Comment']=='g1-prefix200:'+payload['run_id']and n['JobName']=='g1-prefix200-'+payload['run_id'][:12]
assert n['JobState']in('RUNNING','PENDING','COMPLETING'),n
root.mkdir()
def once(p,v):
 with p.open('x')as f:json.dump(v,f,sort_keys=True);f.flush();os.fsync(f.fileno())
once(root/'INTENT.json',dict(payload,operation='HUMAN_CONTROLLED_STOP_FOR_RETAINED50_PROBE_LR_ONLY_RESUME',native_before=n,created_unix=time.time(),goal_paused=False))
r=subprocess.run(['scancel',payload['job']],capture_output=True,text=True,timeout=30);receipt={'job':payload['job'],'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'native_before':n,'requested_resume_outer':50,'human_authorization':payload['authorization'],'goal_paused':False,'training_algorithm_changed':False,'checked_unix':time.time()};once(root/'RECEIPT.json',receipt);out['stop']=receipt;print(json.dumps(out))
''';p=w/('actorhidden-resume50-lr1e5-scoped-stop-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')+'-transport.json');v=json.loads(remote(script,p,control_path=s['management_transport']['control_path']));a=p.with_name(p.name.replace('-transport.json','-actual.json'));a.write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n');cfg=w/'actorhidden-resume50-lr1e5-original-config.json';cfg.write_text(json.dumps(v['read_source_config'],ensure_ascii=False,indent=2)+'\n');print(json.dumps({'actual':str(a),'sha256':hashlib.sha256(a.read_bytes()).hexdigest(),'source_config':str(cfg),'source_config_sha256':v['source_config_sha256'],'source_joint50_sha256':v['source_joint50_sha256'],'latest_before_stop':v['source_latest_outer_before_stop'],'stop_returncode':v['stop'].get('returncode'),'state_before':v['stop'].get('native_before',{}).get('JobState')},ensure_ascii=False))
PY
