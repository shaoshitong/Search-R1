from pathlib import Path
import datetime,json,hashlib,time
from reproduction.prefix_selection200.probe20k_control import observation_code
from reproduction.h1008_qprobe.established_controller_transport import remote
root=Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control');state=root/'work/probe20k-ema-study-state.json';original=json.loads(state.read_text());out=root/'work/probe20k-data-3469-personal-complete';out.mkdir(exist_ok=True)
while True:
 s=json.loads(state.read_text())
 if s['base']!=original['base']or s.get('execution_authorization_active')is not True:print('SUPERSEDED_READONLY_OBSERVER_NO_NATIVE_ACTION',flush=True);break
 tag=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');p=out/('actual-'+tag+'.json')
 try:v=json.loads(remote(observation_code(s['base']),str(out/('actual-'+tag+'.transport.json'))))
 except Exception as e:print(json.dumps(dict(status='OBSERVATION_ERROR_NOT_NATIVE_EXIT',error=repr(e))),flush=True);time.sleep(60);continue
 p.write_text(json.dumps(v,indent=2)+'\n');summary=dict(actual=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),job_id=v['job_id'],native=v['native'].get('JobState','UNKNOWN'),stage=v['stage'],blocks=v['blocks'],complete=v['complete'],cleanup=v['cleanup'],error_tails=v['error_tails']);(out/'latest-small-summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)
 if summary['native']in('COMPLETED','FAILED','CANCELLED','OUT_OF_MEMORY','TIMEOUT'):break
 time.sleep(60)
