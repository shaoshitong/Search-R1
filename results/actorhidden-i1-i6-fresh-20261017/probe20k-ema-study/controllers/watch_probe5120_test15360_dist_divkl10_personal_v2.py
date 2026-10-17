from pathlib import Path
import datetime,json,hashlib,time
from reproduction.prefix_selection200.probe20k_control import fit_observation_code
from reproduction.h1008_qprobe.established_controller_transport import remote
root=Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control');state=root/'work/probe20k-ema-study-state.json';original=json.loads(state.read_text());base=original['fit_base'];out=root/'work/probe5120-test15360-dist-divkl10-eval15360fix-personal-complete';out.mkdir(exist_ok=True)
while True:
 s=json.loads(state.read_text())
 if s.get('fit_base')!=base or s.get('execution_authorization_active')is not True:print('SUPERSEDED_READONLY_OBSERVER_NO_NATIVE_ACTION',flush=True);break
 tag=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');p=out/('actual-'+tag+'.json')
 try:v=json.loads(remote(fit_observation_code(base),str(out/('actual-'+tag+'.transport.json'))))
 except Exception as e:print(json.dumps(dict(status='OBSERVATION_ERROR_NOT_NATIVE_EXIT',error=repr(e))),flush=True);time.sleep(45);continue
 p.write_text(json.dumps(v,indent=2)+'\n');arms={}
 for name,a in v['arms'].items():
  metric=a['metrics'][-1]if a['metrics']else None
  arms[name]=dict(latest=a['latest'],progress=a['progress'],latest_metrics=metric,rank_last_updates=a['rank_last_updates'],complete=a['complete'])
 summary=dict(actual=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),job_id=v['job_id'],native=v['native'].get('JobState','UNKNOWN'),stage=v['stage'],arms=arms,complete=v['complete'],cleanup=v['cleanup'],error_tails=v['error_tails'])
 (out/'latest-small-summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 compact={name:dict(epoch=(a['latest']or{}).get('epoch',0),progress=a['progress'],ema_test_BA=(a['latest_metrics']or{}).get('ema',{}).get('test',{}).get('balanced_accuracy'),raw_test_BA=(a['latest_metrics']or{}).get('raw',{}).get('test',{}).get('balanced_accuracy'),complete=bool(a['complete']))for name,a in arms.items()}
 print(json.dumps(dict(actual=str(p),sha256=summary['sha256'],job_id=v['job_id'],native=summary['native'],arms=compact,complete=v['complete'],cleanup=v['cleanup'],error_tails=v['error_tails'])),flush=True)
 if summary['native']in('COMPLETED','FAILED','CANCELLED','OUT_OF_MEMORY','TIMEOUT'):break
 time.sleep(45)
