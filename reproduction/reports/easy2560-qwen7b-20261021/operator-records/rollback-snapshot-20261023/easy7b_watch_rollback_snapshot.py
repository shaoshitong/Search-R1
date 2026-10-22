import sys,json,shlex,time
from pathlib import Path
w=Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control-6');sys.path.insert(0,str(w/'work/easy7b-fa3-fastpath-20261022'))
from reproduction.operations.easy7b_20261021 import watch_cp34_recovery_local as wc
b=wc.shared.load_bridge(w)
code=Path('/tmp/easy7b_rollback_snapshot_observe_remote.py').read_text()
r=b.run('jxu957',shlex.join(['/data/user/jxu957/experiments/topd-20261008/runtime/env/bin/python','-B','-c',code]),'',str(w/'work'),login_host='10.120.48.31',expected_host_identity={'hostname':'ACD-Login-1','boot_id':'a27512a3-24a0-4ca3-aa92-98d2a5256d9a','pid_namespace':'pid:[4026531836]'},observation_timeout=50)
p=w/'outputs/easy2560-qwen7b-20261021'/('rollback-snapshot-native4-live-'+str(time.time_ns())+'.json');p.write_text(json.dumps({'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr},indent=2)+'\n');print(p)
if r.returncode==0:
 v=json.loads(r.stdout);print(json.dumps({k:v.get(k) for k in ('epoch','job','queue','accounting','log_bytes','snapshot_progress','actual_error_lines')}))
 print('reports',{rank:{k:x.get(k) for k in ('measurement_complete','constructors_observed','constructor_body_wall_seconds_total','parameter_snapshot_bytes_copied_total','adam_snapshot_bytes_copied_total')} for rank,x in v.get('reports',{}).items()});print('log_tail',v.get('log_tail','')[-1600:])
else:print(r.stdout[-1000:]);print(r.stderr[-1500:])
sys.exit(r.returncode)
