"""One finite read-only actual state snapshot; no watcher/submission/retry."""
import argparse,json,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--index',required=True);a=p.parse_args();C=Path(__file__).resolve().parents[2];R=C/'work/qwen7b-probe-warmup-20261019';P=C/'outputs/qwen7b-warmup-20261019/reports'
r=subprocess.run([sys.executable,'-m','reproduction.operations.qwen7b_warmup.deploy_peer_g10_calibration','--control-root',str(C),'--commit','a0f41cb45daa41ed811577037d950f4ff12dfe44','--action','status'],cwd=R,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
if r.returncode:raise RuntimeError(r.stderr[-2000:]+' '+r.stdout[-2000:])
v=json.loads((P/'peer8192-calibration-score-status-latest.json').read_text());small={k:x for k,x in v.items() if k in ('job_id','squeue','sacct','progress','failure','complete','complete_ref','gpu_log_tail','observed_epoch','E4','peer256')}
(P/('peer8192-calibration-progress-'+a.index+'.json')).write_text(json.dumps(small,indent=2,sort_keys=True)+'\n');print(json.dumps(small,sort_keys=True))
