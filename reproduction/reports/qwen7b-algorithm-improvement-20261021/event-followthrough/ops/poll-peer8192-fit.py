"""One finite read-only snapshot, never a persistent watcher/submission."""
import argparse,json,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--index',type=int,required=True);a=p.parse_args();C=Path(__file__).resolve().parents[2];repo=C/'work/qwen7b-probe-warmup-20261019';r=subprocess.run([sys.executable,'-m','reproduction.operations.qwen7b_warmup.deploy_peer_newseed_train','--control-root',str(C),'--action','status'],cwd=repo,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,timeout=90)
if r.returncode:sys.stderr.write(r.stderr);raise SystemExit(r.returncode)
v=json.loads(r.stdout);out=C/'outputs/qwen7b-warmup-20261019/reports'/('peer8192-fit-progress-%03d.json'%a.index);out.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n');keys=['observed_epoch','job_id','squeue','sacct','progress.json','failure.json','gpu-memory.json','complete'];summary={k:v[k] for k in keys if k in v}
if 'latest.json' in v:summary['latest']={k:v['latest.json'][k] for k in ['cursor','steps','sha256','bytes'] if k in v['latest.json']}
if 'next-update-replay.json' in v:summary['first_replay']=v['next-update-replay.json']['status']
print(json.dumps(summary,sort_keys=True))
