"""One finite read-only expanded capture observation, no repeated writer."""
import argparse,json,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--index',required=True);a=p.parse_args();C=Path(__file__).resolve().parents[2];R=C/'work/qwen7b-probe-warmup-20261019';P=C/'outputs/qwen7b-warmup-20261019/reports'
r=subprocess.run([sys.executable,'-m','reproduction.operations.qwen7b_warmup.deploy_peer_expanded_l26','--control-root',str(C),'--commit','bea03d5f1d908f357c368684b9e8c086ca0e687b','--action','status'],cwd=R,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
if r.returncode:raise RuntimeError(r.stderr[-1500:]+' '+r.stdout[-1500:])
v=json.loads(r.stdout);out={k:v[k] for k in ('epoch','squeue','sacct','unavailable_optional_observations') if k in v};out['shards']={}
for n,s in v['shards'].items():
 out['shards'][n]={k:s[k] for k in ('job_id','complete_chunks','progress.json','failure.json','complete.json') if k in s}
 if 'failure.json' in s:out['shards'][n]['logs']=s.get('logs')
(P/('peer-expanded-L26-progress-'+a.index+'.json')).write_text(json.dumps(out,sort_keys=True,indent=2)+'\n');print(json.dumps(out))
