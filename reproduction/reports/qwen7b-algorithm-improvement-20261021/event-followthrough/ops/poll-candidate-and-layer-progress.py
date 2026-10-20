"""One-shot read-only status poll; no watcher, submission or remote mutation."""
import argparse,json,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--index',type=int,required=True);p.add_argument('--layers',action='store_true');a=p.parse_args()
C=Path(__file__).resolve().parents[2];R=C/'work/qwen7b-probe-warmup-20261019';O=C/'outputs/qwen7b-warmup-20261019/reports'
def call(module,extra):
 cmd=['python3','-m','reproduction.operations.qwen7b_warmup.'+module,'--control-root',str(C),'--action','status']+extra
 out=subprocess.run(cmd,cwd=str(R),stdout=subprocess.PIPE,stderr=subprocess.PIPE,universal_newlines=True,timeout=60)
 lines=[json.loads(v) for v in out.stdout.splitlines() if v.startswith('{')]
 if out.returncode or not lines:
  return dict(poll_failed=True,returncode=out.returncode,stderr=out.stderr[-1500:],stdout=out.stdout[-1500:])
 return lines[-1]
x=call('deploy_training_candidate_score',['--candidate-plan','/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/hard-enrichment-20261021/candidates.json','--prepared-commit','9de8a8d'])
(O/('training-candidate-followthrough-status-%d.json'%a.index)).write_text(json.dumps(x)+'\n')
r=dict(index=a.index,observed_epoch=x.get('observed_epoch',time.time()),candidate={k:x[k] for k in ('job_id','sacct','progress.json','failure','complete.json','poll_failed','returncode','stderr','stdout') if k in x})
if a.layers:
 y=call('deploy_multilayer_suite',[]);(O/('multilayer-candidate-concurrent-status-%d.json'%a.index)).write_text(json.dumps(y)+'\n')
 r['layers']=dict(sacct=y.get('sacct'),variants={k:dict(progress=v.get('progress.json'),failure=v.get('failure.json'),completed_epochs=list(v.get('epochs',{}))) for k,v in y.get('variants',{}).items()},poll_failed=y.get('poll_failed',False))
r['complete']='complete.json' in x
bad_states=('FAILED','TIMEOUT','CANCELLED','OUT_OF_MEMORY','NODE_FAIL','PREEMPTED')
scheduler_failure=any(t in x.get('sacct','') for t in bad_states) or ('COMPLETED' in x.get('sacct','') and not r['complete'])
layer_scheduler_failure=any(t in r.get('layers',{}).get('sacct','') for t in bad_states)
r['needs_attention']=bool(scheduler_failure or layer_scheduler_failure or x.get('failure') or x.get('poll_failed') or (r.get('layers',{}).get('poll_failed')) or any(v.get('failure') for v in r.get('layers',{}).get('variants',{}).values()))
print(json.dumps(r),flush=True)
