"""One finite read-only snapshot of owned matched within1 training."""
import argparse,json,subprocess,sys
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--index',required=True);a=p.parse_args();C=Path(__file__).resolve().parents[2];R=C/'work/qwen7b-probe-warmup-20261019';P=C/'outputs/qwen7b-warmup-20261019/reports'
r=subprocess.run([sys.executable,'-m','reproduction.operations.qwen7b_warmup.deploy_peer_within1_matched','--control-root',str(C),'--commit','ebc452595fddfd5eb095b45091a642c1e02ea060','--action','status'],cwd=R,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
if r.returncode:raise RuntimeError(r.stderr[-2000:]+' '+r.stdout[-2500:])
v=json.loads((P/'peer-within1-status-latest.json').read_text());small={k:v[k] for k in ('observed_epoch','job_id','squeue','sacct','start_estimate','failure','first-cp-audit-failure','interrupted') if k in v}
for n in ('progress','latest','next-update-replay','first-cp-independent-audit','controller','complete','complete_ref'):
 if n not in v:continue
 x=v[n]
 if n=='progress':x={k:x[k] for k in ('cursor','steps','loss','components','seconds','original_questions','unique_real_trajectories','virtual_views','actual_forward_trajectory_exposures','recomputed_logits_bit_identical','within_actual_feature_pairs','within_canonical_and_feature_pairs','within_same_causal_excluded_feature_pairs','time') if k in x}
 if n=='latest':x={k:x[k] for k in ('cursor','steps','path','sha256','bytes','config_sha256')}
 if n=='next-update-replay':x=dict(status=x['status'],source_steps=x['source_steps'],candidate_steps=x['candidate_steps'],all_component_hashes_equal=x['expected']==x['actual'])
 small[n]=x
if 'failure' in v or 'first-cp-audit-failure' in v:small['gpu_log_tail']=v.get('gpu_log_tail','')
(P/('peer-within1-progress-'+a.index+'.json')).write_text(json.dumps(small,indent=2,sort_keys=True)+'\n');print(json.dumps(small,sort_keys=True))
