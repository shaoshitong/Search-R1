"""One fresh read-only native observation; no submit, restart, or remote writes."""
from pathlib import Path
import datetime,json,hashlib,sys,re,argparse
W=Path('/Users/shaoshitong/Documents/Codex/2026-10-02/users-shaoshitong-documents-remote-codex-control/work')
R=Path('/Users/shaoshitong/Documents/Search-G1/qprobe-prefix20-online1000-h1008-20261016')
a=argparse.ArgumentParser();a.add_argument('--role',default='current');args=a.parse_args()
if not re.fullmatch('[a-z0-9-]{1,64}',args.role):raise ValueError('Invalid observation tag')
s=json.loads((W/'prefix20-online1000-state.json').read_text());B=s['base'];sys.path.insert(0,str(R))
from reproduction.h1008_qprobe.management_transport import remote
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
code='import sys,json;sys.path.insert(0,'+repr(B+'/repo')+');from reproduction.prefix20_online.observe import observe;print(json.dumps(observe('+repr(B)+'),allow_nan=False))'
r=json.loads(remote(code,W/('prefix20-'+args.role+'-transport-'+stamp+'.json'),control_path=None,target='h100-8'))
p=W/('prefix20-'+args.role+'-actual-'+stamp+'.json');p.write_text(json.dumps(r,indent=2)+'\n');m=r.get('latest_metrics')or{};l=r.get('latest')or{}
if r.get('native',{}).get('JobId')!=str(s['current_job_id']):raise ValueError('Native job differs from current state; inspect original observation')
c=dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),observed_unix=r['observed_unix'],native=r.get('native'),stage=r.get('stage'),complete_epoch=l.get('epoch',0),actual_adam_per_rank=l.get('actual_adam_steps_per_rank'),metric_epoch=m.get('epoch'),train_accuracy=m.get('train_online_accuracy'),test_accuracy=m.get('raw',{}).get('accuracy'),test_balanced_accuracy=m.get('raw',{}).get('balanced_accuracy'),test_BCE=m.get('raw',{}).get('bce'),test_AUROC=m.get('auroc'),actual_optimizer_learning_rates=m.get('actual_optimizer_learning_rates'),rollout_progress=r.get('current_rollout_progress'),current_rank_actual_updates=r.get('current_rank_actual_updates'),current_train_errors=r.get('current_train_errors'),owned_active_births=r.get('owned_active_births'))
(W/'prefix20-online1000-latest-compact.json').write_text(json.dumps(c,indent=2)+'\n')
print(json.dumps(dict(path=str(p),sha256=c['sha256'],utc=datetime.datetime.fromtimestamp(c['observed_unix'],datetime.timezone.utc).isoformat(),native=c['native'],stage=c['stage'],complete_epoch=c['complete_epoch'],actual_adam_per_rank=c['actual_adam_per_rank'],actual_optimizer_learning_rates=c['actual_optimizer_learning_rates'],rollout_completed=sum((v or{}).get('completed_trajectories',0)for v in (c['rollout_progress']or{}).values()),errors=c['current_train_errors'],test_accuracy=c['test_accuracy'],test_BCE=c['test_BCE'],test_AUROC=c['test_AUROC']),ensure_ascii=False))
