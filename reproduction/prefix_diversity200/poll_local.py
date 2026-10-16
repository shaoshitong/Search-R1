"""Local read-only observation; stores only scalar/status/error evidence."""
import argparse
import json
from pathlib import Path
import time
from reproduction.h1008_qprobe.management_transport import remote


SCRIPT = r'''
import pathlib,sys,json
base=pathlib.Path(BASE);sys.path.insert(0,str(base/'repo'))
from reproduction.prefix_diversity200.launch import observe
value=observe(str(base))
joint_path=base/'latest-joint.json'
if joint_path.exists():
 import hashlib
 joint=json.loads(joint_path.read_text())
 value['latest_joint']={k:joint[k] for k in ('run_id','outer','source_commit','observed_unix','phase_timings')}
 value['latest_joint']['path']=str(joint_path)
 value['latest_joint']['sha256']=hashlib.sha256(joint_path.read_bytes()).hexdigest()
 value['latest_joint']['actor_adam']=joint['actor']['actual_adam_steps_per_rank']
 value['latest_joint']['probe_adam']=joint['probe']['actual_adam_steps_per_rank']
 value['latest_joint']['probe_metrics']={k:v for k,v in joint['probe_metrics'].items()
  if k in ('train','test','actor_selected_train','global_step_loss_curve','train_loop_seconds')}
 value['latest_joint']['actor_selection']={k:v for k,v in joint['actor_selection'].items() if k!='selected'}
value['log_tails']=[{'file':p.name,'tail':p.read_text(errors='replace')[-2200:]}
 for p in sorted((base/'logs').glob('*.log'),key=lambda p:p.stat().st_mtime)[-5:]]
outer=(value.get('stage') or {}).get('outer',0)
if outer:
 root=base/'outers'/('outer-%06d'%outer)
 for name,path in [('actor_selection',root/'actor-selection/selection.json'),
                   ('probe_metrics',root/'probe-update/metrics.json')]:
  if path.exists():
   data=json.loads(path.read_text())
   if name=='actor_selection': data={k:v for k,v in data.items() if k!='selected'}
   value[name]=data
 value['actual_rank_update_ledgers']=[]
 paths=list((base/'actor/outers').glob('outer-%04d-*/rank-*-updates.jsonl'%outer)) if (base/'actor/outers').exists() else []
 paths+=list((root/'probe-update').glob('rank-*-updates.jsonl')) if (root/'probe-update').exists() else []
 for p in paths:
  rows=[json.loads(x) for x in p.read_text().splitlines()]
  value['actual_rank_update_ledgers'].append({'path':str(p),'records':len(rows),
   'first':rows[0] if rows else None,'last':rows[-1] if rows else None})
print(json.dumps(value,allow_nan=False))
'''


def poll(config, receipts):
    c=json.loads(Path(config).read_text()); receipts=Path(receipts)
    path=receipts/('actual-read-'+str(time.time_ns())+'.json')
    result=json.loads(remote(SCRIPT.replace('BASE',repr(c['base'])),path,target=c['node']))
    parsed=path.with_name(path.stem+'-result.json');parsed.write_text(json.dumps(result,indent=2)+'\n')
    (receipts/'LATEST_ACTUAL.json').write_text(json.dumps(result,indent=2)+'\n')
    stage=result.get('stage') or {};joint=result.get('latest_joint') or {}
    summary=dict(evidence=str(parsed),observed_unix=result['observed_unix'],
        job=(result.get('native') or {}).get('JobId'),native=(result.get('native') or {}).get('JobState'),
        stage=stage.get('stage'),outer_in_progress=stage.get('outer'),complete_outer=joint.get('outer',0),
        actor_adam=(result.get('actor') or {}).get('actual_adam_steps_per_rank'),
        probe_adam=(result.get('probe') or {}).get('actual_adam_steps_per_rank'),
        actor_selection=result.get('actor_selection'),gpu=result['gpu_actual']['stdout'])
    return summary


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--receipts',required=True)
    args=p.parse_args();print(json.dumps(poll(args.config,args.receipts),ensure_ascii=False))
