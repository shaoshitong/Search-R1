"""Pair saved pre-selection logits with the actual post-update1024G5 loss; CPU only."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
from reproduction.h1008_qprobe.management_transport import remote

SCRIPT = r'''
import os,sys,json,pathlib,hashlib,math,time,resource
os.environ['CUDA_VISIBLE_DEVICES']='';os.nice(19);resource.setrlimit(resource.RLIMIT_CPU,(60,60))
p=PAYLOAD
def read(path):return json.loads(pathlib.Path(path).read_text())
def sha(path):return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()
def softplus(x):return max(x,0.)+math.log1p(math.exp(-abs(x)))
def lse(xs):
 m=max(xs);return m+math.log(math.fsum(math.exp(x-m)for x in xs))
current=pathlib.Path(p['base']);assert sha(current/'preparation/launch-config.json')==p['config_sha256']
chain=[];seen=set()
for _ in range(16):
 assert current.parent==pathlib.Path('/mnt/data/shaoshitong') and current.resolve()==current and str(current)not in seen
 c=read(current/'preparation/launch-config.json');assert c['run_id']==p['run_id']
 chain.append(current);seen.add(str(current));parent=c.get('resume_joint',{}).get('source_base')
 if not parent:break
 current=pathlib.Path(parent)
else:raise ValueError('Continuation chain exceeds bound')
started=time.time();rows=[]
for expected in p['outers']:
 outer=expected['outer'];matches=[]
 for base in chain:
  root=base/'outers'/('outer-%06d'%outer);joint=root/'complete.json'
  if joint.exists() and sha(joint)==expected['joint_sha256']:matches.append(root)
 assert matches,'No exact committed joint identity for outer%d'%outer
 root=matches[0];joint=read(root/'complete.json');selection_path=root/'probe-score/selection.json'
 selection=read(selection_path);metrics=read(root/'probe-update/metrics.json');labels_path=root/'full/own-em.jsonl'
 assert joint['run_id']==selection['run_id']==metrics['run_id']==p['run_id'] and joint['outer']==selection['outer']==metrics['outer']==outer
 assert metrics==joint['probe_metrics'] and joint['actor']['actual_adam_steps_per_rank']==[20*outer]*8
 assert joint['selection_sha256']==sha(selection_path)==metrics['selection_sha256']
 assert metrics['own_em_sha256']==sha(labels_path) and metrics['actual_adam_steps_per_rank']==[20*outer]*8
 assert metrics['actual_learning_rate']==1e-6 and metrics['new_adam_steps_per_rank']==20
 labels={}
 for line in labels_path.read_text().splitlines():
  label=json.loads(line);tid=label['trajectory_id'];assert tid not in labels
  assert label['actor_version']==selection['actor_version'] and label['selection_sha256']==metrics['selection_sha256']
  assert type(label['strict_em'])is int and label['strict_em']in (0,1);labels[tid]=label
 assert len(selection['selected'])==1024 and len(labels)==5120
 totals=[[],[]];used=set();questions=set();mixed_count=0;predicted_correct=0
 for group in selection['selected']:
  ids=group['trajectory_ids'];z=group['trajectory_logits'];qid=str(group['qid'])
  assert qid not in questions and len(ids)==len(set(ids))==len(z)==5 and not used.intersection(ids)
  questions.add(qid);used.update(ids);assert all(math.isfinite(v)for v in z)
  target=[labels[tid]['strict_em']for tid in ids];assert all(str(labels[tid]['qid'])==qid for tid in ids)
  logp=[-softplus(-v)for v in z];logn=[-softplus(v)for v in z]
  patterns=[math.fsum(logp[j]if bits>>j&1 else logn[j]for j in range(5))for bits in range(32)]
  event=lse(patterns[1:31])-lse([patterns[0],patterns[31]])
  q=math.exp(-softplus(-event));assert abs(q-group['q'])<1e-10
  mixed=int(0<sum(target)<5);mixed_count+=mixed;predicted_correct+=int((q>=.5)==bool(mixed))
  totals[0].extend(softplus(-v)if y else softplus(v)for v,y in zip(z,target))
  totals[1].append(softplus(-event)if mixed else softplus(event))
 assert used==set(labels) and len(questions)==1024
 before=dict(trajectory_bce=math.fsum(totals[0])/5120,group_bce=math.fsum(totals[1])/1024)
 before['total_loss']=before['trajectory_bce']+before['group_bce']
 after=metrics['train'];assert after['count']==1024 and after['group_weight']==1. and after['mixed_count']==mixed_count
 assert abs(after['total_loss']-after['trajectory_bce']-after['group_bce'])<1e-12
 rows.append(dict(outer=outer,run_id=p['run_id'],actor_version=selection['actor_version'],questions=1024,trajectories=5120,
  joint_sha256=expected['joint_sha256'],before_probe_checkpoint_sha256=selection['probe_before_checkpoint_sha256'],
  after_probe_checkpoint_sha256=joint['probe']['sha256'],selection_file_sha256=sha(selection_path),own_em_sha256=sha(labels_path),
  metrics_file_sha256=sha(root/'probe-update/metrics.json'),before=before,after={k:after[k]for k in before},
  delta_after_minus_before={k:after[k]-before[k]for k in before},same_batch_comparison=True,
  actual_probe_updates_per_rank=20,new_generation=False,new_model_forward=False))
print(json.dumps(dict(rows=rows,reader=dict(observed_unix=time.time(),wall_seconds=time.time()-started,
 cpu_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime+resource.getrusage(resource.RUSAGE_SELF).ru_stime,
 torch_imported='torch'in sys.modules,training_modified=False)),allow_nan=False))
'''


def collect(root, maximum_new_outers=4):
    if not 1 <= maximum_new_outers <= 4:
        raise ValueError('At most four new paired scalar proofs per call')
    root=Path(root);out=root/'outputs/search-g1-i1i6-fresh200-20261017'
    target=out/'PROBE_SAME_BATCH_LOSS_PAIRS.json'
    previous=json.loads(target.read_text())if target.exists()else dict(rows=[])
    rows={r['outer']:r for r in previous['rows']}
    facts=json.loads((out/'COMPLETED_ITERATION_SCALAR_EVIDENCE.json').read_text())
    for outer,row in rows.items():
        if row['joint_sha256']!=facts[str(outer)]['joint_sha256']:
            raise ValueError('Previously paired joint identity changed')
    pending=[n for n in sorted(map(int,facts))if n not in rows]
    config_path=out/'CURRENT_CONFIG.json';c=json.loads(config_path.read_text())
    connection=json.loads((root/'work/search-g1-prefix1024-diversity512-20261016/MANAGEMENT_DIRECT_MASTER_CURRENT.json').read_text())
    payload=dict(base=c['base'],run_id=c['run_id'],config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
        outers=[dict(outer=n,joint_sha256=facts[str(n)]['joint_sha256'])for n in pending[:maximum_new_outers]])
    transport=out/('probe-pairs-'+str(time.time_ns())+'-transport.json')
    value=json.loads(remote(SCRIPT.replace('PAYLOAD',repr(payload)),transport,control_path=connection['control_path'],target=c['node'],
                            compact_handshake=connection.get('compact_handshake',False)))
    actual=transport.with_name(transport.name.replace('-transport','-actual'));actual.write_text(json.dumps(value,indent=2)+'\n')
    for row in value['rows']:rows[row['outer']]=row
    result=dict(rows=[rows[n]for n in sorted(rows)],pending_outers=[n for n in pending if n not in rows],reader=value['reader'],evidence=str(actual),
        semantics='Same selected1024G5 labels and saved pre-update logits versus actual post-update full-batch loss. Positive delta means loss worsened. No repeated fitting, new generation or model forward; different-batch20-step curves remain separate.')
    target.write_text(json.dumps(result,indent=2)+'\n')
    for dest in (root/'work/tree-grpo/reproduction/baselines/tree-validation-20261002'/out.name,
                 Path('/Users/shaoshitong/Documents/remote_codex_control/research/search-r1-baselines/results/tree-grpo')/out.name):
        dest.mkdir(parents=True,exist_ok=True);shutil.copy2(target,dest/target.name)
    return dict(complete_outers=sorted(rows),pending=result['pending_outers'],latest=result['rows'][-1]if rows else None,reader=value['reader'])


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--maximum-new-outers',type=int,default=4);a=p.parse_args()
    print(json.dumps(collect(a.root,a.maximum_new_outers)))
