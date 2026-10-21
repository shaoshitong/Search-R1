"""Read-only current-run curve/epoch metadata; no model or prediction export."""
import hashlib
import json
import math
from pathlib import Path
import time

base = Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/joint-balanced-native-E4-124194-v1')
config = (base/'config.json').read_bytes()
assert hashlib.sha256(config).hexdigest() == '5aaa63c9348f410357be92ce491ba6d18286d618ab86dae6e63f193f2b4367b6'
cfg = json.loads(config)
assert cfg['steps_per_epoch'] == 3882 and cfg['loss_kwargs'] == dict(trajectory_weight=1.,binary_mixed_weight=1.)
progress = json.loads((base/'progress.json').read_bytes())
path=base/'curve.jsonl';raw=path.read_bytes();complete=raw[:raw.rfind(b'\n')+1]
rows=[json.loads(line)for line in complete.splitlines()]
assert rows and [v['cursor']for v in rows]==list(range(3,rows[-1]['cursor']+1))
assert rows[-1]['cursor'] <= progress['cursor']+1
for row in rows:
    g=row['grouping'];cf=row['question_weighted_confusion']
    assert g['weighted_class_mass_balanced'] and g['source_question_pool_size']==124194 and row['actor_updates']==0
    assert abs(sum(cf.values())-32)<1e-8 and abs(cf['tp']+cf['fn']-16)<1e-8 and abs(cf['tn']+cf['fp']-16)<1e-8
    assert math.isfinite(row['loss']) and abs(row['loss']-row['components']['trajectory']-row['components']['binary_mixed'])<1e-5

def aggregate(selected):
    cf={k:sum(v['question_weighted_confusion'][k]for v in selected)for k in ('tp','fp','fn','tn')}
    return dict(first_update=selected[0]['cursor'],last_update=selected[-1]['cursor'],updates=len(selected),
        epoch=(selected[0]['cursor']-1)//3882+1,
        trajectory_BCE=sum(v['components']['trajectory']for v in selected)/len(selected),
        Mixed_BCE=sum(v['components']['binary_mixed']for v in selected)/len(selected),
        total_loss=sum(v['loss']for v in selected)/len(selected),
        Mixed_balanced_accuracy=.5*(cf['tp']/(cf['tp']+cf['fn'])+cf['tn']/(cf['tn']+cf['fp'])),
        Mixed_recall=cf['tp']/(cf['tp']+cf['fn']),nonMixed_recall=cf['tn']/(cf['tn']+cf['fp']),
        question_weighted_confusion=cf,training_balanced_Mixed_fraction=.5,
        different_training_batches_not_fixed_checkpoint_evaluation=True)

out=dict(epoch=time.time(),job_id='745501',progress_cursor=progress['cursor'],
    curve_snapshot=dict(path=str(path),complete_bytes=len(complete),sha256=hashlib.sha256(complete).hexdigest(),
        last_complete_cursor=rows[-1]['cursor'],first_two_startup_replay_updates_not_in_curve=True),
    current_train_windows=dict(first100=aggregate(rows[:100]),latest100=aggregate(rows[-100:]),
        first500=aggregate(rows[:500]),last500=aggregate(rows[-500:])),
    epoch_train_observed={},test_epoch_results=[],pending_test_epochs=[],
    training_statistics_are_online_preupdate_batches=True,final4096_read=False,training_changed=False)
for ep in range(1,5):
    selected=[v for v in rows if (v['cursor']-1)//3882+1==ep]
    if selected:out['epoch_train_observed'][str(ep)]=dict(aggregate(selected),epoch_complete=selected[-1]['cursor']==ep*3882)
for ep in range(5):
    p=base/'evaluations'/('epoch-%02d'%ep)/'complete.json'
    if not p.exists():out['pending_test_epochs'].append(ep);continue
    payload=p.read_bytes();receipt=json.loads(payload);assert receipt['epoch']==ep and receipt['cursor']==3882*ep
    assert receipt['training_components_unchanged'] and receipt['native_buffers_and_RNG_restored'] and not receipt['final4096_scored']
    result=dict(epoch=ep,receipt=dict(path=str(p),bytes=len(payload),sha256=hashlib.sha256(payload).hexdigest()),states={})
    for state in ('raw','EMA'):
        result['states'][state]={}
        for split,item in receipt['metrics'][state].items():
            loss=item['diagnostic_losses'];pool=item['pool']
            result['states'][state][split]=dict(trajectory_BCE=loss['trajectory_bce'],Mixed_BCE=loss['group_bce'],
                total_loss=loss['trajectory_bce']+loss['group_bce'],
                Mixed_balanced_accuracy=pool['mixed_balanced_accuracy'],classwise=pool['classwise'],
                selected512=item['selected512'],predictions=item['predictions'])
    out['test_epoch_results'].append(result)
print(json.dumps(out,sort_keys=True))
