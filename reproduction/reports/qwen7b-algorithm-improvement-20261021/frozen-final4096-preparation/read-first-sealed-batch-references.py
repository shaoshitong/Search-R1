import json,pathlib,hashlib,time
b=pathlib.Path('/data/user/jxu957/experiments/dynrpo-qwen25-7b-probe-warmup-20261019/algorithm-improvement-20261021/frozen-final4096-generation-20261022-v1')
x=json.loads((b/'prepared.json').read_text());rows=[]
for entry in x['arrays']:
 if entry['key']not in('confirmation2048-A','confirmation2048-B'):continue
 p=pathlib.Path(entry['original_event_root'])/'sampling/shard-00/batches/batch-000000/complete.json';raw=p.read_bytes();v=json.loads(raw)
 assert v['run_id']==entry['run_id']and v['questions']==128 and v['trajectories']==640 and v['actor_optimizer_updates']==0
 assert v['prefix']['trajectories']==640 and v['full']['trajectories']==640
 for k in ('full','own_em'):
  r=v['full'][k];assert pathlib.Path(r['path']).stat().st_size==r['bytes']
 rows.append(dict(event=entry['key'],manifest=dict(path=str(p),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()),questions=v['questions'],trajectories=v['trajectories'],full_journal=v['full']['full'],own_em_receipt=v['full']['own_em'],complete_epoch=v['epoch']))
print(json.dumps(dict(status='TWO_ACTUAL_REAL_G5_BATCHES_SEALED_NEXT_BATCH_ADVANCING',epoch=time.time(),batches=rows,final_EM_values_or_probe_scores_read=False)))
