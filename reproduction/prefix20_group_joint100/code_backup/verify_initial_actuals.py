import json,hashlib,sys,time
from pathlib import Path
b=Path('/mnt/data/shaoshitong/search-g1-question20-groupjoint100-20261016-cee20bcd');sys.path.insert(0,str(b/'repo'))
from reproduction.prefix20_group_joint100.launch import observe
import math
def cosine_learning_rate(t):
 if t==0:return 3e-5
 if t==9999:return 3e-6
 return 3e-6+.5*(3e-5-3e-6)*(1+math.cos(math.pi*t/9999))
x=observe(b);proof=[]
for epoch in (1,2):
 d=b/'train/epochs'/('epoch-%04d'%epoch);metrics=json.loads((d/'metrics.json').read_text());complete=json.loads((d/'complete.json').read_text());ranks=[]
 assert complete['epoch']==epoch and complete['actual_adam_steps_per_rank']==[epoch*100]*8
 for rank in range(8):
  p=d/('rank-%d-updates.jsonl'%rank);data=p.read_bytes();rows=[json.loads(line)for line in data.splitlines()]
  assert len(rows)==100 and [r['epoch_step']for r in rows]==list(range(1,101))
  assert [r['actual_adam_steps']for r in rows]==list(range((epoch-1)*100+1,epoch*100+1))
  assert sum(r['local_trajectories']for r in rows)==3200 and sum(r['local_questions']for r in rows)==640
  assert all(r['actual_learning_rate']==cosine_learning_rate(r['actual_adam_steps']-1)for r in rows)
  assert not(d/('rank-%d-error.json'%rank)).exists()
  ranks.append({'rank':rank,'writes':100,'actual_adam_steps':rows[-1]['actual_adam_steps'],'first_lr':rows[0]['actual_learning_rate'],'last_lr':rows[-1]['actual_learning_rate'],'processed_trajectories':3200,'processed_questions':640,'ledger_sha256':hashlib.sha256(data).hexdigest()})
 proof.append({'epoch':epoch,'source_data_epoch':metrics['source_data_epoch'],'complete':complete,'metrics':metrics,'ranks':ranks,'complete_sha256':hashlib.sha256((d/'complete.json').read_bytes()).hexdigest(),'metrics_sha256':hashlib.sha256((d/'metrics.json').read_bytes()).hexdigest()})
f=b/'preparation/initial-epochs1-2-real-adam-proof-once-v1.json';value={'status':'TWO_COMPLETE_FRESH_JOINT_EPOCHS_ACTUAL8RANK_WRITES_PASS','run_id':'cee20bcd24194b79a9a8a88809bd5886','base':str(b),'job_id':3412,'epochs':proof,'first_initialization':json.loads((b/'train/epochs/epoch-0001/initialization-or-restore.json').read_text()),'large_payload_transferred_to_Mac':False,'model_reloaded':False,'gpu_test_performed':False}
if f.exists():assert json.loads(f.read_text())==value
else:
 with f.open('x')as handle:json.dump(value,handle,sort_keys=True);handle.flush()
print(json.dumps({'observation':x,'first_two_epochs_proof':{'path':str(f),'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'value':value}}))
