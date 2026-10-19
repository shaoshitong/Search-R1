"""Verify existing greedy fixed512 receipts; never start evaluation or copy rows."""
import argparse,json,time
from pathlib import Path
from .control import OUT,remote
from .shared_observe import BASE

SCRIPT=r'''
import json,time,hashlib,math
from pathlib import Path
p=Path(BASE);target=p/'outers'/('outer-%06d'%OUTER)/'validation/complete.json'
def physical(s):
 prefix='/mnt/data/sihe-share5/workspace/training_data_node_10/shaoshitong/'
 return Path('/mnt/data/shaoshitong')/s[len(prefix):]if s.startswith(prefix)else Path(s)
if not target.exists():
 print(json.dumps(dict(completed=False,actor_outer=OUTER,observed_unix=time.time())))
else:
 raw=target.read_bytes();v=json.loads(raw);rows=[]
 actor=json.loads((target.parent.parent/'complete.json').read_bytes())['actor']
 assert v['actor_model_dir']==actor['hf_model_dir']and v['actor_version']==actor['actor_version']
 assert v['qa_count']==512 and v['actor_version'].endswith('-actor%04d'%OUTER)
 assert v['question_source_sha256']=='3d6318365b5bb98039f1e28ca6bc7d1de98eccc5cc833600e39ce1c2306f9ceb'
 assert v['evaluation_contract']['temperature']==0 and v['evaluation_contract']['n']==1
 assert set(v['actual_worker_exit_codes'])=={str(i)for i in range(8)}
 assert all(x==0 for x in v['actual_worker_exit_codes'].values())
 assert len(v['rank_receipts'])==8
 for rank,ref in enumerate(v['rank_receipts']):
  data=physical(ref['path']).read_bytes();assert hashlib.sha256(data).hexdigest()==ref['sha256']
  r=json.loads(data)
  assert r['rank']==rank and r['qa_count']==64 and len(r['rows'])==64
  for key in('run_id','actor_version','actor_model_dir','evaluation_contract','interaction_protocol','question_source_sha256','actual_compute_dtype'):
   assert r[key]==v[key],key
  assert {row['evaluation_ordinal']for row in r['rows']}==set(range(rank,512,8))
  rows.extend(r['rows'])
 rows.sort(key=lambda row:row['evaluation_ordinal'])
 assert [row['evaluation_ordinal']for row in rows]==list(range(512))and len({row['qid']for row in rows})==512
 assert all(row['em']in(0,1)and math.isfinite(row['f1'])and 0<=row['f1']<=1 for row in rows)
 assert abs(sum(row['em']for row in rows)/512-v['em'])<1e-12
 assert abs(sum(row['f1']for row in rows)/512-v['f1'])<1e-12
 assert {row['data_source']for row in rows}=={'nq','hotpotqa'}
 for source in('nq','hotpotqa'):
  sub=[row for row in rows if row['data_source']==source];assert len(sub)==256
  for key in('em','f1'):assert abs(sum(row[key]for row in sub)/256-v['per_source'][source][key])<1e-12
 result=dict(completed=True,observed_unix=time.time(),actor_outer=OUTER,
  **{key:v[key]for key in('actor_version','actor_model_dir','qa_count','em','f1','per_source','actual_worker_exit_codes','question_source_sha256','evaluation_contract')},
  receipt_sha256=hashlib.sha256(raw).hexdigest(),rank_receipt_sha_verified=True,
  unique_ordinal_qid_verified=True,is_final_51713=False)
 previous=p/'outers/outer-000100/validation/complete.json'
 if OUTER>100 and previous.exists():
  old=json.loads(previous.read_bytes())
  for key in('question_source_sha256','evaluation_contract','interaction_protocol','actual_compute_dtype'):assert old[key]==v[key],key
  old_ids=set()
  for ref in old['rank_receipts']:
   data=physical(ref['path']).read_bytes();assert hashlib.sha256(data).hexdigest()==ref['sha256']
   old_ids.update(row['qid']for row in json.loads(data)['rows'])
  assert old_ids=={row['qid']for row in rows}
  result.update(same_protocol_as_outer100=True,same_actual_question_ids_as_outer100=True,
   outer100_em=old['em'],delta_em_from_outer100_percentage_points=100*(v['em']-old['em']))
 print(json.dumps(result))
'''

def main():
 parser=argparse.ArgumentParser();parser.add_argument('outer',type=int,choices=(100,150,200));n=parser.parse_args().outer
 v=json.loads(remote('BASE='+repr(BASE)+'\nOUTER='+str(n)+'\n'+SCRIPT,
  OUT/('validation%d-%d-transport.json'%(n,time.time_ns())),target='h100-10',established_controller=True))
 root=OUT.parents[1];name=('VALIDATION%d_DIRECT_EM_F1.json'if v['completed']else'VALIDATION%d_PENDING.json')%n
 for dest in(OUT,root/'work/tree-grpo/reproduction/baselines/tree-validation-20261002/b-replica-retained18-20261018',Path('/Users/shaoshitong/Documents/remote_codex_control/research/search-r1-baselines/results/tree-grpo/b-replica-retained18-20261018')):
  (dest/name).write_text(json.dumps(v,ensure_ascii=False,indent=2)+'\n')
 print(json.dumps(v))

if __name__=='__main__':main()
